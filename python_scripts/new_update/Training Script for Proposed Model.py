#proposed with calibration , tuning , and manual threshold adjustment
#di ko na nalinis masyado ,baka may mga function na hindi nagamit , ireference nalang yung main function
import torch 
import torch.nn as nn
from transformers import AutoProcessor, WavLMModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import librosa
import os
import numpy as np
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from tqdm import tqdm
import joblib
import logging
from spafe.features.gfcc import gfcc
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import optuna
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE

#imbis na print eto yung ginagamit dito , initialization lang to tho
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#kung anong device yung gagamitin , cuda nakalagay dito kasi nvidia gpu gamit ko , pwinepwersa ko sya sa gpu magtrain , although pwede to palitan ng 'cpu' para sa cpu
device = torch.device("cuda")

#initialization lang nung wavlm , ang gamit dito ay wavlm na finetuned sa 100h ng libri clean , eto din yung gamit nila sa mga sample sa official documentation kaya eto nadin ginamit ko
model_name = "patrickvonplaten/wavlm-libri-clean-100h-base-plus"
processor = AutoProcessor.from_pretrained(model_name)

#yung attention layer
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, hidden_states):
        attention_weights = self.attention(hidden_states).squeeze(-1)
        attention_weights = torch.softmax(attention_weights, dim=-1)
        attended_hidden = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)
        return attended_hidden

#yung custom na wavlm model , custom dahil dun sa additional attention layer
class CustomWavLMForFeatureExtraction(nn.Module):
    def __init__(self, pretrained_weights_path):
        super(CustomWavLMForFeatureExtraction, self).__init__()
        
        self.wavlm = WavLMModel.from_pretrained(model_name)
        
        #tinanggal dito yung mga variables/keys na hindi nagamit , specifically yung classifier ng wavlm inexclude sya para hindi probability ang ilabas
        used_keys = ['feature_extractor', 'feature_projection', 'encoder']
        filtered_state_dict = {k: v for k, v in self.wavlm.state_dict().items() if any(uk in k for uk in used_keys)}
        
        self.wavlm.load_state_dict(filtered_state_dict, strict=False)
        
        self.attention = AttentionLayer(self.wavlm.config.hidden_size)
        
        pretrained_dict = torch.load(pretrained_weights_path, map_location=device)
        model_dict = self.state_dict()
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        
        print(f"Loaded {len(pretrained_dict)} / {len(model_dict)} keys from fine-tuned weights")

    def forward(self, input_values, attention_mask=None):
        outputs = self.wavlm(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        attended_hidden = self.attention(hidden_states)
        return attended_hidden

#eto naman dataset class lang -> load dataset -> convert labels into binary 0/1 -> find the audio files , ang nirereturn neto ay yung audio file at yung corresponding label
class CustomDataset(Dataset):
    def __init__(self, csv_file, base_path, target_sr=16000, add_extension=True):
        self.df = pd.read_csv(csv_file).sample(frac=0.2)
        self.base_path = base_path
        self.target_sr = target_sr
        self.label_map = {'bonafide': 0, 'spoof': 1}
        self.add_extension = add_extension

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row['audio_id']
        if self.add_extension:
            file_name += ".flac"
        
        file_path = os.path.join(self.base_path, file_name)
        
        if not os.path.exists(file_path) and self.add_extension:
            file_path = os.path.join(self.base_path, row['audio_id'])
        
        audio, _ = librosa.load(file_path, sr=self.target_sr)
        label = self.label_map[row['label']]
        return audio, label

#eto naman yung pangconvert ng audio and labels into tensors para maprocess ng wavlm , sa pagkakaalam ko tensors lang ang binabasa ng wavlm 
def collate_fn(batch):
    audios, labels = zip(*batch)
    inputs = processor(audios, sampling_rate=16000, return_tensors="pt", padding=True)
    return inputs.input_values, torch.tensor(labels, dtype=torch.long)

#eto naman yung handcrafted features, CQT at GFCC , ineextract tapos iniistack sa one dimension 
def extract_handcrafted_features(audio, sr=16000):
    cqt = np.abs(librosa.cqt(audio, sr=sr, n_bins=84, bins_per_octave=12, fmin=librosa.note_to_hz('C1')))
    cqt_features = np.hstack([
        np.mean(cqt, axis=1),
        np.std(cqt, axis=1),
        np.max(cqt, axis=1),
        np.min(cqt, axis=1)
    ])

    gfcc_features = gfcc(audio, fs=sr, num_ceps=20, nfilts=40)
    gfcc_features = np.hstack([
        np.mean(gfcc_features, axis=0),
        np.std(gfcc_features, axis=0),
        np.max(gfcc_features, axis=0),
        np.min(gfcc_features, axis=0)
    ])

    combined_features = np.hstack([cqt_features, gfcc_features])
    return combined_features

#function para maextract yung features  , dito din kinocombine yung wavlm + cqt and gfcc
def extract_features(model, data_loader):
    model.eval()
    all_features, all_labels = [], []
    
    with torch.no_grad():
        for input_values, batch_labels in tqdm(data_loader, desc="Extracting features"):
            input_values = input_values.to(device)
            wavlm_features_batch = model(input_values).cpu().numpy()
            
            for i in range(input_values.shape[0]):
                audio = input_values[i].cpu().numpy()
                robust_feature = extract_handcrafted_features(audio)
                
                combined_feature = np.hstack([wavlm_features_batch[i], robust_feature])
                all_features.append(combined_feature)
            
            all_labels.extend(batch_labels.numpy())

        torch.cuda.empty_cache()
    return np.array(all_features), np.array(all_labels)

#eto naman yung parameters para sa hyperparameter optimization , range yung values na nasa loob , bale randomly pumipili ng value sa loob ng range , para ibat ibang config yung matetest , kung ilang beses , according sa kung ilang trials
def objective(trial, X_train, y_train):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 0.5),
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-5, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 10.0),
        'scale_pos_weight': trial.suggest_uniform('scale_pos_weight', 0.1, 20.0)
    }
    
    #pwinepwersa ko ulit dito na yung hyperparameter optimization process ay sa gpu magrun
    model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss', tree_method='gpu_hist', gpu_id=0)
    
    #yung training data ay iniisplit into 5 folds  tapos ang bawat fold ay nag aact as both training at validation dataset , ang goal dito ay hanapin yung best roc-auc score (pinakamalapit sa EER na available na metric)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict_proba(X_val_fold)[:, 1]
        score = roc_auc_score(y_val_fold, y_pred)
        scores.append(score)
    
    return np.mean(scores)

#eto yung calibration function , ang ginagawa lang nito niloaload yung trained model , pinapapredict yung validation set , tapos kinocorrect ,
#iba sya sa finetuning dahil walang nababago sa parameters ng actual model during calibration , ang nagaadjust ay yung predicted probabilities , yung mismong values sa pagcocompute nya ng prediction , treated na to as post processing iirc
def calibrate_scores(model, X_val, y_val):
    calibrated_model = CalibratedClassifierCV(model, cv='prefit', method='sigmoid')
    calibrated_model.fit(X_val, y_val)
    return calibrated_model

#evaluation lang using 0.5 threshold
def evaluate_model(model, X, y, threshold=0.5):
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    balanced_acc = balanced_accuracy_score(y, y_pred)
    
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = roc_auc_score(y, y_pred_proba)
    
    average_precision = average_precision_score(y, y_pred_proba)
    
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    logging.info(f"Threshold: {threshold:.4f}")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1-score: {f1:.4f}")
    logging.info(f"Balanced Accuracy: {balanced_acc:.4f}")
    logging.info(f"False Positive Rate: {fp / (fp + tn):.4f}")
    logging.info(f"ROC AUC: {roc_auc:.4f}")
    logging.info(f"Average Precision: {average_precision:.4f}")
    logging.info(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

#hindi provided yung EER sa sklearn kaya need pa gawin to para macompute yung EER
def calculate_eer(y_true, y_scores, threshold=None):
    if threshold is None:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    else:
        y_pred = (y_scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        eer = (fpr + fnr) / 2
    return eer

#eto na yung threshold adjustment , in addition to calibration may ganto pa
def threshold_adjustment(y_true, y_scores, calibrated_model, val_scaled_features):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    logging.info(f"ROC AUC: {roc_auc:.4f}")

    logging.info("\nThreshold | EER")
    logging.info("-" * 20)

    #ang meaning lang nito , from thresholds 0.01 to 0.99 kukuha ka ng 200 na threshold randomly tapos gagamitin 
    custom_thresholds = np.linspace(0.01, 0.99, 200)
    eer_values = []

    #sa bawat threshold dun sa 200 , tinetest isa isa tapos kinukuha yung EER kinocompare yung ground truth at predicted scores (probability na ang sample ay spoof)
    for threshold in custom_thresholds:
        eer = calculate_eer(y_true, y_scores, threshold)
        eer_values.append((threshold, eer))

    #after nun , kinukuha lahat ng threshold at corresponding EER , tapos nirarank based sa lowest EER , pwedeng ibahin kung ilan pero dito 20 lang yung dinidisplay ko
    top_thresholds = sorted(eer_values, key=lambda x: x[1])[:20]
    for threshold, eer in top_thresholds:
        logging.info(f"{threshold:.4f} | {eer:.4f}")

    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer_threshold = interp1d(fpr, thresholds)(eer)
    logging.info(f"\nEqual Error Rate (EER): {eer:.4f}")
    logging.info(f"EER Threshold: {eer_threshold:.4f}")

    #tapos eto yung part na ginawa ko para maadjust ko manually yung decision threshold
    while True:
        user_input = input("\nEnter a threshold to evaluate (or type 'save' to finish): ").strip()
        if user_input.lower() == 'save':
            break
        try:
            custom_threshold = float(user_input)
            logging.info("\nEvaluation with custom threshold:")
            evaluate_model(calibrated_model, val_scaled_features, y_true, threshold=custom_threshold)
        except ValueError:
            print("Invalid input. Please enter a valid number or 'save'.")

    return float(input("Enter the final threshold you'd like to use: "))


#puro function call nalang to
def main():
    # Load and preprocess data
    train_dataset = CustomDataset('/home/osaka/jupyter_env/pytorch_env/bin/asvspoof2019_train_balanced.csv', '/home/osaka/jupyter_env/pytorch_env/bin/ASVspoof2019_LA_train/flac/')
    val_dataset = CustomDataset('/run/media/osaka/84920499920491BA/For_Real Dataset/Samples/All Samples/For_Realv2.csv', '/run/media/osaka/84920499920491BA/For_Real Dataset/Samples/All Samples/split/')
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # Extract features
    pretrained_weights_path = '/home/osaka/jupyter_env/pytorch_env/bin/best_wavlm_asvspoof_model.pth'
    wavlm_model = CustomWavLMForFeatureExtraction(pretrained_weights_path).to(device)
    train_features, train_labels = extract_features(wavlm_model, train_loader)
    val_features, val_labels = extract_features(wavlm_model, val_loader)
    
    # Free up GPU memory
    del wavlm_model
    torch.cuda.empty_cache()
    
    # Scale features
    scaler = StandardScaler()
    train_scaled_features = scaler.fit_transform(train_features)
    val_scaled_features = scaler.transform(val_features)
    
    # Apply SMOTE to balance the training data
    smote = SMOTE(random_state=42)
    train_scaled_features, train_labels = smote.fit_resample(train_scaled_features, train_labels)
    
     # Hyperparameter tuning with Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, train_scaled_features, train_labels), n_trials=10)  # Increased number of trials
    best_params = study.best_params
    
    # Train the best model
    best_model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss', tree_method='gpu_hist', gpu_id=0)
    best_model.fit(train_scaled_features, train_labels)
    
    # Calibrate the model
    calibrated_model = calibrate_scores(best_model, val_scaled_features, val_labels)
    
    # Calculate scores
    y_scores = calibrated_model.predict_proba(val_scaled_features)[:, 1]
    
    # Use the new threshold_adjustment function
    final_threshold = threshold_adjustment(val_labels, y_scores, calibrated_model, val_scaled_features)
    
    # Final evaluation with the user-selected threshold
    logging.info("\nFinal evaluation with user-selected threshold:")
    evaluate_model(calibrated_model, val_scaled_features, val_labels, threshold=final_threshold)
    
    # Save the calibrated model
    joblib.dump({
        'calibrated_model': calibrated_model,
        'scaler': scaler,
        'final_threshold': final_threshold
    }, 'misa-v4-nkz2r32.joblib')
    logging.info("Calibrated model, preprocessing objects, and final threshold saved successfully.")

if __name__ == "__main__":
    main()
