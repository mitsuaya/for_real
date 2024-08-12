import torch
import torch.nn as nn
from transformers import AutoProcessor, WavLMModel
import librosa
import numpy as np
import xgboost as xgb
from spafe.features.gfcc import gfcc
import os

#just for checking if detected yung gpu ko , pwede nyo icomment if wala naman kayo gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#eto yung WavLM model and processor 
model_name = "patrickvonplaten/wavlm-libri-clean-100h-base-plus"
#yung processor yung responsible sa pag coconvert nung audio into tensors para magamit ng wavLM
processor = AutoProcessor.from_pretrained(model_name)

#eto yung added na attention layer kasi diba custom yung WavLM natin may attention layer na extra
class AttentionLayer(nn.Module):
    #initialization lang ng structure nung attention layer
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    #eto naman yung mismong nagcoconvert ng mga hidden states into weights
    # hidden states yung tawag sa mga output ng each layer 
    #ang goal nito ay mag assign ng weights sa certain features na relevant/important
    def forward(self, hidden_states):
        attention_weights = self.attention(hidden_states).squeeze(-1)
        attention_weights = torch.softmax(attention_weights, dim=-1)
        attended_hidden = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)
        return attended_hidden

#eto na yung custom wavlm model natin with added attention layer
class CustomWavLMForFeatureExtraction(nn.Module):
    def __init__(self):
        super(CustomWavLMForFeatureExtraction, self).__init__()
        # model for feature extraction
        self.wavlm = WavLMModel.from_pretrained(model_name)
        # the additional attention layer for assigning weights
        self.attention = AttentionLayer(self.wavlm.config.hidden_size)

    #parang yung forward function lang din sa nauna yung concept neto  see line 28
    def forward(self, input_values, attention_mask=None):
        outputs = self.wavlm(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        attended_hidden = self.attention(hidden_states)
        return attended_hidden

#weirdly enough eto na lahat nung cqt feature extraction , madali lang sya iextract same with gfcc 
#yung specifics makikita nyo sa parameter space below (eg. n_bins - number of frequency bins , etc.)
#fmin is minimum frequency , default (ata) yung C1 (around 32.7hz daw idk)
#librosa yung gamit natin dito 
def extract_cqt(audio, sr=16000, n_bins=84, bins_per_octave=12, fmin=librosa.note_to_hz('C1')):
    #np.abs para makuha yung magnitude spectrum , will test more kung eto ba talaga yung tamang method
    cqt = np.abs(librosa.cqt(audio, sr=sr, n_bins=n_bins, bins_per_octave=bins_per_octave, fmin=fmin))
    #para yung originally 2d array maging 1d array , kinocompute yung mean na nagrerepresent sa average energy in each frequency bin
    #sa librosa axis =1 yung time 
    return np.mean(cqt, axis=1)

#pang extract ng gfcc , spafe yung library na gamit , yung parameters default yun
def extract_gfcc(audio, sr=16000, num_ceps=20, num_filters=40):
    gfcc_features = gfcc(audio, fs=sr, num_ceps=num_ceps, nfilts=num_filters)
    #same reason sa cqt although in this case axis 0 yung representation ng time
    return np.mean(gfcc_features, axis=0)

#yung actual na nagprepredict
def predict_spoof(audio_path, model, xgb_model):
    #load audio , yung duration pwede sya baguhin based sa samples , for example if mga sample mo ay up to 20 secs long pwede gawin to 20secs
    audio, sr = librosa.load(audio_path, sr=16000, duration=15)
    # kinoconvert yung input (audio) into tensors para magamit ng wavlm , yung inputvalues to ibig sabihin inaassign sa gpu , pag walang gpu default to sa cpu
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(device)

    #Pagkuha lang nung WavLM features , tinatawag lang yung function and then convert into numpy array
    with torch.no_grad():
        wavlm_features = model(input_values).cpu().numpy()

    #function calls lang din to extract features
    cqt_features = extract_cqt(audio)
    gfcc_features = extract_gfcc(audio)

    #eto na yung pagcoconcat ng features into one output , in this case to numpy array
    #kung tama ang alala ko ang output neto ay 1 column and rows na nag aadjust magisa to accomodate the required rows
    #magiging 2D array with the shape of (n,1) where n is the number of rows tapos 1 column
    combined_features = np.concatenate([
        wavlm_features.squeeze(),
        cqt_features,
        gfcc_features
    ]).reshape(1, -1)

    #you can uncomment the line below if gusto nyo makita yung output nung combined features
    #print(combined_features.shape)

    #uses the xgboost model to predict on the combined features
    prediction_proba = xgb_model.predict_proba(combined_features)[0]
    #kinukuha yung probability na yung sample ay spoof , pag [0] yung probability na bonafide
    spoof_probability = prediction_proba[1]

    #eto yung prinopose ko na hyperparameter , yung decision threshold , at this time yung default lang muna ginamit ko 
    prediction = "spoof" if spoof_probability > 0.5 else "bonafide"

    #para makita lang yung important features sa pagpredict
    feature_importances = xgb_model.feature_importances_
    
    return prediction, spoof_probability, feature_importances, combined_features.shape[1]

#kung ano lang din yung nakalagay , mga print statements nalang to at function calls , although meron din ako nilagay na 
#pinapakita yung mga features na nakapagcontribute ng malaki into the prediction
def get_audio_path():
    while True:
        audio_path = input("Enter the path to your audio file (or 'quit' to exit): ")
        if audio_path.lower() == 'quit':
            return None
        if os.path.exists(audio_path):
            return audio_path
        else:
            print(f"Error: File '{audio_path}' not found. Please try again.")

def print_result(prediction, spoof_probability, feature_importances, num_features):
    print("\nAnalysis Result:")
    print(f"Prediction: {prediction}")
    print(f"Spoof probability: {spoof_probability:.4f}")
    
    print("\nFeature Importance Analysis:")
    
    wavlm_importance = np.sum(feature_importances[:768])
    cqt_importance = np.sum(feature_importances[768:768+84])
    gfcc_importance = np.sum(feature_importances[768+84:])
    
    total_importance = wavlm_importance + cqt_importance + gfcc_importance
    
    print(f"WavLM features: {wavlm_importance/total_importance:.2%}")
    print(f"CQT features: {cqt_importance/total_importance:.2%}")
    print(f"GFCC features: {gfcc_importance/total_importance:.2%}")
    
    print("\nTop 5 most important individual features:")
    top_features = np.argsort(feature_importances)[-5:][::-1]
    for i, feature_idx in enumerate(top_features, 1):
        feature_type = "WavLM" if feature_idx < 768 else "CQT" if feature_idx < 768+84 else "GFCC"
        print(f"{i}. Feature {feature_idx} ({feature_type}): {feature_importances[feature_idx]/np.sum(feature_importances):.2%}")

def main():
    print("Audio Spoof Detection Tool")
    print("\nInstructions:")
    print("1. Enter the path to your audio file when prompted.")
    print("2. The tool will analyze the audio and provide a prediction with explanations.")
    print("3. Type 'quit' to exit the program.")
    print()

    model = CustomWavLMForFeatureExtraction().to(device)
    
    
    #eto yung pagloaload ng weights para sa wavLM model
    pretrained_dict = torch.load("best_wavlm_asvspoof_model.pth", map_location=device)
    model_dict = model.state_dict()
    #eto naman yung pagfilter out ng mga bagay na di natin kailangan diba kasi wala tayong classifier sa wavlm model
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    #eval mode ng xgboost , aka frozen weights para puro prediction lang
    model.eval()

    #yung xgboost model mismo
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("misa-v2.json")

    while True:
        audio_path = get_audio_path()
        
        if audio_path is None:
            break

        try:
            prediction, spoof_probability, feature_importances, num_features = predict_spoof(audio_path, model, xgb_model)
            print_result(prediction, spoof_probability, feature_importances, num_features)
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    print("\nThank you for using the Audio Spoof Detection Tool. Goodbye!")

if __name__ == "__main__":
    main()