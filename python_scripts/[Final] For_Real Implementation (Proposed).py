import joblib
import torch
import torch.nn as nn
import librosa
import numpy as np
from transformers import AutoProcessor, WavLMModel
from spafe.features.gfcc import gfcc

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

class CustomWavLMForFeatureExtraction(nn.Module):
    def __init__(self, model_name):
        super(CustomWavLMForFeatureExtraction, self).__init__()
        self.wavlm = WavLMModel.from_pretrained(model_name)
        self.attention = AttentionLayer(self.wavlm.config.hidden_size)

    def forward(self, input_values, attention_mask=None):
        outputs = self.wavlm(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        attended_hidden = self.attention(hidden_states)
        return attended_hidden

model_data = joblib.load('output_models/clean2percent.joblib')
calibrated_model = model_data['calibrated_model']
selector = model_data['selector']

#eto yung pinaka okay na threshold based sa testing ko sa real world data aka samples from youtube pero applicable lang sya for this model exactly
best_threshold = 0.89

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "patrickvonplaten/wavlm-libri-clean-100h-base-plus"
processor = AutoProcessor.from_pretrained(model_name)

wavlm_model = CustomWavLMForFeatureExtraction(model_name).to(device)

state_dict = torch.load('/home/osaka/jupyter_env/pytorch_env/bin/best_wavlm_asvspoof_model.pth', map_location=device)
keys_to_remove = ["classifier.weight", "classifier.bias"]
for key in keys_to_remove:
    state_dict.pop(key, None)

wavlm_model.load_state_dict(state_dict, strict=False)
wavlm_model.eval()

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
    return cqt_features, gfcc_features

def monte_carlo_dropout(model, input_values, num_samples=10):
    model.train()  
    mc_samples = []
    for _ in range(num_samples):
        with torch.no_grad():
            mc_samples.append(model(input_values).unsqueeze(0))
    mc_samples = torch.cat(mc_samples, dim=0)
    mean = torch.mean(mc_samples, dim=0)
    variance = torch.var(mc_samples, dim=0)
    return mean, variance

def predict_sample(file_path):
    audio, sr = librosa.load(file_path, sr=16000, duration=30)
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(device)

    wavlm_features, wavlm_uncertainty = monte_carlo_dropout(wavlm_model, input_values)

    cqt_features, gfcc_features = extract_handcrafted_features(audio)

    wavlm_weight, cqt_weight, gfcc_weight = 0.4, 0.3, 0.3
    
    weighted_wavlm = wavlm_features.cpu().numpy().flatten() * wavlm_weight
    weighted_cqt = cqt_features.flatten() * cqt_weight
    weighted_gfcc = gfcc_features.flatten() * gfcc_weight
    wavlm_uncertainty = wavlm_uncertainty.cpu().numpy().flatten()
    
    combined_feature = np.concatenate([weighted_wavlm, weighted_cqt, weighted_gfcc, wavlm_uncertainty])

    selected_features = selector.transform(combined_feature.reshape(1, -1))

    prediction_proba = calibrated_model.predict_proba(selected_features)[0, 1]
    prediction = 1 if prediction_proba >= best_threshold else 0

    return prediction, prediction_proba

def main():
    while True:
        file_path = input("\nEnter the path to the audio file: ")

        if file_path.lower() == 'q':
                break
        
        try:
            prediction, probability= predict_sample(file_path)
            
            print("\nPrediction Results:")
            print(f"Prediction: {'Spoof' if prediction == 1 else 'Bonafide'}")
            print(f"Samples with probabilities higher than our threshold will be considered as Spoof/Deepfake")
            print(f"Our Threshold : {best_threshold}")
            print(f"Probability: {probability:.4f}")
            

        except Exception as e:
            print(f"Error processing the file: {str(e)}")

if __name__ == "__main__":
    main()
