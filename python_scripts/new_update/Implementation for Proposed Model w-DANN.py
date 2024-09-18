import torch
import torch.nn as nn
import librosa
import numpy as np
import joblib
from transformers import AutoProcessor, WavLMModel
from spafe.features.gfcc import gfcc
from scipy.signal import resample


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

class DomainAdversarialNetwork(nn.Module):
    def __init__(self, feature_dim):
        super(DomainAdversarialNetwork, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        return features


saved_model = joblib.load('misa-v8-singlefire.joblib')
calibrated_model = saved_model['calibrated_model']
scaler = saved_model['scaler']
dan_model = saved_model['dan_model']
final_threshold = saved_model['final_threshold']
offset = 0.0010
offset_threshold = saved_model['final_threshold'] - offset



model_name = "patrickvonplaten/wavlm-libri-clean-100h-base-plus"
processor = AutoProcessor.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wavlm_model = CustomWavLMForFeatureExtraction(model_name).to(device)


state_dict = torch.load('/home/osaka/jupyter_env/pytorch_env/bin/best_wavlm_asvspoof_model.pth', map_location=device)
keys_to_remove = ["classifier.weight", "classifier.bias"]
for key in keys_to_remove:
    state_dict.pop(key, None)


wavlm_model.load_state_dict(state_dict, strict=False)
wavlm_model.eval()

def extract_robust_features(audio, sr=16000):
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
    return combined_features.reshape(1, -1)  

def process_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    max_length = 60 * sr
    if len(audio) > max_length:
        audio = audio[:max_length]
    elif len(audio) < max_length:
        audio = np.pad(audio, (0, max_length - len(audio)), 'constant')
    
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
    with torch.no_grad():
        wavlm_features = wavlm_model(**inputs).detach().cpu().numpy()
    
    if wavlm_features.ndim == 3:
        wavlm_features = wavlm_features.squeeze(0)  
    elif wavlm_features.ndim == 1:
        wavlm_features = wavlm_features.reshape(1, -1)  
    
    robust_features = extract_robust_features(audio)
    
    if robust_features.ndim == 1:
        robust_features = robust_features.reshape(1, -1)
    

    combined_features = np.hstack((wavlm_features, robust_features))
    
    with torch.no_grad():
        adapted_features = dan_model(torch.FloatTensor(combined_features).to(device)).detach().cpu().numpy()
    
    scaled_features = scaler.transform(adapted_features)
    
    return scaled_features

def predict_sample(file_path):
    features = process_audio(file_path)
    
    probability = calibrated_model.predict_proba(features)[0, 1]
    prediction = "spoof" if probability >= offset_threshold else "bonafide"
    
    return prediction, probability



def main():
    while True:
        file_path = input("Enter the path to the audio file (or 'q' to quit): ")
        if file_path.lower() == 'q':
            break
        
        try:
            prediction, probability = predict_sample(file_path)
            print(f"Prediction: {prediction}")
            print(f"Probability of being spoof: {probability:.4f}")
            print(f"Threshold: {offset_threshold:.4f}")
        except Exception as e:
            print(f"Error processing file: {e}")
        
        print()

if __name__ == "__main__":
    main()