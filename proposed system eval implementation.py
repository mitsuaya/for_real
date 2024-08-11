import torch
import torch.nn as nn
from transformers import AutoProcessor, WavLMModel
import librosa
import numpy as np
import xgboost as xgb
from spafe.features.gfcc import gfcc
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "patrickvonplaten/wavlm-libri-clean-100h-base-plus"
processor = AutoProcessor.from_pretrained(model_name)

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
    def __init__(self):
        super(CustomWavLMForFeatureExtraction, self).__init__()
        self.wavlm = WavLMModel.from_pretrained(model_name)
        self.attention = AttentionLayer(self.wavlm.config.hidden_size)

    def forward(self, input_values, attention_mask=None):
        outputs = self.wavlm(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        attended_hidden = self.attention(hidden_states)
        return attended_hidden

def extract_cqt(audio, sr=16000, n_bins=84, bins_per_octave=12, fmin=librosa.note_to_hz('C1')):
    cqt = np.abs(librosa.cqt(audio, sr=sr, n_bins=n_bins, bins_per_octave=bins_per_octave, fmin=fmin))
    return np.mean(cqt, axis=1)

def extract_gfcc(audio, sr=16000, num_ceps=20, num_filters=40):
    gfcc_features = gfcc(audio, fs=sr, num_ceps=num_ceps, nfilts=num_filters)
    return np.mean(gfcc_features, axis=0)

def predict_spoof(audio_path, model, xgb_model):
    audio, sr = librosa.load(audio_path, sr=16000, duration=15)
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(device)

    with torch.no_grad():
        wavlm_features = model(input_values).cpu().numpy()

    cqt_features = extract_cqt(audio)
    gfcc_features = extract_gfcc(audio)

    combined_features = np.concatenate([
        wavlm_features.squeeze(),
        cqt_features,
        gfcc_features
    ]).reshape(1, -1)

    prediction_proba = xgb_model.predict_proba(combined_features)[0]
    spoof_probability = prediction_proba[1]
    prediction = "spoof" if spoof_probability > 0.5 else "bonafide"

    feature_importances = xgb_model.feature_importances_
    
    return prediction, spoof_probability, feature_importances, combined_features.shape[1]

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
    
    # Load pre-trained weights with filtering
    pretrained_dict = torch.load("best_wavlm_asvspoof_model.pth", map_location=device)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    model.eval()

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