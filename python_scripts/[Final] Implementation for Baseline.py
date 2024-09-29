import numpy as np
import librosa
import xgboost as xgb
from spafe.features.gfcc import gfcc
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

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

def predict_spoof(audio_path, xgb_model):
    audio, sr = librosa.load(audio_path, sr=16000, duration=60)

    cqt_features, gfcc_features = extract_handcrafted_features(audio)

    combined_features = np.concatenate([
        cqt_features,
        gfcc_features
    ]).reshape(1, -1)

    prediction_proba = xgb_model.predict_proba(combined_features)[0]
    spoof_probability = prediction_proba[1]
    prediction = "spoof" if spoof_probability > 0.5 else "bonafide"

    feature_importances = xgb_model.feature_importances_
    
    return prediction, spoof_probability, feature_importances, combined_features.shape[1]

def print_result(audio_file, prediction, spoof_probability):
    print(f"File: {audio_file}")
    print(f"Prediction: {prediction}")
    print(f"Spoof probability: {spoof_probability:.4f}")
    print()

def main():
    print("Audio Spoof Detection Tool")
    
    # Load the model
    xgb_model = xgb.XGBClassifier()
    model_path = "/home/osaka/jupyter_env/pytorch_env/BASELINE_XGBoostCLEAN_Optuna.json"
    xgb_model.load_model(model_path)
    
    while True:
        audio_path = input("\nEnter the path to an audio file (or 'q' to quit): ")
        
        if audio_path.lower() == 'q':
            break
        
        try:
            prediction, spoof_probability, _, _ = predict_spoof(audio_path, xgb_model)
            print_result(audio_path, prediction, spoof_probability)
        except Exception as e:
            print(f"An error occurred processing {audio_path}: {str(e)}")
            print()

    print("\nThank you for using the Audio Spoof Detection Tool.")

if __name__ == "__main__":
    main()