import numpy as np
import librosa
import os
import xgboost as xgb
from spafe.features.gfcc import gfcc
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def extract_cqt(audio, sr=16000, n_bins=84, bins_per_octave=12, fmin=librosa.note_to_hz('C1')):
    cqt = np.abs(librosa.cqt(audio, sr=sr, n_bins=n_bins, bins_per_octave=bins_per_octave, fmin=fmin))
    return np.mean(cqt, axis=1)

def extract_gfcc(audio, sr=16000, num_ceps=20, num_filters=40):
    gfcc_features = gfcc(audio, fs=sr, num_ceps=num_ceps, nfilts=num_filters)
    return np.mean(gfcc_features, axis=0)

def predict_spoof(audio_path, xgb_model):
    audio, sr = librosa.load(audio_path, sr=16000, duration=60)

    cqt_features = extract_cqt(audio)
    gfcc_features = extract_gfcc(audio)

    combined_features = np.concatenate([
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
    #print("\nAnalysis Result:")
    print(f"Prediction: {prediction}")
    print(f"Spoof probability: {spoof_probability:.4f}")
    
    #print("\nFeature Importance Analysis:")
    
    #cqt_importance = np.sum(feature_importances[:84])
    #gfcc_importance = np.sum(feature_importances[84:])
    
    #total_importance = cqt_importance + gfcc_importance
    
    #print(f"CQT features: {cqt_importance/total_importance:.2%}")
    #print(f"GFCC features: {gfcc_importance/total_importance:.2%}")
    
    #print("\nTop 5 most important individual features:")
    #top_features = np.argsort(feature_importances)[-5:][::-1]
    #for i, feature_idx in enumerate(top_features, 1):
        #feature_type = "CQT" if feature_idx < 84 else "GFCC"
        #print(f"{i}. Feature {feature_idx} ({feature_type}): {feature_importances[feature_idx]/np.sum(feature_importances):.2%}")

def main():
    #print("Audio Spoof Detection Tool")
    print("\nInstructions:")
    print("1. Enter the path to your audio file when prompted.")
    print("2. Type 'quit' to exit the program.")
    print()

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("/home/osaka/jupyter_env/pytorch_env/bin/BASELINE_best_xgboost_asvspoof_model.json")

    while True:
        audio_path = get_audio_path()
        
        if audio_path is None:
            break

        try:
            prediction, spoof_probability, feature_importances, num_features = predict_spoof(audio_path, xgb_model)
            print_result(prediction, spoof_probability, feature_importances, num_features)
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    print("\nThank you for using the Audio Spoof Detection Tool. Goodbye!")

if __name__ == "__main__":
    main()