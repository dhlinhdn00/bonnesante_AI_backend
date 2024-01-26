import numpy as np
import pywt
import cv2
import torch
from scipy.signal import find_peaks
from skorch import NeuralNetClassifier
from ecg_classification.main import MyModule, visualize_processed_ecg_signals

class_mapping = {
    0: "Normal",
    1: "Supraventricular Ectopic Beat",
    2: "Ventricular Ectopic Beat",
    3: "Fusion Beat"
}

def predict_ecg(file_path, model_path, wavelet="mexh"):
    def read_ecg_data(file_path):
        with open(file_path, 'r') as file:
            ecg_data = np.array([float(line.strip()) for line in file])
        return ecg_data

    def find_r_peaks(ecg_signal, sampling_rate=360):
        distance = int(sampling_rate * 0.6)  # Approximate distance between R-peaks
        peaks, _ = find_peaks(ecg_signal, distance=distance)
        return peaks

    def preprocess_ecg_data(ecg_segment, wavelet, scales):
        coeffs, _ = pywt.cwt(ecg_segment, scales, wavelet)
        ecg_processed = cv2.resize(coeffs, (100, 100))
        ecg_processed = np.expand_dims(ecg_processed, axis=0)  # Add channel dimension
        ecg_processed = np.expand_dims(ecg_processed, axis=0)  # Add batch dimension
        return ecg_processed.astype(np.float32)

    def load_model(model_path, wavelet):
        model = NeuralNetClassifier(MyModule)
        model.initialize()
        model.load_params(f_params=model_path.format(wavelet))
        return model

    # Read ECG data
    ecg_data = read_ecg_data(file_path)
    # print(len(ecg_data))

    # Detect R-peaks
    r_peaks = find_r_peaks(ecg_data)
    # print(r_peaks)

    # Preprocess ECG data
    sampling_rate = 360
    scales = pywt.central_frequency(wavelet) * sampling_rate / np.arange(1, 101, 1)
    preprocessed_data = [preprocess_ecg_data(ecg_data[max(0, r - 90):r + 110], wavelet, scales) for r in r_peaks if r + 110 <= len(ecg_data)]

    # Load the trained model
    model = load_model(model_path, wavelet)

    # Make predictions and map to class names
    predictions = [model.predict(segment) for segment in preprocessed_data]
    mapped_predictions = [class_mapping[pred[0]] for pred in predictions]

    return predictions, mapped_predictions

def classify():
    file_path = "inference_array.txt"
    model_path = "./models/model_{}.pkl"
    predictions, mapped_predictions = predict_ecg(file_path, model_path)

    overall_pred = []
    for i, pred in enumerate(predictions):
        print(f"R-peak {i+1}: {pred}")
        overall_pred.extend(pred) 

    for i, mapped_pred in enumerate(mapped_predictions):
        print(f"R-peak {i+1}: {mapped_pred}")

    avg_pred = np.mean([float(p) for p in overall_pred])  
    avg_pred = np.rint(avg_pred)

    mapped_pred = class_mapping[avg_pred]
    print(mapped_pred)

def classify_for_app(file_path):
    model_path = "./ecg_classification/models/model_{}.pkl"
    predictions, mapped_predictions = predict_ecg(file_path, model_path)
    

    return mapped_predictions


if __name__ == "__main__":
    a = classify_for_app("./inference_array.txt")
    print(a)
