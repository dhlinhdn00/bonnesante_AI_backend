import numpy as np
import pickle
import pywt
import cv2
import torch
from skorch import NeuralNetClassifier
# from ecg_classification.main import MyModule
from main import MyModule
import matplotlib.pyplot as plt

class_mapping = {
    0: "Normal",
    1: "Supraventricular Ectopic Beat",
    2: "Ventricular Ectopic Beat",
    3: "Fusion Beat"
}

def read_ecg_data(file_path):
    with open(file_path, 'r') as file:
        ecg_data = np.array([float(line.strip()) for line in file])
    return ecg_data

def preprocess_ecg_data(ecg_data, wavelet, scales, sampling_rate):
    before, after = 90, 110  # or any other values based on your requirement
    coeffs, _ = pywt.cwt(ecg_data, scales, wavelet, sampling_period=1. / sampling_rate)
    ecg_processed = cv2.resize(coeffs[:, -before - after:], (100, 100))
    ecg_processed = np.expand_dims(ecg_processed, axis=0).astype(np.float32)  # Adding batch and channel dimensions
    return np.expand_dims(ecg_processed, axis=0)

def visualize_ecg(original_ecg, processed_ecg):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    
    original_ecg = original_ecg
    axes[0].plot(original_ecg)
    axes[0].set_title(f'Original ECG Signal')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Amplitude')

    # Processed ECG signal (reshaped to 100x100 for visualization)
    processed_signal = processed_ecg.reshape(100, 100)
    axes[1].imshow(processed_signal, aspect='auto', cmap='jet', origin='lower')
    axes[1].set_title(f'Processed ECG Signal')
    axes[1].set_xlabel('Scaled Time')
    axes[1].set_ylabel('Frequency')
    axes[1].set_yticks(np.linspace(0, 100, num=5))  


    plt.tight_layout()
    plt.show()

def load_model(wavelet):
    model = NeuralNetClassifier(MyModule)
    model.initialize()
    model.load_params(f_params="./models/model_{}.pkl".format(wavelet))
    # model.load_params(f_params="./ecg_classification/models/model_{}.pkl".format(wavelet))
    return model

def infer_ecg(file_path, wavelet="mexh"):
    # Load ECG data
    ecg_data = read_ecg_data(file_path)

    # Preprocess ECG data
    sampling_rate = 360
    scales = pywt.central_frequency(wavelet) * sampling_rate / np.arange(1, 101, 1)
    ecg_processed = preprocess_ecg_data(ecg_data, wavelet, scales, sampling_rate)

    # Load the trained model
    model = load_model(wavelet)

    # Inference
    prediction = model.predict(ecg_processed)

    # Map numeric prediction to class name
    predicted_class = class_mapping[prediction[0]]
    return predicted_class

def amplify_signal(ecg_signal, amplification_factor=3):
    return ecg_signal * amplification_factor

if __name__ == "__main__":
    file_path = "inference_array.txt"
    ecg_data = read_ecg_data(file_path)
    sampling_rate = 360
    amplified_ecg_signal = amplify_signal(ecg_data, amplification_factor=2)
    wavelet = "mexh"
    scales = pywt.central_frequency(wavelet) * sampling_rate / np.arange(1, 101, 1)
    ecg_processed = preprocess_ecg_data(amplified_ecg_signal, wavelet, scales, sampling_rate)

    # Visualize ECG Data
    visualize_ecg(amplified_ecg_signal, ecg_processed)

    # Inference
    prediction = infer_ecg(file_path, wavelet)
    print("Predicted Class:", prediction)