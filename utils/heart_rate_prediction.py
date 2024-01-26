from math import sqrt
import numpy as np
from scipy.signal import welch

def HR_prediction(prediction, fs, window, step):
    samples = fs * window
    # print("samples",samples)
    step = fs * step
    hr_array = []
    for i in np.arange(len(prediction.keys())):
        predict_signal = prediction[i]
        signal_length = len(predict_signal)
        # print("signal length",signal_length)
        for j in np.arange(0, signal_length, step):
            j = int(j)
        
            if j + samples >= signal_length:
            # #if j + samples >= 10000:
                break
            predict_segment = predict_signal[j:j + samples]
            # # directly use fourier transform
            # predict_fft = np.square(np.abs(np.fft.rfft(predict_segment)))
            # frequency = (np.linspace(0, fs/ 2, len(predict_fft)))
            # pre_idx = np.argmax(predict_fft)
            # predict_hr = frequency[pre_idx] * 60
            # # else:

            NyquistF = fs/2.
            FResBPM = 0.5
            nfft = np.ceil((60*2*NyquistF)/FResBPM)
            if samples < 256:
                seglength = samples
                overlap = int(0.8*samples)  # fixed overlapping
            else:
                seglength = 256
                overlap = 200  
            
            pred_freqs, pred_psd = welch(predict_segment, fs=fs, nperseg=seglength, noverlap=overlap, nfft=nfft) # use welch algorithm
            
            # range of HR frequencies to consider
            pred_max_power_idx = np.argmax(pred_psd)
            predict_hr = pred_freqs[pred_max_power_idx]*60
            hr_array.append(predict_hr)
    avg_hr = np.mean(hr_array)
    
    return avg_hr, hr_array

