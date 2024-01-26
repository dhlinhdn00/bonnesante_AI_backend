from math import sqrt
import numpy as np
from scipy.signal import welch

def HR_Metric(groundtruth, prediction, fs, window, step):
    count = 0
    correct3 = 0
    correct5 = 0
    correct10 = 0
    error_sum = 0.
    squared_error_sum = 0.
    if type(fs) == int:
        samples = fs * window
        step = fs * step

    for i in np.arange(len(groundtruth.keys())):
        target_signal = groundtruth[i]
        predict_signal = prediction[i]
        signal_length = len(target_signal)
        if type(fs) == list:
            step = fs[i] * step
            samples = fs[i] * window   # window 10s, 20s, 30s
        for j in np.arange(0, signal_length, step):
            j = int(j)
            if j + samples >= signal_length:
                break
            predict_segment = predict_signal[j:j + samples]
            target_segment = target_signal[j:j + samples]
            if type(fs) == list:
                # directly use fourier transform
                predict_fft = np.square(np.abs(np.fft.rfft(predict_segment)))
                gt_fft = np.square(np.abs(np.fft.rfft(target_segment)))
                frequency = (np.linspace(0, fs[i] / 2, len(predict_fft)))
                pre_idx = np.argmax(predict_fft)
                gt_idx = np.argmax(gt_fft)
                predict_hr = frequency[pre_idx] * 60
                gt_hr = frequency[gt_idx] * 60
            else:
                if type(fs) == list:
                    NyquistF = int(fs[i])/2.
                else:   # from H.Anh (Welch's method)
                    NyquistF = fs/2.
                FResBPM = 0.5
                nfft = np.ceil((60*2*NyquistF)/FResBPM)
                if samples < 256:
                    seglength = samples
                    overlap = int(0.8*samples)  # fixed overlapping
                else:
                    seglength = 256
                    overlap = 200  
                
                if type(fs) == list:
                    pred_freqs, pred_psd = welch(predict_segment, fs=int(fs[i]), nperseg=seglength, noverlap=overlap, nfft=nfft) # use welch algorithm
                else:
                    pred_freqs, pred_psd = welch(predict_segment, fs=fs, nperseg=seglength, noverlap=overlap, nfft=nfft) # use welch algorithm
                
                # range of HR frequencies to consider
                pred_max_power_idx = np.argmax(pred_psd)
                predict_hr = pred_freqs[pred_max_power_idx]*60

                # gt_freqs, gt_psd = welch(target_segment, fs=fs, nperseg=None, noverlap=0) # use welch algorithm
                if type(fs)==list:
                    gt_freqs, gt_psd = welch(target_segment, fs=int(fs[i]), nperseg=seglength, noverlap=overlap, nfft = nfft)
                else:
                    gt_freqs, gt_psd = welch(target_segment, fs=fs, nperseg=seglength, noverlap=overlap, nfft = nfft)
                # range of HR frequencies to consider
                gt_max_power_idx = np.argmax(gt_psd)
                gt_hr = gt_freqs[gt_max_power_idx]*60

            p_g=predict_hr - gt_hr
            error_sum += abs(p_g)
            if abs(p_g) < 3:
                correct3 += 1
            if abs(p_g) < 5:
                correct5 += 1
            if abs(p_g) < 10:
                correct10 += 1
            squared_error_sum += abs(p_g)**2
            count += 1
    mae_loss = error_sum / count
    rmse_loss = sqrt(squared_error_sum / count)
    acc3 = correct3/count
    acc5 = correct5/count
    acc10 = correct10/count
    return mae_loss, rmse_loss, acc3, acc5, acc10, predict_hr


def Pearson_Corr(target, inference):
    pearson = 0
    for i in np.arange(len(target.keys())):
        target_signal = target[i]
        inference_signal = inference[i]
        pearson += np.corrcoef(target_signal, inference_signal)[0][1]
    pearson /= len(target.keys())
    return pearson
