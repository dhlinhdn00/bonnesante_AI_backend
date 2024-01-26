import tensorflow as tf
import numpy as np
import scipy.io
import os
import sys
import argparse
sys.path.append('../')
from resp_prediction.model import Attention_mask, MTTS_CAN
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter, find_peaks
from resp_prediction.inference_preprocess import preprocess_raw_video, detrend

def find_respiratory_peaks(signal, distance=None, height=None):
    peaks, _ = find_peaks(signal, distance=distance, height=height)
    return peaks

def predict_vitals(path, batch_size, sampling_rate):
    img_rows = 36
    img_cols = 36
    frame_depth = 10
    model_checkpoint = './resp_prediction/mtts_can.hdf5'
    batch_size = batch_size
    fs = sampling_rate
    sample_data_path = path

    dXsub = preprocess_raw_video(sample_data_path, dim=36)
    print('dXsub shape', dXsub.shape)

    dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
    dXsub = dXsub[:dXsub_len, :, :, :]

    model = MTTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    model.load_weights(model_checkpoint)

    yptest = model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)

    resp_pred = yptest[1]
    resp_pred = detrend(np.cumsum(resp_pred), 100)
    [b_resp, a_resp] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')
    resp_pred = scipy.signal.filtfilt(b_resp, a_resp, np.double(resp_pred))

    peaks, _ = find_peaks(resp_pred)
    num_peaks = len(peaks)
    resp_rate = int((num_peaks * 30)/32)

    # plt.figure(figsize=(10, 6))
    # plt.plot(resp_pred)
    # plt.title(f'Resp Prediction - Respiratory Rate: {resp_rate} breaths per minute')
    # plt.show()

    return resp_rate


def resp_prediction():

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--video_path', type=str, default = './DATASETS/TESTAPP/testapp/testapp.avi', help='processed video path')
    # parser.add_argument('--sampling_rate', type=int, default = 30, help='sampling rate of your video')
    # parser.add_argument('--batch_size', type=int, default = 100, help='batch size (multiplier of 10)')
    # args = parser.parse_args()
    result = predict_vitals('./DATASETS/TESTAPP/testapp/testapp.avi', 30, 100)
    print("Breaths per minute: ", result)
    return result

