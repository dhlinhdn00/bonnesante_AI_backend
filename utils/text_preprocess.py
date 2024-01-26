import numpy as np
import h5py
import scipy.signal
import pandas as pd
from utils.funcs import BPF_signal
import json
import matplotlib.pyplot as plt


def UBFC_preprocess_Label(path, frame_total):
    # Trace values are normalized by default
    f = open(path, 'r')
    f_read = f.read()
    f_read = f_read.split('\n')

    label = f_read[0].split()
    label = list(map(float, label))
    label = np.array(label)
    f.close()
    if len(label) >= 1.5*frame_total:
        label = BPF_signal(label, 60, 0.4, 4)
    else:
        label = BPF_signal(label, 30, 0.4, 4)
    label = signal_normalization(label)
    label = scipy.signal.resample(label, frame_total)

    return label


def MTTS_cohface_Label(path, frame_total):
    f = h5py.File(path, "r")
    label = list(f['pulse'])
    f.close()
    label = np.interp(np.arange(0, len(frame_total)+1),
                      np.linspace(0, len(frame_total)+1, num=len(label)), label)

    return label


def MTTS_VIPL_Label(path, frame_total):
    f = pd.read_csv(path)
    label = f['Wave']
    label = np.asarray(label, dtype=np.float32)
    label = BPF_signal(label, 60, 0.4, 4)
    label = scipy.signal.resample(label, frame_total)
    label = signal_normalization(label)
    return label


def MTTS_MMSE_Label(path, frame_total):
    # Trace values are normalized by default
    # video frame rate is 25 fps
    # signal sampling rate is 1000 Hz
    f = open(path, 'r')
    f_read = f.readlines()
    label = list(map(float, f_read))
    label = np.array(label)

    f.close()
    ratio = 1000 / 25
    if len(label) < ratio*frame_total - 0.1*len(label):
        return None
    label = BPF_signal(label, 1000, 0.4, 2.67)
    label = signal_normalization(label)
    label = scipy.signal.resample(label, frame_total)

    return label


def MTTS_MANHOB_Label(path, frame_total):
    f = open(path, 'r')
    f_read = f.read()
    label = f_read.split()
    f.close()

    label = np.asarray(list(map(float, label)))
    label = BPF_signal(label, 256, 0.67, 2.)
    label = scipy.signal.resample(label, frame_total)
    return label


def PURE_preprocess_label(path, frame_total):
    file = open(path)
    data = json.loads(file.read())
    data = data['/FullPackage']
    label = []
    for p in data:
        label.append(p['Value']['waveform'])
    file.close()

    label = np.asarray(label)
    label = BPF_signal(label, 60, 0.5, 2.67)
    label = signal_normalization(label)
    label = scipy.signal.resample(label, frame_total)
    return label


def signal_normalization(signal):
    signal = (signal - np.mean(signal)) / np.std(signal)
    return signal
