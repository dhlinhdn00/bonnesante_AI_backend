from concurrent.futures import ProcessPoolExecutor

import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pywt
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from functools import partial
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, Initializer, LRScheduler, TensorBoard
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from torch.backends import cudnn
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

cudnn.benchmark = False
cudnn.deterministic = True

torch.manual_seed(0)


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 7)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.pooling1 = nn.MaxPool2d(5)
        self.pooling2 = nn.MaxPool2d(3)
        self.pooling3 = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x1):
        x1 = F.relu(self.bn1(self.conv1(x1)))  # (16 x 94 x 94)
        x1 = self.pooling1(x1)  # (16 x 18 x 18)
        x1 = F.relu(self.bn2(self.conv2(x1)))  # (32 x 16 x 16)
        x1 = self.pooling2(x1)  # (32 x 5 x 5)
        x1 = F.relu(self.bn3(self.conv3(x1)))  # (64 x 3 x 3)
        x1 = self.pooling3(x1)  # (64 x 1 x 1)
        x1 = x1.view((-1, 64))  # (64,)
        x = F.relu(self.fc1(x1))  # (32,)
        x = self.fc2(x)  # (4,)
        return x


def worker(data, wavelet, scales, sampling_period):
    # heartbeat segmentation interval
    before = 90
    after = 110

    coeffs, frequencies = pywt.cwt(data["signal"], scales, wavelet, sampling_period)
    r_peaks, categories = data["r_peaks"], data["categories"]

    # for remove inter-patient variation
    avg_rri = np.mean(np.diff(r_peaks))

    x1, y, groups = [], [], []
    for i in range(len(r_peaks)):
        if i == 0 or i == len(r_peaks) - 1:
            continue
        if categories[i] == 4:  # remove AAMI Q class
            continue
        x1.append(cv2.resize(coeffs[:, r_peaks[i] - before: r_peaks[i] + after], (100, 100)))
        y.append(categories[i])
        groups.append(data["record"])
    return x1, y, groups


def load_data(wavelet, scales, sampling_rate, filename="./dataset/mitdb.pkl"):
    import pickle
    from sklearn.preprocessing import RobustScaler

    with open(filename, "rb") as f:
        train_data, test_data = pickle.load(f)

    cpus = 22 if joblib.cpu_count() > 22 else joblib.cpu_count() - 1  # for multi-process

    # for training
    x1_train, y_train, groups_train = [], [], []
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        for x1, y, groups in executor.map(
                partial(worker, wavelet=wavelet, scales=scales, sampling_period=1. / sampling_rate), train_data):
            x1_train.append(x1)
            y_train.append(y)
            groups_train.append(groups)

    x1_train = np.expand_dims(np.concatenate(x1_train, axis=0), axis=1).astype(np.float32)
    y_train = np.concatenate(y_train, axis=0).astype(np.int64)
    groups_train = np.concatenate(groups_train, axis=0)

    # for test
    x1_test, y_test, groups_test = [], [], []
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        for x1, y, groups in executor.map(
                partial(worker, wavelet=wavelet, scales=scales, sampling_period=1. / sampling_rate), test_data):
            x1_test.append(x1)
            y_test.append(y)
            groups_test.append(groups)

    x1_test = np.expand_dims(np.concatenate(x1_test, axis=0), axis=1).astype(np.float32)
    y_test = np.concatenate(y_test, axis=0).astype(np.int64)
    groups_test = np.concatenate(groups_test, axis=0)

    # normalization
    scaler = RobustScaler()

    return (x1_train, y_train, groups_train, train_data), (x1_test, y_test, groups_test, test_data)

def visualize_processed_ecg_signals(processed_data):
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))


    # Processed ECG signal
    processed_signal = processed_data[0].reshape(100, 100)
    axes[0].imshow(processed_signal, aspect='auto', cmap='jet', origin='lower')
    axes[0].set_title(f'Beat 1')
    axes[0].set_xlabel('Scaled Time')
    axes[0].set_ylabel('Frequency')

    processed_signal = processed_data[1].reshape(100, 100)
    axes[1].imshow(processed_signal, aspect='auto', cmap='jet', origin='lower')
    axes[1].set_title(f'Beat 2')
    axes[1].set_xlabel('Scaled Time')
    axes[1].set_ylabel('Frequency')

    processed_signal = processed_data[2].reshape(100, 100)
    axes[2].imshow(processed_signal, aspect='auto', cmap='jet', origin='lower')
    axes[2].set_title(f'Beat 3')
    axes[2].set_xlabel('Scaled Time')
    axes[2].set_ylabel('Frequency')

    processed_signal = processed_data[3].reshape(100, 100)
    axes[3].imshow(processed_signal, aspect='auto', cmap='jet', origin='lower')
    axes[3].set_title(f'Beat 4')
    axes[3].set_xlabel('Scaled Time')
    axes[3].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def cl_main():
    sampling_rate = 360

    wavelet = "mexh"  # mexh, morl, gaus8, gaus4
    scales = pywt.central_frequency(wavelet) * sampling_rate / np.arange(1, 101, 1)
    
    (x1_train, y_train, groups_train, train_data), (x1_test, y_test, groups_test, test_data) = load_data(
        wavelet=wavelet, scales=scales, sampling_rate=sampling_rate)
    # print("Data loaded successfully!")
    # print(type(train_data))
    # print(train_data[12])
    # print(len(train_data))
    # sample_index = 0  # Index of the sample you want to visualize
    processed_data = [x1_train[1], x1_train[3], x1_train[4569], x1_train[4]]
    visualize_processed_ecg_signals(processed_data)
    #418
    #1962
    #4355
    # log_dir = "./logs/{}".format(wavelet)
    # shutil.rmtree(log_dir, ignore_errors=True)

    # callbacks = [
    #     Initializer("[conv|fc]*.weight", fn=torch.nn.init.kaiming_normal_),
    #     Initializer("[conv|fc]*.bias", fn=partial(torch.nn.init.constant_, val=0.0)),
    #     LRScheduler(policy=StepLR, step_size=5, gamma=0.1),
    #     EpochScoring(scoring=make_scorer(f1_score, average="macro"), lower_is_better=False, name="valid_f1"),
    #     TensorBoard(SummaryWriter(log_dir))
    # ]
    # net = NeuralNetClassifier(  # skorch is extensive package of pytorch for compatible with scikit-learn
    #     MyModule,
    #     criterion=torch.nn.CrossEntropyLoss,
    #     optimizer=torch.optim.Adam,
    #     lr=0.001,
    #     max_epochs=30,
    #     batch_size=1024,
    #     train_split=predefined_split(Dataset({"x1": x1_test}, y_test)),
    #     verbose=1,
    #     device="cuda",
    #     callbacks=callbacks,
    #     iterator_train__shuffle=True,
    #     optimizer__weight_decay=0,
    # )
    # net.fit({"x1": x1_train}, y_train)
    # y_true, y_pred = y_test, net.predict({"x1": x1_test})

    # print(confusion_matrix(y_true, y_pred))
    # print(classification_report(y_true, y_pred, digits=4))

    # net.save_params(f_params="./models/model_{}.pkl".format(wavelet))


if __name__ == "__main__":
    
    cl_main()
