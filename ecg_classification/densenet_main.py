from concurrent.futures import ProcessPoolExecutor
import cv2
import joblib
import numpy as np
import pywt
import shutil
import torch
import torch.nn as nn
import torch.utils.data
from functools import partial
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, Initializer, LRScheduler, TensorBoard
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import timm

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

class CustomDenseNet121(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(CustomDenseNet121, self).__init__()
        # Load the DenseNet121 model
        self.densenet121 = timm.create_model('densenet121', pretrained=pretrained, num_classes=num_classes)

        # If your input channel is not 3 (e.g., grayscale images), you need to modify the first conv layer
        if pretrained:
            # Change the first convolution layer to accept 1-channel input
            self.densenet121.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        return self.densenet121(x)

def worker(data, wavelet, scales, sampling_period):
    # heartbeat segmentation interval
    before, after = 90, 110
    coeffs, frequencies = pywt.cwt(data["signal"], scales, wavelet, sampling_period)
    r_peaks, categories = data["r_peaks"], data["categories"]

    x1, y, groups = [], [], []
    for i in range(len(r_peaks)):
        if i == 0 or i == len(r_peaks) - 1:
            continue
        if categories[i] == 4:
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

    return (x1_train, y_train, groups_train), (x1_test, y_test, groups_test)


def main():
    model = CustomDenseNet121(num_classes=4, pretrained=True)

# Đếm số lượng tham số huấn luyện
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    # sampling_rate = 900
    # wavelet = "mexh"
    # scales = pywt.central_frequency(wavelet) * sampling_rate / np.arange(1, 101, 1)

    # (x1_train, y_train, groups_train), (x1_test, y_test, groups_test) = load_data(wavelet, scales, sampling_rate)
    # print("Data loaded successfully!")

    # log_dir = "./logs_densenet/{}".format(wavelet)
    # shutil.rmtree(log_dir, ignore_errors=True)

    # callbacks = [
    #     Initializer("[conv|fc]*.weight", fn=torch.nn.init.kaiming_normal_),
    #     Initializer("[conv|fc]*.bias", fn=partial(torch.nn.init.constant_, val=0.0)),
    #     LRScheduler(policy=StepLR, step_size=5, gamma=0.1),
    #     EpochScoring(scoring=make_scorer(f1_score, average="macro"), lower_is_better=False, name="valid_f1"),
    #     TensorBoard(SummaryWriter(log_dir))
    # ]

    # net = NeuralNetClassifier(
    #     CustomDenseNet121,
    #     criterion=torch.nn.CrossEntropyLoss,
    #     optimizer=torch.optim.Adam,
    #     lr=0.001,
    #     max_epochs=30,
    #     batch_size=32,
    #     train_split=predefined_split(Dataset(x1_test, y_test)),
    #     verbose=1,
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    #     callbacks=callbacks,
    #     iterator_train__shuffle=True,
    #     optimizer__weight_decay=0,
    # )
    
    # net.fit(x1_train, y_train)
    # y_true, y_pred = y_test, net.predict(x1_test)

    # print(confusion_matrix(y_true, y_pred))
    # print(classification_report(y_true, y_pred, digits=4))

    # net.save_params(f_params="./models/model_densenet_{}.pkl".format(wavelet))

if __name__ == "__main__":
    main()
