import copy
import datetime
import time

import h5py
import torch
import optim
from loss2 import loss_fn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from dataset.dataset_loader2 import dataset_loader
from log import log_info_time
from models2 import is_model_support, get_model
from torch.optim import lr_scheduler
from utils.funcs2 import plot_graph_only_inference, BPF_dict, normalize
from utils.eval_metrics2 import *
from utils.heart_rate_prediction import *
import matplotlib.pyplot as plt
from torchvision import utils
import multiprocessing
from dataset_preprocess import preprocessing
from log import log_info_time
import sys
from ecg_classification.predict_for_app import classify_for_app
from resp_prediction.predict_resp_for_app import resp_prediction



print("test3.py sys.path:", sys.path)

def write_array_to_file(array, file_name):
    with open(file_name, 'w') as f:
        for item in array:
            f.write("%s\n" % item)

def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1):
    n, c, w, h = tensor.shape

    if allkernels:
        tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure(figsize=(nrow, rows))
    plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))


def main_ecg():
    '''Setting up'''
    __TIME__ = True
    train = 2  # 0: train from the scratch, 1: continue to train from a pth file, 2: test a model
    test = True

    rppg_folder_name = "pytorch_rppg/"
    backend_folder_name = "heartrate/"

    model_name = "SlowFast_FD"
    save_root_path = "Baseline/" 
    data_root_path = "DATASETS/"
    checkpoint_path = "QAnh_Checkpoints/"
    dataset_name = "TESTAPP"
    # checkpoint_name = "SlowFast_FD_MANHOH_HCI_MMSE_T_10_shift_0.5_best_model.pth"
    checkpoint_name = "SlowFast_FD_MANHOB_HCI_PURE_T_10_shift_0.5_best_model.pth"
    # checkpoint_name = "SlowFast_FD_MANHOB_HCI_MMSE_T_10_shift_0.5_best_model.pth"
    batch_size = 32 
    loss_metric = "combined_loss"
    optimizer_str = "ada_delta"
    learning_rate = 0.5
    tot_epochs = 25
    model_list = "SlowFast_FD"
    window_length = 10
    fs = 30 #25, 30, 60
    skip_connection = True
    new_group_tsm = False
    shift_factor = 0.5
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    if __TIME__:
        start_time = time.time()

    test_dataset = dataset_loader(2, save_root_path, model_name, dataset_name, window_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=SequentialSampler(test_dataset),
                                   num_workers=0, pin_memory=True, drop_last=False)

    app_mean = []
    app_std = []
    motion_mean = []
    motion_std = []

    with tqdm(total=len(test_dataset), position=0, leave=True,
              desc='Calculating population statistics') as pbar:
        for data in test_loader:

            motion_data = data[0][0]
            app_data = data[0][2]
            B, T, C, H, W = motion_data.shape
            motion_data = motion_data.view(B*T, C, H, W)
            app_data = app_data.view(B*T, C, H, W)

            batch_motion_mean = torch.mean(motion_data, dim=(0, 2, 3)).tolist()
            batch_motion_std = torch.std(motion_data, dim=(0, 2, 3)).tolist()
            batch_app_mean = torch.mean(app_data, dim=(0, 2, 3)).tolist()
            batch_app_std = torch.std(app_data, dim=(0, 2, 3)).tolist()

            app_mean.append(batch_app_mean)
            app_std.append(batch_app_std)
            motion_mean.append(batch_motion_mean)
            motion_std.append(batch_motion_std)

            pbar.update(B)

        pbar.close()

    app_mean = np.array(app_mean).mean(axis=0) / 255
    app_std = np.array(app_std).mean(axis=0) / 255
    motion_mean = np.array(motion_mean).mean(axis=0) / 255
    motion_std = np.array(motion_std).mean(axis=0) / 255
    pop_mean = np.stack((app_mean, motion_mean))  # 0 is app, 1 is motion
    pop_std = np.stack((app_std, motion_std))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_model_support(model_name, model_list)

    model = get_model(model_name, pop_mean, pop_std, frame_depth=window_length, skip=skip_connection,
                      shift_factor=shift_factor, group_on=new_group_tsm)
    model.to(device)

    optimizer = optim.optimizer(model.parameters(), learning_rate, optimizer_str)

    torch.backends.cudnn.benchmark = True


    if __TIME__:
        log_info_time("Preprocessing time \t: ", datetime.timedelta(seconds=time.time() - start_time))

    if __TIME__:
        start_time = time.time()
    
    checkpoint = torch.load(checkpoint_path + checkpoint_name)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    with tqdm(test_loader, desc="Validation ", total=len(test_loader), colour='green') as tepoch:
        model.eval()

        inference_array = []

        with torch.no_grad():
            for inputs, _ in tepoch:
                tepoch.set_description(f"Test")

                inputs = [_inputs.cuda() for _inputs in inputs]

                outputs = model(inputs)
                if torch.isnan(outputs).any():
                    print('A')
                    return
                if torch.isinf(outputs).any():
                    print('B')
                    return

                inference_array = np.append(inference_array, np.reshape(outputs.cpu().detach().numpy(), (1, -1)))
                
        write_array_to_file(inference_array, "inference_array.txt")

    result = {}
    start_idx = 0
    n_frames_per_video = test_dataset.n_frames_per_video
    for i, value in enumerate(n_frames_per_video):
        result[i] = normalize(inference_array[start_idx:start_idx + value])
        start_idx += value

    # plot_graph_only_inference(0, 500, result[0])
    result = BPF_dict(result, fs)
    # plot_graph_only_inference(0, 500, result[0])
    predict_hr, hr_array= HR_prediction(result, fs, 10, 1)
    # print(result)
    # print(len(result[0]))
    print(f"Heart rate: {predict_hr}")
    ecg_class = classify_for_app("./inference_array.txt")
    resp_rate = resp_prediction()
    print(ecg_class)
    # print(hr_array)

    if __TIME__:
        log_info_time("Total time \t: ", datetime.timedelta(seconds=time.time() - start_time))
    
    return result, predict_hr, ecg_class, resp_rate

if __name__ == '__main__':
    start_time = time.time()
    multiprocessing.set_start_method('forkserver')
    preprocessing()
    print("Finish!")
    main_ecg()
    # resp_prediction()