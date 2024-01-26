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
from utils.funcs2 import plot_graph, plot_loss_graph, BPF_dict, normalize
from utils.eval_metrics2 import *
import matplotlib.pyplot as plt
from torchvision import utils


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


def main():
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    '''Setting up'''
    __TIME__ = True
    train = 2  # 0: train from the scratch, 1: continue to train from a pth file, 2: test a model
    test = True
    model_name = "SlowFast_FD"
    save_root_path = "Baseline/"            #"Fivefold/"
    checkpoint_path = "QAnh_Checkpoints/"    #"AFFcheckpoint5fold/" 
    dataset_name = "TESTAPP" #
    # checkpoint_name = "SlowFast_FD_MANHOH_HCI_MMSE_T_10_shift_0.5_best_model.pth"
    checkpoint_name = "SlowFast_FD_MANHOB_HCI_PURE_T_10_shift_0.5_best_model.pth"
    # checkpoint_name = "SlowFast_FD_MANHOB_HCI_MMSE_T_10_shift_0.5_best_model.pth"
    batch_size = 32 
    loss_metric = "combined_loss"
    optimizer = "ada_delta"
    learning_rate = 0.5
    tot_epochs = 25
    model_list = ["MTTS", "TSDAN", "MTTS_CSTM", "SlowFast_FD"]
    window_length = 10
    fs = 30
    skip_connection = True
    new_group_tsm = False
    shift_factor = 0.5

    if __TIME__:
        start_time = time.time()

    # train_dataset, valid_dataset = dataset_loader(train, save_root_path, model_name, dataset_name, window_length)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
    #                           num_workers=6, pin_memory=True, drop_last=False)

    # validation_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=SequentialSampler(valid_dataset),
    #                                num_workers=6, pin_memory=True, drop_last=False)
    
    test_dataset = dataset_loader(2, save_root_path, model_name, dataset_name, window_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=SequentialSampler(test_dataset),
                                   num_workers=6, pin_memory=True, drop_last=False)

    app_mean = []
    app_std = []
    motion_mean = []
    motion_std = []

    # with tqdm(total=len(train_dataset) + len(valid_dataset), position=0, leave=True,
    #           desc='Calculating population statistics') as pbar:
    #     for data in train_loader:
    #         if model_name in ['TSDAN', 'MTTS', 'MTTS_CSTM']:
    #             data = data[0]  # -> (Batch, 2, T, H, W, 3)
    #             motion_data, app_data = torch.tensor_split(data, 2, dim=1)
    #             B, one, T, C, H, W = motion_data.shape

    #             motion_data = motion_data.view(B*one, T, C, H, W)
    #             app_data = app_data.view(B*one, T, C, H, W)
    #             motion_data = motion_data.reshape(B*T, C, H, W)
    #             app_data = app_data.reshape(B*T, C, H, W)

    #             batch_motion_mean = torch.mean(motion_data, dim=(0, 2, 3)).tolist()
    #             batch_motion_std = torch.std(motion_data, dim=(0, 2, 3)).tolist()
    #             batch_app_mean = torch.mean(app_data, dim=(0, 2, 3)).tolist()
    #             batch_app_std = torch.std(app_data, dim=(0, 2, 3)).tolist()

    #             app_mean.append(batch_app_mean)
    #             app_std.append(batch_app_std)
    #             motion_mean.append(batch_motion_mean)
    #             motion_std.append(batch_motion_std)

    #         elif model_name == 'SlowFast_FD':
    #             motion_data = data[0][0]
    #             app_data = data[0][2]
    #             B, T, C, H, W = motion_data.shape
    #             motion_data = motion_data.view(B*T, C, H, W)
    #             app_data = app_data.view(B*T, C, H, W)

    #             batch_motion_mean = torch.mean(motion_data, dim=(0, 2, 3)).tolist()
    #             batch_motion_std = torch.std(motion_data, dim=(0, 2, 3)).tolist()
    #             batch_app_mean = torch.mean(app_data, dim=(0, 2, 3)).tolist()
    #             batch_app_std = torch.std(app_data, dim=(0, 2, 3)).tolist()

    #             app_mean.append(batch_app_mean)
    #             app_std.append(batch_app_std)
    #             motion_mean.append(batch_motion_mean)
    #             motion_std.append(batch_motion_std)

    #         elif model_name in ['STM_Phys', 'New']:
    #             data = data[0].numpy()  # B, T+1, H, W, C
    #             if window_length == 10:
    #                 data = data[:, :-1, :, :, :]
    #             else:
    #                 data = data[:, :-2, :, :, :]
    #             B, T, C, H, W = data.shape
    #             data = np.reshape(data, (B*T, C, H, W))
    #             batch_app_mean = np.mean(data, axis=(0, 2, 3))
    #             batch_app_std = np.std(data, axis=(0, 2, 3))
    #             app_mean.append(batch_app_mean)
    #             app_std.append(batch_app_std)

    #         pbar.update(B)

    #     for i, data in enumerate(validation_loader):
    #         if model_name in ['TSDAN', 'MTTS', 'MTTS_CSTM']:
    #             data = data[0]  # shape (Batch, T+1, H, W, 6)
    #             motion_data, app_data = torch.tensor_split(data, 2, dim=1)
    #             B, one, T, C, H, W = motion_data.shape

    #             motion_data = motion_data.view(B*one, T, C, H, W)
    #             app_data = app_data.view(B*one, T, C, H, W)
    #             motion_data = motion_data.reshape(B*T, C, H, W)
    #             app_data = app_data.reshape(B*T, C, H, W)

    #             batch_motion_mean = torch.mean(motion_data, dim=(0, 2, 3)).tolist()
    #             batch_motion_std = torch.std(motion_data, dim=(0, 2, 3)).tolist()
    #             batch_app_mean = torch.mean(app_data, dim=(0, 2, 3)).tolist()
    #             batch_app_std = torch.std(app_data, dim=(0, 2, 3)).tolist()

    #             app_mean.append(batch_app_mean)
    #             app_std.append(batch_app_std)
    #             motion_mean.append(batch_motion_mean)
    #             motion_std.append(batch_motion_std)

    #         elif model_name == 'SlowFast_FD':
    #             motion_data = data[0][0]
    #             app_data = data[0][2]
    #             B, T, C, H, W = motion_data.shape
    #             motion_data = motion_data.view(B*T, C, H, W)
    #             app_data = app_data.view(B*T, C, H, W)

    #             batch_motion_mean = torch.mean(motion_data, dim=(0, 2, 3)).tolist()
    #             batch_motion_std = torch.std(motion_data, dim=(0, 2, 3)).tolist()
    #             batch_app_mean = torch.mean(app_data, dim=(0, 2, 3)).tolist()
    #             batch_app_std = torch.std(app_data, dim=(0, 2, 3)).tolist()

    #             app_mean.append(batch_app_mean)
    #             app_std.append(batch_app_std)
    #             motion_mean.append(batch_motion_mean)
    #             motion_std.append(batch_motion_std)

    #         elif model_name in ['STM_Phys', 'New']:
    #             data = data[0].numpy()  # B, T+1, H, W, C
    #             if window_length == 10:
    #                 data = data[:, :-1, :, :, :]
    #             else:
    #                 data = data[:, :-2, :, :, :]
    #             B, T, C, H, W = data.shape
    #             data = np.reshape(data, (B * T, C, H, W))

    #             batch_app_mean = np.mean(data, axis=(0, 2, 3))
    #             batch_app_std = np.std(data, axis=(0, 2, 3))

    #             app_mean.append(batch_app_mean)
    #             app_std.append(batch_app_std)

    #         pbar.update(B)
    #     pbar.close()

    with tqdm(total=len(test_dataset), position=0, leave=True,
              desc='Calculating population statistics') as pbar:
        for data in test_loader:
            if model_name in ['TSDAN', 'MTTS', 'MTTS_CSTM']:
                data = data[0]  # -> (Batch, 2, T, H, W, 3)
                motion_data, app_data = torch.tensor_split(data, 2, dim=1)
                B, one, T, C, H, W = motion_data.shape

                motion_data = motion_data.view(B*one, T, C, H, W)
                app_data = app_data.view(B*one, T, C, H, W)
                motion_data = motion_data.reshape(B*T, C, H, W)
                app_data = app_data.reshape(B*T, C, H, W)

                batch_motion_mean = torch.mean(motion_data, dim=(0, 2, 3)).tolist()
                batch_motion_std = torch.std(motion_data, dim=(0, 2, 3)).tolist()
                batch_app_mean = torch.mean(app_data, dim=(0, 2, 3)).tolist()
                batch_app_std = torch.std(app_data, dim=(0, 2, 3)).tolist()

                app_mean.append(batch_app_mean)
                app_std.append(batch_app_std)
                motion_mean.append(batch_motion_mean)
                motion_std.append(batch_motion_std)

            elif model_name == 'SlowFast_FD':
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

            elif model_name in ['STM_Phys', 'New']:
                data = data[0].numpy()  # B, T+1, H, W, C
                if window_length == 10:
                    data = data[:, :-1, :, :, :]
                else:
                    data = data[:, :-2, :, :, :]
                B, T, C, H, W = data.shape
                data = np.reshape(data, (B*T, C, H, W))
                batch_app_mean = np.mean(data, axis=(0, 2, 3))
                batch_app_std = np.std(data, axis=(0, 2, 3))
                app_mean.append(batch_app_mean)
                app_std.append(batch_app_std)

            pbar.update(B)

        pbar.close()

    if model_name in ['TSDAN', 'MTTS', 'MTTS_CSTM', 'SlowFast_FD']:
        # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
        app_mean = np.array(app_mean).mean(axis=0) / 255
        app_std = np.array(app_std).mean(axis=0) / 255
        motion_mean = np.array(motion_mean).mean(axis=0) / 255
        motion_std = np.array(motion_std).mean(axis=0) / 255
        pop_mean = np.stack((app_mean, motion_mean))  # 0 is app, 1 is motion
        pop_std = np.stack((app_std, motion_std))

    elif model_name in ['STM_Phys', 'New']:
        pop_mean = np.array(app_mean).mean(axis=0) / 255
        pop_std = np.array(app_std).mean(axis=0) / 255

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_model_support(model_name, model_list)

    model = get_model(model_name, pop_mean, pop_std, frame_depth=window_length, skip=skip_connection,
                      shift_factor=shift_factor, group_on=new_group_tsm)
    model.to(device)

    criterion = loss_fn(loss_metric)
    optimizer = optim.optimizer(model.parameters(), learning_rate, optimizer)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    min_val_loss = 10000
    min_val_loss_model = None
    torch.backends.cudnn.benchmark = True

    train_loss = []
    valid_loss = []

    if __TIME__:
        log_info_time("Preprocessing time \t: ", datetime.timedelta(seconds=time.time() - start_time))

    if __TIME__:
        start_time = time.time()
    
        
    checkpoint = torch.load(checkpoint_path + checkpoint_name)
    model.load_state_dict(checkpoint["model"])
    epoch = tot_epochs
    optimizer.load_state_dict(checkpoint["optimizer"])
    # scheduler.load_state_dict(checkpoint['scheduler'])
    train_loss = checkpoint["train_loss"]
    valid_loss = checkpoint["valid_loss"]
    min_val_loss = valid_loss[-1]
    min_val_loss_model = copy.deepcopy(model)

    with tqdm(test_loader, desc="Validation ", total=len(test_loader), colour='green') as tepoch:
        model.eval()
        running_loss = 0.0

        inference_array = []
        target_array = []

        with torch.no_grad():
            for inputs, target in tepoch:
                tepoch.set_description(f"Test")
                if torch.isnan(target).any():
                    print('A')
                    return
                if torch.isinf(target).any():
                    print('B')
                    return

                if model_name == 'SlowFast_FD':
                    inputs = [_inputs.cuda() for _inputs in inputs]
                else:
                    inputs = inputs.to(device)
                target = target.to(device)

                outputs = model(inputs)
                if torch.isnan(outputs).any():
                    print('A')
                    return
                if torch.isinf(outputs).any():
                    print('B')
                    return
                loss = criterion(outputs, target)

                running_loss += loss.item() * target.size(0) * target.size(1)
                tepoch.set_postfix(loss='%.6f' % (running_loss / len(test_loader) / window_length/batch_size))


                inference_array = np.append(inference_array, np.reshape(outputs.cpu().detach().numpy(), (1, -1)))
                target_array = np.append(target_array, np.reshape(target.cpu().detach().numpy(), (1, -1)))

            valid_loss.append(running_loss / len(test_loader) / window_length/batch_size)


    result = {}
    groundtruth = {}
    start_idx = 0
    n_frames_per_video = test_dataset.n_frames_per_video
    for i, value in enumerate(n_frames_per_video):
        # if dataset_name == 'PURE':
        #     result[i] = inference_array[start_idx:start_idx + value]
        # else:
        result[i] = normalize(inference_array[start_idx:start_idx + value])
        groundtruth[i] = target_array[start_idx:start_idx + value]
        start_idx += value

    # plot_loss_graph(train_loss, valid_loss)
    plot_graph(0, 500, groundtruth[3], result[3])
    result = BPF_dict(result, fs)
    groundtruth = BPF_dict(groundtruth, fs)
    plot_graph(0, 500, groundtruth[3], result[3])

    mae, rmse, acc3, acc5, acc10, _ = HR_Metric(groundtruth, result, fs, 30, 1)
    pearson = Pearson_Corr(groundtruth, result)
    print('MAE 30s: ' + str(round(mae, 3)))
    print('RMSE 30s: ' + str(round(rmse, 3)))
    print('Accuracy 3 30s: ' + str(round(acc3, 3)))
    print('Accuracy 5 30s: ' + str(round(acc5, 3)))
    print('Accuracy 10 30s: ' + str(round(acc10, 3)))

    print('Pearson 30s: ' + str(round(pearson, 3)))

    mae, rmse, acc3, acc5, acc10c, _ = HR_Metric(groundtruth, result, fs, 20, 1)
    print('MAE 20s: ' + str(round(mae, 3)))
    print('RMSE 20s: ' + str(round(rmse, 3)))
    print('Accuracy 3 20s: ' + str(round(acc3, 3)))
    print('Accuracy 5 20s: ' + str(round(acc5, 3)))
    print('Accuracy 10 20s: ' + str(round(acc10, 3)))

    mae, rmse, acc3, acc5, acc10, _ = HR_Metric(groundtruth, result, fs, 10, 1)
    print('MAE 10s: ' + str(round(mae, 3)))
    print('RMSE 10s: ' + str(round(rmse, 3)))
    print('Accuracy 3 10s: ' + str(round(acc3, 3)))
    print('Accuracy 5 10s: ' + str(round(acc5, 3)))
    print('Accuracy 10 10s: ' + str(round(acc10, 3)))
    if __TIME__:
        log_info_time("Total time \t: ", datetime.timedelta(seconds=time.time() - start_time))

if __name__ == '__main__':
    main()
