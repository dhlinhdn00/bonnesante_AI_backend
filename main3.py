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
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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


    ''' MSE is Mean Square Error   '''

    '''Setting up'''
    __TIME__ = True
    train = 0  # 0: train from the scratch, 1: continue to train from a pth file, 2: test a model
    model_name = "SlowFast_FD"
    save_root_path = "Baseline/"
    checkpoint_path = "QAnh_Checkpoints/"
    dataset_name = ["PURE"]
    checkpoint_name = "SlowFast_FD_MANHOB_HCI_MMSE_T_10_shift_0.5_best_model.pth"
    batch_size = 32
    loss_metric = "combined_loss" 
    optimizer = "ada_delta"    
    learning_rate = 0.5
    tot_epochs = 25
    model_list = ["MTTS", "TSDAN", "MTTS_CSTM", "SlowFast_FD", "SlowFast_AM"]
    window_length = 10         #block_length
    fs = [30]   
    skip_connection = True
    new_group_tsm = False                       
    shift_factor = 0.5     
               

    if __TIME__:
        start_time = time.time()

    if train == 0 or train == 1:
        train_dataset, valid_dataset = dataset_loader(train, save_root_path, model_name, dataset_name, window_length, fold=None)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=6, pin_memory=True, drop_last=False)

        validation_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=SequentialSampler(valid_dataset),
                                       num_workers=6, pin_memory=True, drop_last=False)

        app_mean = []
        app_std = []
        motion_mean = []
        motion_std = []

        with tqdm(total=len(train_dataset) + len(valid_dataset), position=0, leave=True,
                  desc='Calculating population statistics') as pbar:
            for data in train_loader:
                #TSDAN is used to be an evaluation model MTTS_CSTM = TS_CST
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

                elif model_name == 'SlowFast_AM':
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

            for i, data in enumerate(validation_loader):
                if model_name in ['TSDAN', 'MTTS', 'MTTS_CSTM']:
                    data = data[0]  # shape (Batch, T+1, H, W, 6)
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

                elif model_name == 'SlowFast_AM':
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
                    data = np.reshape(data, (B * T, C, H, W))

                    batch_app_mean = np.mean(data, axis=(0, 2, 3))
                    batch_app_std = np.std(data, axis=(0, 2, 3))

                    app_mean.append(batch_app_mean)
                    app_std.append(batch_app_std)

                pbar.update(B)
            pbar.close()

        if model_name in ['TSDAN', 'MTTS', 'MTTS_CSTM', 'SlowFast_FD', "SlowFast_AM"]:
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    if train == 0 or train == 1:
        torch.autograd.set_grad_enabled(True)
        if train == 0:
            start_epoch = 1
        else:
            checkpoint = torch.load(checkpoint_path + checkpoint_name)
            model.load_state_dict(checkpoint["model"])
            start_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            # scheduler.load_state_dict(checkpoint['scheduler'])
            train_loss = checkpoint["train_loss"]
            valid_loss = checkpoint["valid_loss"]
            min_val_loss = valid_loss[-1]
            min_val_loss_model = copy.deepcopy(model)

        # if len(train_dataset) % batch_size == 1:            # for AFF MANHOB_HCI
        #     total = len(train_loader) - 1
        # else:
        #     total = len(train_loader)

        for epoch in np.arange(start_epoch, tot_epochs + 1):
            with tqdm(train_loader, desc="Train ", total=len(train_loader), colour='red') as tepoch:
                model.train()
                running_loss = 0.0
                for inputs, target, frame_fs in tepoch:
                    # if inputs[0].shape[0] == 1:
                    #     continue
                    optimizer.zero_grad(set_to_none=True)
                    if torch.isnan(target).any():
                        print('A1')
                        return
                    if torch.isinf(target).any():
                        print('B1')
                        return
                    tepoch.set_description(f"Train Epoch {epoch}")

                    if model_name == 'SlowFast_FD' or model_name == 'SlowFast_AM':
                        inputs = [_inputs.cuda() for _inputs in inputs]
                    else:
                        inputs = inputs.to(device)
                    target = target.to(device)
                    outputs = model(inputs)

                    if torch.isnan(outputs).any():
                        print('A2')
                        return
                    if torch.isinf(outputs).any():
                        print('B2')
                        return

                    if loss_metric == "snr":
                        loss = criterion(outputs, target, frame_fs)
                    else:
                        loss = criterion(outputs, target)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * target.size(0) * target.size(1)
                    del loss, outputs, inputs, target
                    tepoch.set_postfix(loss='%.6f' % (running_loss / len(train_loader) / window_length / batch_size))
                train_loss.append(running_loss / len(train_loader) / window_length / batch_size)

            if epoch == tot_epochs and min_val_loss_model is not None:
                model = min_val_loss_model

            # scheduler.step()

            with tqdm(validation_loader, desc="Validation ", total=len(validation_loader), colour='green') as tepoch:
                model.eval()
                running_loss = 0.0

                # if epoch == tot_epochs:
                inference_array = []
                target_array = []
                # torch.set_anomaly_enabled(True)

                with torch.no_grad():
                    for inputs, target, frame_fs in tepoch:
                        tepoch.set_description(f"Validation")
                        if torch.isnan(target).any():
                            print('A3')
                            return
                        if torch.isinf(target).any():
                            print('B3')
                            return
                        if model_name == 'SlowFast_FD' or model_name == 'SlowFast_AM':
                            inputs = [_inputs.cuda() for _inputs in inputs]
                        else:
                            inputs = inputs.to(device)
                       
                        target = target.to(device)
                        outputs = model(inputs)
                        if torch.isnan(outputs).any():
                            print('A4')
                            return
                        if torch.isinf(outputs).any():
                            print('B4')
                            return
                        if loss_metric == "snr":
                            loss = criterion(outputs, target, frame_fs)
                        else:
                            loss = criterion(outputs, target)

                        running_loss += loss.item() * target.size(0) * target.size(1)
                        tepoch.set_postfix(loss='%.6f' % (running_loss / len(validation_loader) / window_length / batch_size))

                        # if epoch == tot_epochs:
                            # inference_array.extend(normalize(torch.squeeze(outputs).cpu().detach().numpy()))
                            # target_array.extend(normalize(torch.squeeze(target).cpu().detach().numpy()))
                        inference_array = np.append(inference_array, np.reshape(outputs.cpu().detach().numpy(), (1, -1)))
                        target_array = np.append(target_array, np.reshape(target.cpu().detach().numpy(), (1, -1)))

                    valid_loss.append(running_loss / len(validation_loader) / window_length / batch_size)

            if min_val_loss > valid_loss[-1]:  # save the train model
                min_val_loss = valid_loss[-1]
                min_val_loss_model = copy.deepcopy(model)

                checkpoint = {'epoch': epoch,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              # 'scheduler': scheduler.state_dict(),
                              'loss': loss,
                              'train_loss': train_loss,
                              'valid_loss': valid_loss}
                # Anh
                # torch.save(checkpoint,   
                #            checkpoint_path + "/" + model_name + "_" + dataset_name + "_T_" + str(T) + "_shift_"+str(shift_factor)+ '_best_model.pth')
                # Hao
                # torch.save(checkpoint, checkpoint_path + "/" + checkpoint_name)
                #QAnh
                torch.save(checkpoint,   
                        checkpoint_path + "/" + model_name + "_" + "_".join(dataset_name) + "_T_" + str(T) + "_shift_"+str(shift_factor)+ '_best_model.pth')
                
                
            # if epoch % 5 == 0:
            #     checkpoint = {'epoch': epoch,
            #                   'model': model.state_dict(),
            #                   'optimizer': optimizer.state_dict(),
            #                   # 'scheduler': scheduler.state_dict(),
            #                   'loss': loss,
            #                   'train_loss': train_loss,
            #                   'valid_loss': valid_loss}
            #     torch.save(checkpoint, checkpoint_path + "/" + model_name + "_" + dataset_name + "_" + str(epoch) + "_" +
            #                str(running_loss / len(validation_loader) / window_length / batch_size) + '.pth')

            if epoch == tot_epochs:
                # first_layer = model.motion_model_fast.block_1.tsm.conv_2d[0].weight.data.clone()
                # visTensor(first_layer, ch=0, allkernels=False)
                #
                # plt.axis('off')
                # plt.ioff()
                # plt.show()

                result = {}
                groundtruth = {}
                start_idx = 0
                n_frames_per_video = valid_dataset.n_frames_per_video
                vid_fs = valid_dataset.video_fs
                for i, value in enumerate(n_frames_per_video):
                    # if dataset_name == 'PURE':
                    #     result[i] = inference_array[start_idx:start_idx + value]
                    # else:
                    result[i] = normalize(inference_array[start_idx:start_idx + value])
                    groundtruth[i] = target_array[start_idx:start_idx + value]
                    start_idx += value

                # plot_loss_graph(train_loss, valid_loss)
                # plot_graph(0, 500, groundtruth[3], result[3])
                result = BPF_dict(result, vid_fs)
                groundtruth = BPF_dict(groundtruth, vid_fs)
                # plot_graph(0, 500, groundtruth[3], result[3])

                mae, rmse, acc3, acc5, acc10 = HR_Metric(groundtruth, result, vid_fs, 30, 1)
                pearson = Pearson_Corr(groundtruth, result)

                # print(checkpoint_name)

                print('MAE 30s: ' + str(round(mae, 3)))
                print('RMSE 30s: ' + str(round(rmse, 3)))
                # print('Accuracy 3 30s: ' + str(round(acc3, 3)))
                # print('Accuracy 5 30s: ' + str(round(acc5, 3)))
                # print('Accuracy 10 30s: ' + str(round(acc10, 3)))

                print('Pearson 30s: ' + str(round(pearson, 3)))

                mae, rmse, acc3, acc5, acc10c = HR_Metric(groundtruth, result, vid_fs, 20, 1)
                print('MAE 20s: ' + str(round(mae, 3)))
                print('RMSE 20s: ' + str(round(rmse, 3)))
                # print('Accuracy 3 30s: ' + str(round(acc3, 3)))
                # print('Accuracy 5 30s: ' + str(round(acc5, 3)))
                # print('Accuracy 10 30s: ' + str(round(acc10, 3)))

                mae, rmse, acc3, acc5, acc10 = HR_Metric(groundtruth, result, vid_fs, 10, 1)
                print('MAE 10s: ' + str(round(mae, 3)))
                print('RMSE 10s: ' + str(round(rmse, 3)))
                # print('Accuracy 3 30s: ' + str(round(acc3, 3)))
                # print('Accuracy 5 30s: ' + str(round(acc5, 3)))
                # print('Accuracy 10 30s: ' + str(round(acc10, 3)))

        if __TIME__:
            log_info_time("Total training time \t: ", datetime.timedelta(seconds=time.time() - start_time))

    elif train == 2:
        checkpoint = torch.load(checkpoint_path + checkpoint_name)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        if __TIME__:
            start_time = time.time()
        test_dataset = dataset_loader(train, save_root_path=save_root_path, model_name=model_name,
                                      dataset_name=dataset_name, window_length=window_length)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                 drop_last=True)

        with tqdm(test_loader, desc="test ", total=len(test_loader)) as tepoch:
            inference_array = []
            target_array = []
            with torch.no_grad():
                for inputs, target in tepoch:
                    tepoch.set_description("test")
                    inputs = inputs.to(device)
                    target = target.to(device)
                    outputs = model(inputs)

                    inference_array.extend(outputs.cpu().detach().numpy())
                    target_array.extend(target.cpu().detach().numpy())

                if __TIME__:
                    log_info_time("inference time \t: ", datetime.timedelta(seconds=time.time() - start_time))

        plot_graph(0, 300, target_array, inference_array)
        if len(np.shape(inference_array)) >= 2:
            inference_array = np.reshape(inference_array, (-1, 1))
        if len(np.shape(target_array)) >= 2:
            target_array = np.reshape(target_array, (-1, 1))
        print('MAE: ' + str(MAE(target_array, inference_array)[0]))
        print('RMSE: ' + str(RMSE(target_array, inference_array)))
        print('Pearson: ' + str(pearson_corr(target_array, inference_array)[0]))


if __name__ == '__main__':
    main()