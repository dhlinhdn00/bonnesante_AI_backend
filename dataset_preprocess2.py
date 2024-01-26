import gc
import os
import h5py
import time
import datetime
import natsort
# natsorted() identifies numbers anywhere in a string and sorts them naturally
from log import log_info_time
from utils.image_preprocess import preprocess_Video_RGB_only
from utils.text_preprocess import *

def preprocessing(save_root_path: str = "Baseline/",
                  data_root_path: str = "DATASETS/",
                  dataset_name: str = "TESTAPP",
                  cv_ratio: int = 0.8,
                  start_time: float = time.time()):
    if dataset_name == "TESTAPP":
        dataset_root_path = data_root_path + dataset_name
        data_list = [data for data in natsort.natsorted(os.listdir(dataset_root_path))]

    img_size = 36

    for data_path in data_list:
        video_data, label_data = preprocess_dataset(dataset_root_path + "/" + data_path, True, dataset_name, img_size)

        if video_data is not None and label_data is not None:
            with h5py.File(save_root_path + dataset_name + ".hdf5", "a") as file:
                if data_path in file:
                    del file[data_path]
                dset = file.create_group(data_path)

                video_shape = video_data.shape
                label_shape = label_data.shape
                dset.create_dataset('video', video_shape, np.uint8, video_data)
                dset.create_dataset('label', label_shape, np.float32, label_data)

    gc.collect()

    split_datasets(save_root_path, dataset_name, cv_ratio)

    log_info_time("Data Processing Time \t: ", datetime.timedelta(seconds=time.time() - start_time))


def preprocess_dataset(path, flag, dataset_name, img_size):
    if dataset_name == 'TESTAPP':
        rst, video = preprocess_Video_RGB_only(path + "/" + str(path.split("/")[-1]) + ".avi", flag, vid_res=img_size)
        if not rst:
            return None, None
        label = PURE_preprocess_label(path + "/" + str(path.split("/")[-1]) + ".json", video.shape[0])
        if label is None:
            return None, None
        return video, label

    return None, None


def split_datasets(save_root_path, dataset_name, cv_ratio):
    with h5py.File(save_root_path + dataset_name + ".hdf5", "r") as file:
        keys = list(file.keys())
        train_length = int(len(keys) * cv_ratio)

        with h5py.File(save_root_path + dataset_name + "_train.hdf5", "a") as train_file:
            for data_path in keys[:train_length]:
                if data_path in train_file:
                    del train_file[data_path]  
                file.copy(file[data_path], train_file, data_path)

        with h5py.File(save_root_path + dataset_name + "_test.hdf5", "a") as test_file:
            for data_path in keys[train_length:]:
                if data_path in test_file:
                    del test_file[data_path]  
                file.copy(file[data_path], test_file, data_path)

if __name__ == '__main__':
    start_time = time.time()
    preprocessing()
    print("Finish!")
