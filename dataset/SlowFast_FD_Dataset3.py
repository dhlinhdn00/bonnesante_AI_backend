import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision import transforms

class SlowFast_FD_Dataset(Dataset):
    def __init__(self, file, ds_name, window_length, valid=False, labeled=True):
        self.transform = torch.nn.Sequential(
            transforms.Resize((36, 36))
        )
        super(SlowFast_FD_Dataset, self).__init__()
        self.ds_name = ds_name
        self.size = 36
        self.valid = valid
        self.window_length = window_length
        self.labeled = labeled
        if type(self.ds_name) == list:
            self.fs = []                
            self.video_fs = []          
        self._get_arrays(file)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        wl = self.window_length
        if type(self.ds_name) == list:      # overlapping
            if idx + wl > self.total_length-1:  # check if idx + wl > total_length => reduce the idx
                idx -= idx + wl - self.total_length + 1
        else:
            idx = int(idx * self.window_length)     # non-overlapping
        x = torch.tensor(self.video[idx:idx + wl+1], dtype=torch.float32).permute(0, 3, 1, 2)    # window //permutation
        x = self.transform(x)
        motion_fast = torch.empty((wl, 3, self.size, self.size), dtype=torch.float32)
        motion_slow = torch.empty((wl//2, 3, self.size, self.size), dtype=torch.float32)
        for i in range(wl):
            motion_fast[i] = x[i+1] - x[i]
            if i % 2 == 0:
                j = i//2
                motion_slow[j] = x[i+2] - x[i]
        appearance_frame = x[:-1]

        if self.labeled:
            y = torch.tensor(self.label[idx:idx + wl], dtype=torch.float32)
            if type(self.ds_name) == list:
                fs = self.fs[idx]  # fs of frame
                return (motion_fast, motion_slow, appearance_frame), y, fs    # add fs
            else:
                return (motion_fast, motion_slow, appearance_frame), y
        else:
            if type(self.ds_name) == list:
                fs = self.fs[idx]  # fs of frame
                return (motion_fast, motion_slow, appearance_frame), None, fs    # add fs
            else:
                return (motion_fast, motion_slow, appearance_frame), None

    def _get_arrays(self, file):
        if self.labeled:
            if type(self.ds_name) == list:
                # prev_idx = 0
                # self.start_idx =[]
                # self.dataset_idx = []

                for t, f in enumerate(file):
                    with tqdm(total=len(list(file[t].keys())), position=0, leave=True, desc=f'Reading from file {t}') as pbar:
                        self.n_frames_per_video = np.empty((len(list(file[t].keys()))), dtype=int)

                        for i, data_path in enumerate(list(file[t].keys())):
                            n_frames_per_video = len(file[t][data_path]['label'])
                            
                            # self.start_idx.append(prev_idx)
                            # prev_idx += n_frames_per_video
                            # self.dataset_idx.append(t)
                            self.n_frames_per_video[i] = n_frames_per_video
                            if self.ds_name[t] == "UBFC" or "PURE":   # change here
                                self.fs.extend([30] * n_frames_per_video)  # change the fs of window
                                self.video_fs.append(30)
                            elif self.ds_name[t] == "MMSE":  # change here
                                self.fs.extend([25] * n_frames_per_video)   # change the fs
                                self.video_fs.append(25)  # fs of video
                            elif self.ds_name[t] == "MANHOB_HCI":  # change here
                                self.fs.extend([61] * n_frames_per_video)   # change the fs
                                self.video_fs.append(61)  # fs of video
                            # if downsampling MANHOB change 61 to 30
                            # uncomment under code

                            # if self.ds_name[t] == "MANHOB_HCI":                 
                            #     video_frames = self.down(file[t][data_path]['video'])
                            #     labels = self.down(file[t][data_path]['label'])       # downsampling manhob
                            # else: 
                            video_frames = file[t][data_path]['video']
                            labels = file[t][data_path]['label']

                            if i == 0 and t == 0:
                                self.video = video_frames
                                self.label = labels
                            else:
                                self.video = np.append(self.video, video_frames, axis=0)
                                self.label = np.append(self.label, labels)
                        pbar.update(1)
                self.total_length = (len(self.label) - 1)        # frame index
            else:
                with tqdm(total=len(list(file.keys())), position=0, leave=True, desc='Reading from file') as pbar:
                    self.n_frames_per_video = np.empty((len(list(file.keys()))), dtype=int)
                    for i, data_path in enumerate(list(file.keys())):
                        n_frames_per_video = len(file[data_path]['label'])
                        self.n_frames_per_video[i] = n_frames_per_video
                        video_frames = file[data_path]['video']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
                        labels = file[data_path]['label']
                        if i == 0:
                            self.video = video_frames
                            self.label = labels
                        else:
                            self.video = np.append(self.video, video_frames, axis=0)
                            self.label = np.append(self.label, labels)
                        pbar.update(1)
                self.total_length = (len(self.label) - 1) // self.window_length
        else:
            if type(self.ds_name) == list:
                for t, f in enumerate(file):
                    with tqdm(total=len(list(file[t].keys())), position=0, leave=True, desc=f'Reading from file {t}') as pbar:
                        self.n_frames_per_video = np.empty((len(list(file[t].keys()))), dtype=int)
                        for i, data_path in enumerate(list(file[t].keys())):
                            video_frames = file[t][data_path]['video']
                            n_frames_per_video = len(video_frames)
                            self.n_frames_per_video[i] = n_frames_per_video
                            if self.ds_name[t] == "UBFC" or "PURE":
                                self.fs.extend([30] * n_frames_per_video)
                                self.video_fs.append(30)
                            elif self.ds_name[t] == "MMSE":
                                self.fs.extend([25] * n_frames_per_video)
                                self.video_fs.append(25)
                            elif self.ds_name[t] == "MANHOB_HCI":
                                self.fs.extend([61] * n_frames_per_video)
                                self.video_fs.append(61)
                            if i == 0 and t == 0:
                                self.video = video_frames
                            else:
                                self.video = np.append(self.video, video_frames, axis=0)
                            pbar.update(1)
                    self.total_length = (len(self.video) - 1)
            else:
                with tqdm(total=len(list(file.keys())), position=0, leave=True, desc='Reading from file') as pbar:
                    self.n_frames_per_video = np.empty((len(list(file.keys()))), dtype=int)
                    for i, data_path in enumerate(list(file.keys())):
                        print("Keys at the current level:", file[data_path].keys())
                        print("Does 'video' exist?", 'video' in file[data_path].keys())
                        video_frames = file[data_path]['video']
                        n_frames_per_video = len(video_frames)
                        self.n_frames_per_video[i] = n_frames_per_video
                        if i == 0:
                            self.video = video_frames
                        else:
                            self.video = np.append(self.video, video_frames, axis=0)
                        pbar.update(1)
                    self.total_length = (len(self.video) - 1) // self.window_length

    def update_state(self):
        self.pre_train = False
        self.total_length = (self.total_length * self.window_length) - self.window_length

    def down(self, fr):
        self.down_fr = np.delete(fr,np.arange(fr.shape[0]//2)*2+1,axis=0)
        # print(self.down)
        return self.down_fr
    
        
