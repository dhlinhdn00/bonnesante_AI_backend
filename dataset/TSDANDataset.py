import torch
from torch.utils.data import Dataset


class TSDANDataset(Dataset):
    def __init__(self, file, window_length):
        super(TSDANDataset, self).__init__()
        self.video = file["preprocessed_video"]
        self.label = file["preprocessed_label"]
        self.tot_length = len(self.label)

        self.window_length = window_length

    def __len__(self):
        return (self.tot_length - 1) // self.window_length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq_i = idx * self.window_length
        x = torch.tensor(self.video[seq_i:seq_i + self.window_length + 1], dtype=torch.float32).permute(0, 3, 1, 2)
        y = torch.tensor(self.label[seq_i:seq_i + self.window_length], dtype=torch.float32)

        motion_frames = torch.empty((self.window_length, 3, 72, 72), dtype=torch.float32)
        for i in range(self.window_length):
            motion_frames[i] = x[i+1] - x[i]
        average_frame = x[:-1]
        x = torch.stack((motion_frames, average_frame))  # -> 2, T, 72, 72, 3
        return x, y
