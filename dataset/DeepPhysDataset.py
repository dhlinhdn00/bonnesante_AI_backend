import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DeepPhysDataset(Dataset):
    def __init__(self, file, window_length):
        super(DeepPhysDataset, self).__init__()
        self.transform = torch.nn.Sequential(
            transforms.Resize((36, 36))
        )
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
        y = torch.tensor(self.label[seq_i:seq_i + self.window_length + 1], dtype=torch.float32)

        x = self.transform(x)
        motion_frames = torch.empty((self.window_length, 3, 36, 36), dtype=torch.float32)
        labels = torch.empty((self.window_length, 1), dtype=torch.float32)
        for i in range(self.window_length):
            motion_frames[i] = x[i+1] - x[i]
            labels[i] = y[i+1] - y[i]
        average_frame = x[:-1]
        x = torch.stack((motion_frames, average_frame))  # -> 2, T, 36, 36, 3
        y = labels

        return x, y
