import torch
import numpy as np
from nets.models.sub_models.AppearanceModel import AppearanceModel_2D
from nets.models.sub_models.LinearModel import LinearModel_SlowFast
from nets.models.sub_models.MotionModel2 import MotionModel_AM
from torchvision import transforms


class SlowFast_AM(torch.nn.Module):
    def __init__(self, frame_depth, pop_mean, pop_std, eca, shift_factor, group_on):
        super().__init__()
        self.transforms_app = transforms.Compose([
            transforms.Normalize(mean=pop_mean[0], std=pop_std[0]),
            transforms.RandomHorizontalFlip(0.),
            transforms.RandomVerticalFlip(0.)])

        self.transforms_motion = transforms.Compose([
            transforms.Normalize(mean=pop_mean[1], std=pop_std[1]),
            transforms.RandomHorizontalFlip(0.),
            transforms.RandomVerticalFlip(0.)])

        slow_channels = 32
        fast_channels = 32
        self.kernel_size = (3, 3)
        # self.appearance_model_fast = AppearanceModel_2D(eca, fast_channels, self.kernel_size)
        # self.appearance_model_slow = AppearanceModel_2D(eca, slow_channels, self.kernel_size)

        # modified and adopted by Anh
        self.motion_model_fast = MotionModel_AM(eca, fast_channels, self.kernel_size, frame_depth, shift_factor, group_on, 36, 36)
        self.motion_model_slow = MotionModel_AM(eca, slow_channels, self.kernel_size, frame_depth//2, shift_factor, group_on, 36, 36)

        self.hr_linear_model = LinearModel_SlowFast(frame_depth)

    def forward(self, x):
        # X has shape: B, 2, window_length (T), 3, H, W
        fast_motion = x[0]
        slow_motion = x[1]
        # avg_frame = x[2]

        # appearance_input = torch.mean(avg_frame, dim=1) / 255
        # appearance_input = self.transforms_app(appearance_input)  # -> B, C, H, W

        fast_motion = fast_motion / 255
        slow_motion = slow_motion / 255
        fast_motion = self.transforms_motion(fast_motion)  # -> B, T, C, H, W
        slow_motion = self.transforms_motion(slow_motion)

        # attention_mask1, attention_mask2 = self.appearance_model_fast(appearance_input)  # -> B, C, H, W
        # attention_mask1_slow, attention_mask2_slow = self.appearance_model_slow(appearance_input)
        # attention_mask1_slow = attention_mask1_fast.clone()
        # attention_mask2_slow = attention_mask2_fast.clone()
        motion_fast = self.motion_model_fast(fast_motion)
        motion_slow = self.motion_model_slow(slow_motion)
        hr_out = self.hr_linear_model(motion_fast, motion_slow)

        return hr_out

    
