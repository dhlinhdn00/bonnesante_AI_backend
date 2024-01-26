import torch
import numpy as np
from nets.models.sub_models.AppearanceModel import AppearanceModel_2D
from nets.models.sub_models.LinearModel import LinearModel_SlowFast, AFF_SlowFast
from nets.models.sub_models.MotionModel import MotionModel_TS_CSTM
from torchvision import transforms


class  SlowFast_FD(torch.nn.Module):
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

        self.frame_depth = frame_depth
        slow_channels = 32
        fast_channels = 32
        self.kernel_size = (3, 3)
        self.appearance_model_fast = AppearanceModel_2D(eca, fast_channels, self.kernel_size)
        # self.appearance_model_slow = AppearanceModel_2D(eca, slow_channels, self.kernel_size)

        self.motion_model_fast = MotionModel_TS_CSTM(eca, fast_channels, self.kernel_size, frame_depth, shift_factor, group_on)   #shift_factor
        self.motion_model_slow = MotionModel_TS_CSTM(eca, slow_channels, self.kernel_size, frame_depth//2, shift_factor, group_on)

        self.hr_linear_model = LinearModel_SlowFast(frame_depth)
        # self.hr_linear_model = AFF_SlowFast(channels=64, frame_depth = self.frame_depth)         # AFF

    def forward(self, x):
        # X has shape: 3, B, window_length (T), 3, H, W
        fast_motion = x[0]
        slow_motion = x[1]
        avg_frame = x[2]

        appearance_input = torch.mean(avg_frame, dim=1) / 255
        appearance_input = self.transforms_app(appearance_input)  # -> B, C, H, W

        fast_motion = fast_motion / 255
        slow_motion = slow_motion / 255
        fast_motion = self.transforms_motion(fast_motion)  # -> B, T, C, H, W
        slow_motion = self.transforms_motion(slow_motion)

        attention_mask1, attention_mask2 = self.appearance_model_fast(appearance_input)  # -> B, C, H, W
        # attention_mask1_slow, attention_mask2_slow = self.appearance_model_slow(appearance_input)
        # attention_mask1_slow = attention_mask1_fast.clone()
        # attention_mask2_slow = attention_mask2_fast.clone()
        motion_fast = self.motion_model_fast(fast_motion, attention_mask1, attention_mask2)
        motion_slow = self.motion_model_slow(slow_motion, attention_mask1, attention_mask2)
        hr_out = self.hr_linear_model(motion_fast, motion_slow)
        # hr_out = self.hr_linear_model(motion_slow, motion_fast)       # AFF
        
        return hr_out

    def get_attention_mask(self):
        return self.attention_mask1, self.attention_mask2
