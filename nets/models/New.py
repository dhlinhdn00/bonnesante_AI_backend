import torch
from nets.models.sub_models.LinearModel import LinearModel_SlowFast
from nets.models.sub_models.MotionModel import MotionModel_SF
from torchvision import transforms


class New(torch.nn.Module):
    def __init__(self, frame_depth, pop_mean, pop_std, eca):
        super().__init__()
        self.transforms_app = transforms.Compose([
            transforms.Normalize(mean=pop_mean, std=pop_std),
            transforms.RandomHorizontalFlip(0.),
            transforms.RandomVerticalFlip(0.)])

        self.in_channels = 3
        self.out_channels = 36
        self.kernel_size = (3, 3)

        self.motion_model = MotionModel_SF(self.out_channels, frame_depth)

        self.hr_linear_model = LinearModel_SlowFast(frame_depth)

    def forward(self, x):
        # X has shape: B, T+1, C, H, W
        x = self.transforms_app(x / 255)  # -> B, T+1, C, H, W

        fast, slow = self.motion_model(x)
        hr_out = self.hr_linear_model(fast, slow)

        return hr_out

    def get_attention_mask(self):
        return self.attention_mask1, self.attention_mask2
