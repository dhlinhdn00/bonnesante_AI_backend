import torch


class AttentionBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = torch.nn.Conv2d(in_channels, 1, kernel_size=(1, 1), padding='same')

    def forward(self, input):
        mask = torch.sigmoid(self.attention(input))
        _, _, H, W = input.shape
        xsum = torch.sum(mask, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        return input / xsum * H * W * .5
