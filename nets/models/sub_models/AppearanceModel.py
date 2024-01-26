import torch
import torch.nn as nn

from nets.blocks.attentionBlocks import AttentionBlock
from nets.blocks.blocks import ECA_Block


class AppearanceModel_2D(nn.Module):
    def __init__(self, eca, in_planes, kernel_size=(3, 3)):
        # Appearance model
        super().__init__()
        self.conv1 = nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=in_planes, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(True))

        self.conv2 = nn.Sequential(
            torch.nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(True))

        if eca:
            self.eca1 = ECA_Block(in_planes)
            self.eca2 = ECA_Block(in_planes * 2)
            self.eca = True
        else:
            self.eca = False

        # Attention mask1
        self.attention_mask1 = AttentionBlock(in_planes)

        self.pooling = nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=(2, 2)), torch.nn.Dropout2d(p=0.25))

        self.conv3 = nn.Sequential(
            torch.nn.Conv2d(in_channels=in_planes, out_channels=in_planes * 2, kernel_size=kernel_size,
                            padding='same'),
            nn.BatchNorm2d(in_planes * 2),
            nn.ReLU(True))
        self.conv4 = nn.Sequential(
            torch.nn.Conv2d(in_channels=in_planes * 2, out_channels=in_planes * 2, kernel_size=kernel_size,
                            padding='same'),
            nn.BatchNorm2d(in_planes * 2),
            nn.ReLU(True))

        # Attention mask2
        self.attention_mask2 = AttentionBlock(in_planes * 2)

    def forward(self, inputs):
        # inputs has shape B, C, H, W
        if self.eca:
            A1 = self.eca1(self.conv1(inputs))
        else:
            A1 = self.conv1(inputs)
        A2 = self.conv2(A1)
        # Calculate Mask1
        M1 = self.attention_mask1(A2)
        # Pooling and Dropout
        A3 = self.conv3(self.pooling(A2))

        if self.eca:
            M2 = self.attention_mask2(self.conv4(self.eca1(A3)))
        else:
            M2 = self.attention_mask2(self.conv4(A3))

        return M1, M2


class AppearanceModel_MTTS(nn.Module):
    def __init__(self, eca, in_planes, kernel_size=(3, 3)):
        # Appearance model
        super().__init__()
        self.conv1 = nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=in_planes, kernel_size=kernel_size, padding='valid'),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(True))

        self.conv2 = nn.Sequential(
            torch.nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=kernel_size, padding='valid'),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(True))

        if eca:
            self.eca1 = ECA_Block(in_planes)
            self.eca2 = ECA_Block(in_planes * 2)
            self.eca = True
        else:
            self.eca = False

        # Attention mask1
        self.attention_mask1 = AttentionBlock(in_planes)

        self.pooling = nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=(2, 2)), torch.nn.Dropout2d(p=0.25))

        self.conv3 = nn.Sequential(
            torch.nn.Conv2d(in_channels=in_planes, out_channels=in_planes * 2, kernel_size=kernel_size,
                            padding='valid'),
            nn.BatchNorm2d(in_planes * 2),
            nn.ReLU(True))
        self.conv4 = nn.Sequential(
            torch.nn.Conv2d(in_channels=in_planes * 2, out_channels=in_planes * 2, kernel_size=kernel_size,
                            padding='valid'),
            nn.BatchNorm2d(in_planes * 2),
            nn.ReLU(True))

        # Attention mask2
        self.attention_mask2 = AttentionBlock(in_planes * 2)

    def forward(self, inputs):
        # inputs has shape B, C, H, W
        if self.eca:
            A1 = self.eca1(self.conv1(inputs))
        else:
            A1 = self.conv1(inputs)
        A2 = self.conv2(A1)
        # Calculate Mask1
        M1 = self.attention_mask1(A2)
        # Pooling and Dropout
        A3 = self.conv3(self.pooling(A2))

        if self.eca:
            M2 = self.attention_mask2(self.conv4(self.eca1(A3)))
        else:
            M2 = self.attention_mask2(self.conv4(A3))

        return M1, M2


class AppearanceModel_STM(nn.Module):
    def __init__(self, eca, in_channels, out_channels, kernel_size=(3, 3)):
        # Appearance model
        super().__init__()
        self.a_conv1 = nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                            padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))

        self.a_conv2 = nn.Sequential(
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                            padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))

        # ECA module for TSDAN:
        # if eca:
        #     # self.a_eca1 = ECA_Block(out_channels)
        #     # self.a_eca2 = ECA_Block(out_channels * 2)
        #     # self.a_eca3 = ECA_Block(out_channels * 4)
        #     # self.eca = True
        # else:
        self.eca = False

        # Attention mask1
        self.attention_mask1 = AttentionBlock(out_channels)

        self.pooling_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(p=0.25))

        self.a_conv3 = nn.Sequential(
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=kernel_size,
                            padding='same'),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(True))

        self.a_conv4 = nn.Sequential(
            torch.nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels * 2, kernel_size=kernel_size,
                            padding='same'),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(True))

        # Attention mask2
        self.attention_mask2 = AttentionBlock(out_channels * 2)

        # self.pooling_2 = nn.Sequential(
        #     nn.AvgPool2d(kernel_size=(2, 2)),
        #     nn.Dropout2d(p=0.25))

        # self.a_conv5 = nn.Sequential(
        #     nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels * 4, kernel_size=kernel_size, padding='same'),
        #     nn.BatchNorm2d(out_channels * 4),
        #     nn.ReLU(True))
        # self.a_conv6 = nn.Sequential(
        #     nn.Conv2d(in_channels=out_channels * 4, out_channels=out_channels * 4, kernel_size=kernel_size, padding='same'),
        #     nn.BatchNorm2d(out_channels * 4),
        #     nn.ReLU(True))
        # # Attention mask 3
        # self.attention_mask3 = AttentionBlock(out_channels * 4)

    def forward(self, inputs):
        # inputs has shape B, T, C, H, W
        B, T, C, H, W = inputs.shape
        inputs = inputs.view(B * T, C, H, W)
        if self.eca:
            F0 = self.a_eca1(self.a_conv1(inputs))
        else:
            F0 = self.a_conv1(inputs)
        F1 = self.a_conv2(F0)
        M1 = self.attention_mask1(F1)

        F2 = self.a_conv3(self.pooling_1(F1))
        if self.eca:
            F3 = self.a_conv4(self.a_eca2(F2))
        else:
            F3 = self.a_conv4(F2)
        M2 = self.attention_mask2(F3)

        # F4 = self.a_conv5(self.pooling_2(F3))
        # if self.eca:
        #     F5 = self.a_conv6(self.a_eca3(F4))
        # else:
        #     F5 = self.a_conv6(F4)
        # M3 = self.attention_mask3(F5)
        return M1, M2


