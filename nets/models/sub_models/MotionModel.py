import torch
import torch.nn as nn
from nets.blocks.blocks import TSM_Block, ECA_Block, STM, CSTM, TSM_CSTM, TAM, TSM_Block_Adv, CMM, TSM_CSTM_Serial
import numpy as np


class MotionModel(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        # Motion model
        self.motion_tsm1 = nn.Sequential(
            TSM_Block(in_channels, out_channels, skip=False),
            TSM_Block(out_channels, out_channels, skip=skip),
            )
        self.pooling1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(p=0.25)
            )

        self.motion_tsm2 = nn.Sequential(
            TSM_Block(out_channels, out_channels * 2, skip=False),
            TSM_Block(out_channels * 2, out_channels * 2, skip=skip),
            )

        if eca:
            self.eca = True
            self.meca = ECA_Block(out_channels * 2)
        else:
            self.eca = False

        self.pooling2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(p=0.25)
            )

    def forward(self, inputs, mask1, mask2):
        M1 = self.motion_tsm1(inputs)
        B, T, C, H, W = M1.shape
        mask1 = torch.reshape(mask1, (B, 1, C, H, W))
        mask1 = torch.tile(mask1, (1, T, 1, 1, 1))
        M2 = self.pooling1(torch.mul(mask1, M1).view(B*T, C, H, W))
        _, C, H, W = M2.shape
        M3 = self.motion_tsm2(M2.view(B, T, C, H, W))
        B, T, C, H, W = M3.shape
        mask2 = torch.reshape(mask2, (B, 1, C, H, W))
        mask2 = torch.tile(mask2, (1, T, 1, 1, 1))
        g2 = torch.mul(mask2, M3).view(B*T, C, H, W)
        if self.eca:
            g2 = self.meca(g2)
        out = self.pooling2(g2)
        _, C, H, W = out.shape
        return out.view(B, T, C, H, W)


class MotionModel_MTTS(nn.Module):
    def __init__(self, eca, out_channels, shift_factor, group_on):
        super().__init__()
        # Motion model
        self.motion_tsm1 = nn.Sequential(
            TSM_Block(3, out_channels, skip=False, padding='valid', shift_factor=shift_factor, group_on=group_on),
            TSM_Block(out_channels, out_channels, skip=False, padding='valid', shift_factor=shift_factor, group_on=group_on),
            )
        self.pooling1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(p=0.25)
            )

        self.motion_tsm2 = nn.Sequential(
            TSM_Block(out_channels, out_channels * 2, skip=False, padding='valid', shift_factor=shift_factor, group_on=group_on),
            TSM_Block(out_channels * 2, out_channels * 2, skip=False, padding='valid', shift_factor=shift_factor, group_on=group_on),
            )

        if eca:
            self.eca = True
            self.meca = ECA_Block(out_channels * 2)
        else:
            self.eca = False

        self.pooling2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(p=0.25)
            )

    def forward(self, inputs, mask1, mask2):
        M1 = self.motion_tsm1(inputs)
        B, T, C, H, W = M1.shape
        mask1 = torch.reshape(mask1, (B, 1, C, H, W))
        mask1 = torch.tile(mask1, (1, T, 1, 1, 1))
        M2 = self.pooling1(torch.mul(mask1, M1).view(B*T, C, H, W))
        _, C, H, W = M2.shape
        M3 = self.motion_tsm2(M2.view(B, T, C, H, W))
        B, T, C, H, W = M3.shape
        mask2 = torch.reshape(mask2, (B, 1, C, H, W))
        mask2 = torch.tile(mask2, (1, T, 1, 1, 1))
        g2 = torch.mul(mask2, M3).view(B*T, C, H, W)
        if self.eca:
            g2 = self.meca(g2)
        out = self.pooling2(g2)
        _, C, H, W = out.shape
        return out.view(B, T, C, H, W)


class MotionModel_TS_CSTM(nn.Module):
    def __init__(self, eca, in_planes, kernel_size, frame_depth, shift_factor, group_on):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=in_planes, kernel_size=kernel_size, padding='same'),
            nn.LayerNorm([in_planes, 36, 36]),
            nn.Tanh())

        self.block_1 = TSM_CSTM(in_planes, frame_depth, shift_factor, group_on)

        self.pooling1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(p=0.25)
        )

        self.tsm = TSM_Block_Adv(in_planes, in_planes * 2, frame_depth, False, shift_factor, group_on, on=True)

        self.block_2 = TSM_CSTM(in_planes * 2, frame_depth, shift_factor, group_on)

        if eca:
            self.eca = True
            self.meca = ECA_Block(in_planes * 2)
        else:
            self.eca = False

        self.pooling2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(p=0.25)
        )

    def forward(self, inputs, mask1, mask2):
        B, T, C, H, W = inputs.shape
        inputs = inputs.view(B*T, C, H, W)

        F0 = self.conv_1(inputs)
        _, C, H, W = F0.shape
        F1 = self.block_1(F0.view(B, T, C, H, W))
        # F2 = self.tam_1(F1.view(B, T, C, H, W))

        _, C, H, W = F1.shape
        mask1 = torch.reshape(mask1, (B, 1, C, H, W))
        mask1 = torch.tile(mask1, (1, T, 1, 1, 1))
        mask1 = torch.reshape(mask1, (B*T, C, H, W))

        F3 = torch.mul(mask1, F1)
        F4 = self.pooling1(F3)

        _, C, H, W = F4.shape
        F5 = self.tsm(F4.view(B, T, C, H, W))
        F6 = self.block_2(F5)
        # _, C, H, W = F6.shape
        # F7 = self.tam_2(F6.view(B, T, C, H, W))

        _, C, H, W = F6.shape
        mask2 = torch.reshape(mask2, (B, 1, C, H, W))
        mask2 = torch.tile(mask2, (1, T, 1, 1, 1))
        mask2 = torch.reshape(mask2, (B*T, C, H, W))

        F8 = torch.mul(mask2, F6)

        if self.eca:
            F9 = self.meca(F8)
        else:
            F9 = F8

        out = self.pooling2(F9)
        _, C, H, W = out.shape
        return out.view(B, T, C, H, W)


class MotionModel_New(nn.Module):
    def __init__(self, eca, in_channels, out_channels, kernel_size, frame_depth):
        super().__init__()
        self.cmm = CMM(out_channels)
        self.block_1 = TSM_CSTM(out_channels, frame_depth, 36, 36)

        self.pooling1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(p=0.25)
        )

        if frame_depth == 10:
            self.tsm = TSM_Block_Adv(out_channels, out_channels * 2, 7, False, 18, 18)
        else:
            self.tsm = TSM_Block_Adv(out_channels, out_channels * 2, 17, False, 18, 18)

        self.block_2 = TSM_CSTM(out_channels * 2, frame_depth, 18, 18)

        if eca:
            self.eca = True
            self.meca = ECA_Block(out_channels * 2)
        else:
            self.eca = False

        self.pooling2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(p=0.25)
        )

    def forward(self, inputs):
        # B, T, C, H, W = inputs.shape

        F1 = self.cmm(inputs)  # -> T = T-1
        B, T, C, H, W = F1.shape
        F2 = self.block_1(F1)

        F4 = self.pooling1(F2)

        _, C, H, W = F4.shape
        F5 = self.tsm(F4.view(B, T, C, H, W))
        F6 = self.block_2(F5)

        if self.eca:
            F9 = self.meca(F6)
        else:
            F9 = F6

        out = self.pooling2(F9)
        _, C, H, W = out.shape
        return out.view(B, T, C, H, W)


class MotionModel_SF(nn.Module):
    def __init__(self, in_planes, frame_depth, group_on):
        super().__init__()
        f_in_planes = in_planes
        s_in_planes = in_planes*2

        self.conv_f = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=f_in_planes, kernel_size=(3, 3), padding='same'),
            nn.LayerNorm([f_in_planes, 36, 36]),
            nn.Tanh())

        self.conv_s = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=s_in_planes, kernel_size=(3, 3), padding='same'),
            nn.LayerNorm([s_in_planes, 36, 36]),
            nn.Tanh())

        self.block_F_1 = TSM_CSTM(f_in_planes, frame_depth, 36, group_on)
        self.block_S_1 = TSM_CSTM(s_in_planes, frame_depth//2, 36, group_on)

        self.pooling1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(p=0.25)
        )

        # self.connect_1 = nn.Sequential(
        #     nn.Conv3d(f_in_planes, f_in_planes, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
        #     nn.Tanh(),
        # )

        self.tsm_S = TSM_Block_Adv(s_in_planes, s_in_planes * 2, frame_depth//2, False, 18, group_on)
        self.tsm_F = TSM_Block_Adv(f_in_planes, f_in_planes * 2, frame_depth, False, 18, group_on)

        # self.connect_2 = nn.Sequential(
        #     nn.Conv3d(2*f_in_planes, 2*f_in_planes, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
        #     nn.Tanh(),
        # )

        self.block_S_2 = TSM_CSTM(s_in_planes * 2, frame_depth//2, 18, group_on)
        self.block_F_2 = TSM_CSTM(f_in_planes * 2, frame_depth, 18, group_on)

        self.pooling2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(p=0.25)
        )

        # self.connect_3 = nn.Sequential(
        #     nn.Conv3d(f_in_planes * 2, f_in_planes * 2, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
        #     nn.Tanh(),
        # )

    def forward(self, inputs, mask1, mask2):
        B, T, C, H, W = inputs.shape
        Tf = T
        Ts = Tf//2
        i = np.arange(0, Tf, 2)
        slow_inputs = inputs[:, i]
        fast_inputs = inputs

        slow_inputs = slow_inputs.view(B*Ts, C, H, W)
        fast_inputs = fast_inputs.view(B*Tf, C, H, W)

        F1 = self.conv_f(fast_inputs)
        S1 = self.conv_s(slow_inputs)

        _, Cf, _, _ = F1.shape
        _, Cs, _, _ = S1.shape

        F2 = self.block_F_1(F1.view(B, Tf, Cf, H, W))
        S2 = self.block_S_1(S1.view(B, Ts, Cs, H, W))

        F2 = F2.view(B, Tf, Cf, H, W)
        S2 = S2.view(B, Ts, Cs, H, W)

        # M1 = self.connect_1(F2.permute(0, 2, 1, 3, 4))
        # S4 = torch.cat((S3, M1.permute(0, 2, 1, 3, 4)), dim=2)
        # S2_c = M1.permute(0, 2, 1, 3, 4) + S2

        # mask1 = torch.reshape(mask1, (B, 1, Cf, H, W))
        # mask1_f = torch.tile(mask1, (1, Tf, 1, 1, 1))
        # mask1_s = torch.tile(mask1, (1, Ts, 1, 1, 1))

        # F3 = torch.mul(mask1_f, F2)
        # S3 = torch.mul(mask1_s, S2)

        F4 = self.pooling1(F2.view(B*Tf, Cf, H, W))
        S4 = self.pooling1(S2.view(B*Ts, Cs, H, W))

        _, _, H, W = F4.shape
        F4 = F4.view(B, Tf, Cf, H, W)
        S4 = S4.view(B, Ts, Cs, H, W)

        F5 = self.tsm_F(F4)   # -> Double channels
        S5 = self.tsm_S(S4)

        # M2 = self.connect_2(F5.permute(0, 2, 1, 3, 4))
        # # S5 = torch.concat((S4, M2.permute(0, 2, 1, 3, 4)), dim=2)
        # S5_c = M2.permute(0, 2, 1, 3, 4) + S5

        F6 = self.block_F_2(F5)
        S6 = self.block_S_2(S5)

        _, Cf, H, W = F6.shape
        _, Cs, _, _ = S6.shape
        F6 = F6.view(B, Tf, Cf, H, W)
        S6 = S6.view(B, Ts, Cs, H, W)

        # M3 = self.connect_3(F5.permute(0, 2, 1, 3, 4))
        # # S8 = torch.concat((F6, M3.permute(0, 2, 1, 3, 4)), dim=2)
        # S6_c = M3.permute(0, 2, 1, 3, 4) + S6

        # mask2 = torch.reshape(mask2, (B, 1, Cf, H, W))
        # mask2_f = torch.tile(mask2, (1, Tf, 1, 1, 1))
        # mask2_s = torch.tile(mask2, (1, Ts, 1, 1, 1))

        # F7 = torch.mul(mask2_f, F6)
        # S7 = torch.mul(mask2_s, S6)

        F8 = self.pooling2(F6.view(B*Tf, Cf, H, W))
        S8 = self.pooling2(S6.view(B*Ts, Cs, H, W))

        _, Cf, H, W = F8.shape
        Cs = S8.shape[1]
        F9 = F8.view(B, Tf, Cf, H, W)
        S9 = S8.view(B, Ts, Cs, H, W)

        return F9, S9


class MotionModel_SF_CMM(nn.Module):
    def __init__(self, in_planes, frame_depth):
        super().__init__()
        f_in_planes = in_planes
        s_in_planes = in_planes

        self.cmm_F = CMM(f_in_planes)
        self.cmm_S = CMM(s_in_planes)

        self.block_F_1 = TSM_CSTM(f_in_planes, frame_depth, 36)
        self.block_S_1 = TSM_CSTM(s_in_planes, frame_depth//2, 36)

        self.pooling1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(p=0.25)
        )

        self.tsm_S = TSM_Block_Adv(s_in_planes, s_in_planes * 2, frame_depth // 2, False, 18)
        self.tsm_F = TSM_Block_Adv(f_in_planes, f_in_planes * 2, frame_depth, False, 18)

        self.block_S_2 = TSM_CSTM(2 * s_in_planes, frame_depth//2, 18)
        self.block_F_2 = TSM_CSTM(f_in_planes * 2, frame_depth, 18)

        self.pooling2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(p=0.25)
        )

    def forward(self, inputs):
        B, T, C, H, W = inputs.shape
        Tf = 20
        Ts = 10

        fast_inputs = inputs[:, :-1]
        i = np.arange(0, 22, 2)
        slow_inputs = inputs[:, i]

        F1 = self.cmm_F(fast_inputs)  # -> T = T-1
        S1 = self.cmm_S(slow_inputs)

        _, _, Cf, _, _ = F1.shape
        _, _, Cs, _, _ = S1.shape

        F2 = self.block_F_1(F1.view(B, Tf, Cf, H, W))
        S2 = self.block_S_1(S1.view(B, Ts, Cs, H, W))
        F2 = F2.view(B, Tf, Cf, H, W)
        S2 = S2.view(B, Ts, Cs, H, W)

        # M1 = self.connect_1(F2.permute(0, 2, 1, 3, 4))
        # S4 = torch.cat((S3, M1.permute(0, 2, 1, 3, 4)), dim=2)
        # S2_c = M1.permute(0, 2, 1, 3, 4) + S2

        F3 = self.pooling1(F2.view(B * Tf, Cf, H, W))
        S3 = self.pooling1(S2.reshape(B * Ts, Cs, H, W))

        _, _, H, W = F3.shape
        F3 = F3.view(B, Tf, Cf, H, W)
        S3 = S3.view(B, Ts, Cs, H, W)

        F4 = self.tsm_F(F3)  # -> Double channels
        S4 = self.tsm_S(S3)

        # M2 = self.connect_2(F4.permute(0, 2, 1, 3, 4))
        # S5 = torch.concat((S4, M2.permute(0, 2, 1, 3, 4)), dim=2)
        # S4_c = M2.permute(0, 2, 1, 3, 4) + S4

        F5 = self.block_F_2(F4)
        S5 = self.block_S_2(S4)

        _, Cf, H, W = F5.shape
        _, Cs, _, _ = S5.shape
        F6 = F5.view(B, Tf, Cf, H, W)
        S6 = S5.view(B, Ts, Cs, H, W)

        # M3 = self.connect_3(F6.permute(0, 2, 1, 3, 4))
        # S8 = torch.concat((F6, M3.permute(0, 2, 1, 3, 4)), dim=2)
        # S6_c = M3.permute(0, 2, 1, 3, 4) + S6

        F7 = self.pooling2(F6.view(B * Tf, Cf, H, W))
        S7 = self.pooling2(S6.reshape(B * Ts, Cs, H, W))

        _, Cf, H, W = F7.shape
        Cs = S7.shape[1]
        F8 = F7.view(B, Tf, Cf, H, W)
        S8 = S7.view(B, Ts, Cs, H, W)

        return F8, S8


class MotionModel_STM(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        # Motion model
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3, in_planes, kernel_size=3, stride=1, padding="same", bias=True),
            nn.Tanh())  #-> Batch, T, in_planes, W/2, H/2

        self.STM_1_1 = STM(in_planes, channel_reductor=2, no_groups=8)
        self.STM_1_2 = STM(in_planes, channel_reductor=2, no_groups=8)
        self.STM_1_3 = STM(in_planes, channel_reductor=2, no_groups=8)

        self.pooling_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2)))

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes*2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh())

        self.STM_2_1 = STM(in_planes*2, channel_reductor=2, no_groups=8)
        self.STM_2_2 = STM(in_planes*2, channel_reductor=2, no_groups=8)
        self.STM_2_3 = STM(in_planes*2, channel_reductor=2, no_groups=8)
        self.STM_2_4 = STM(in_planes*2, channel_reductor=2, no_groups=8)


    def forward(self, x, mask_1, mask_2):
        # x has shape: B, T + 1, C, H, W: including the next frame
        B, T, C, H, W = x.shape
        A1 = self.conv_block_1.forward(x.view(B*T, C, H, W))
        _, C, H, W = A1.shape
        A1 = A1.view(B, T, C, H, W)

        A2 = torch.tanh(A1 + self.STM_1_1.forward(A1))
        A3 = torch.tanh(A2 + self.STM_1_2.forward(A2))
        A4 = torch.tanh(A3 + self.STM_1_3.forward(A3))

        _, _, C, H, W = A4.shape
        mask_1 = mask_1.view(B, T, C, H, W)
        G1 = torch.tanh(torch.mul(A4, mask_1))

        A5 = self.conv_block_2.forward(self.pooling_1(G1.view(B*T, C, H, W)))
        _, C, H, W = A5.shape
        A5 = A5.view(B, T, C, H, W)
        A6 = torch.tanh(A5 + self.STM_2_1.forward(A5))
        A7 = torch.tanh(A6 + self.STM_2_2.forward(A6))
        A8 = torch.tanh(A7 + self.STM_2_3.forward(A7))
        A9 = torch.tanh(A8 + self.STM_2_4.forward(A8))
        mask_2 = mask_2.view(B, T, C, H, W)

        G2 = torch.tanh(torch.mul(A9, mask_2))   # -> B, T, C, H, W

        return G2[:, :-1, :, :, :]
