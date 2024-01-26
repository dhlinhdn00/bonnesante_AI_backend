import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearModel_MTTS(torch.nn.Module):
    def __init__(self, eca, frame_depth):
        super().__init__()
        self.linear_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2304*frame_depth, 1280),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(1280, frame_depth)
        )

    def forward(self, x):
        out = self.linear_layer.forward(x)
        return out


class LinearModel_STM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(82944, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        out = self.linear_layer.forward(x)
        return out.view(B, T)


class LinearModel_TS_CSTM(torch.nn.Module):
    def __init__(self, eca, frame_depth):
        super().__init__()
        self.linear_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5184*frame_depth, 128*frame_depth),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(128*frame_depth, frame_depth)
        )

    def forward(self, x):
        # B, T, C, H, W = x.shape
        # x = x.view(B*T, C, H, W)
        out = self.linear_layer.forward(x)
        return out


class  LinearModel_SlowFast(torch.nn.Module):
    def __init__(self, frame_depth):
        super().__init__()
        self.fast_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5184*frame_depth, 1280),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(1280, frame_depth)
            )

        self.slow_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2592*frame_depth, 640),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(640, frame_depth)
            )

        self.connect = nn.Sequential(
            nn.Conv3d(2 * 32, 2 * 32, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.Tanh(),
        )

        # self.out = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(int(frame_depth * 2), frame_depth)
        # )

    def forward(self, fast, slow):
        # B, Tf, Cf, H, W = fast.shape
        # _, Ts, Cs, _, _ = slow.shape
        # slow_up = F.interpolate(slow.permute(0, 2, 1, 3, 4), scale_factor=(2, 1, 1), mode='trilinear').permute(0, 2, 1,
        #                                                                                                        3, 4)
        # fast_down = self.connect(fast.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        # fast = (fast + slow_up)
        # slow = (fast_down + slow)
        fast_out = self.fast_linear.forward(fast)
        slow_out = self.slow_linear.forward(slow)
        # concat = torch.cat((fast_out, slow_out), dim=1)
        # out = self.out(concat)
        out = (fast_out + slow_out) / 2   #B,T
        return out


class AFF_SlowFast(nn.Module):
    '''
    多特征融合 AFF
    Source: https://github.com/YimianDai/open-aff/blob/master/aff_pytorch/aff_net/fusion.py ###
    Modified by Anh
    r: channel reduction ratio
    '''

    def __init__(self, channels=64 , r=4, frame_depth = 10):
        super(AFF_SlowFast, self).__init__()
        inter_channels = int(channels // r)
        
        self.linear = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(5184*frame_depth, 1280),   #5184 * frame_depth
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(1280, frame_depth)
            )
        
        self.local_att = nn.Sequential(
            nn.Conv3d(channels, inter_channels, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
            nn.BatchNorm3d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, inter_channels, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
            nn.BatchNorm3d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):  # this is MS-CAM  # x: slow-stream   residual: fast-stream
        # x.shape: B,T/2,C,H,W
        # residual.shape: B,T,C,H,W
        x_up = F.interpolate(x.permute(0, 2, 1, 3, 4), scale_factor=(2, 1, 1), mode='trilinear').permute(0, 2, 1, 3, 4)
        
        xa = x_up + residual                #   B,T,C,H,W
        xa = xa.permute(0, 2, 1, 3, 4)      #   B,C,T,H,W
        xl = self.local_att(xa)             #   B,C,1,1,1
        xg = self.global_att(xa)            #   B,C,T,H,W
        xlg = xl + xg                       #   B,C,T,H,W
        wei = self.sigmoid(xlg)    

        x_up = x_up.permute(0, 2, 1, 3, 4)      #   B,C,T,H,W
        residual = residual.permute(0,2,1,3,4)  #   B,C,T,H,W

        xo = 2 * x_up * abs(wei -0.3) + 2 * residual * abs(wei-0.7)          #   B,C,T,H,W
        
        xo = self.linear(xo.flatten(1))
        return xo