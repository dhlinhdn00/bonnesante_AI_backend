import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import init
import functools
from torch.autograd import Variable
import math


class TSM(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, shift_factor, fold_div=3):
        B, T, C, H, W = x.shape
        fold = C // fold_div
        out = torch.zeros_like(x)
        k = int(np.floor(T*shift_factor))
        out[:, :-k, :fold] = x[:, k:, :fold]
        out[:, k:, fold: 2 * fold] = x[:, :-k, fold: 2 * fold]
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

        return out


class TSM_New(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, shift_factor, elements=3):
        B, T, C, H, W = x.shape
        idx_for = np.arange(0, C, elements)
        if idx_for[-1] >= C - 1:
            idx_for = idx_for[:-1]
        idx_back = idx_for + 1
        if idx_back[-1] >= C - 1:
            idx_back = idx_back[:-1]

        k = int(np.floor(T * shift_factor))
        out = x.clone()
        # Forward:
        out[:, :-k, idx_for] = x[:, k:, idx_for]
        out[:, :-k, idx_for] = 0

        # Backward:
        out[:, k:, idx_back] = x[:, :-k, idx_back]
        out[:, :k, idx_back] = 0

        return out


class TSM_Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, padding, skip, shift_factor, group_on):
        super().__init__()
        if group_on:
            self.tsm = TSM_New()
        else:
            self.tsm = TSM()
        self.conv_2d = nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=padding),
            nn.Tanh())

        self.skip = skip
        self.shift_factor = shift_factor

    def forward(self, x):
        # X-> B, T, C, H, W
        A = self.tsm(x, self.shift_factor)
        b, t, c, h, w = A.shape
        out = self.conv_2d(A.reshape(b*t, c, h, w))
        _, c, h, w = out.shape
        if self.skip:
            return out.view(b, t, c, h, w) + x
        else:
            return out.view(b, t, c, h, w)


class TSM_Block_Adv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, frame_depth, skip, shift_factor, group_on, on=True):
        super().__init__()
        if group_on:
            self.tsm = TSM_New()
        else:
            self.tsm = TSM()
        self.activate = on
        if self.activate:
            self.Enhancer = Local_Tempo_Enhancer(in_channels, frame_depth)
        self.conv_2d = nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'),
            # nn.LayerNorm([out_channels, size, size]),
            nn.Tanh())
        self.skip = skip
        self.shift_factor = shift_factor

    def forward(self, x):
        # X-> B, T, C, H, W
        b, t, c, h, w = x.shape
        if self.activate:
            F0 = self.Enhancer(x)
        else:
            F0 = x
        F1 = self.tsm(F0, self.shift_factor)
        F1 = self.conv_2d(F1.reshape(b*t, c, h, w))
        c = F1.shape[1]
        F2 = F1.view(b, t, c, h, w)
        if self.skip:
            return F2 + F0
        else:
            return F2


class TIM_Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, tempo_kernel, skip, h, w):
        super().__init__()
        self.tim = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding='same'),
            nn.LayerNorm([out_channels, ])
        )

        self.conv_2d = nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'),
            nn.LayerNorm([out_channels, h, w]),
            nn.Tanh())
        self.skip = skip

    def forward(self, x):
        # X-> B, T, C, H, W
        F0 = self.tsm(x)
        b, t, c, h, w = F0.shape
        F1 = self.conv_2d(F0.reshape(b*t, c, h, w))
        c = F1.shape[1]
        F2 = self.Enhancer(F1.view(b, t, c, h, w))
        if self.skip:
            return F2 + x
        else:
            return F2


class ECA_Block(nn.Module):
    """Constructs an ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, gamma=2, b=1):
        super(ECA_Block, self).__init__()
        t = int(abs((np.log2(channel) + b) / gamma))
        if (t % 2) != 0:
            k = t
        else:
            k = t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int((k-1)/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class CSTM(torch.nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=in_planes, out_channels=in_planes, kernel_size=7, padding='same', groups=in_planes),
            nn.Tanh()
        )
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=(3, 3), padding='same'),
            nn.Tanh()
        )

    def forward(self, x):
        # X has shape B, T, C, H, W
        x = x.permute(0, 3, 4, 2, 1)
        B, H, W, C, T = x.shape
        x = x.reshape(B*H*W, C, T)
        A1 = self.conv1d(x)
        A1 = A1.view(B, H, W, C, T)
        A1 = A1.permute(0, 4, 3, 1, 2)
        A1 = A1.reshape(B*T, C, H, W)
        A2 = self.conv2d(A1)

        return A2  # -> B*T, C, H, W


class CMM(torch.nn.Module):
    def __init__(self, in_planes):  # -> STM paper suggests 16 groups
        super().__init__()
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(3, in_planes, kernel_size=3, padding='same', groups=3),
            nn.LayerNorm([in_planes, 36, 36]),
            nn.ReLU(True),
            nn.Conv2d(in_planes, in_planes*2, kernel_size=3, stride=2, padding=0, groups=3),
            nn.LayerNorm([in_planes*2, 17, 17]),
            nn.ReLU(True)
        )

        self.conv2d_2 = nn.Sequential(
            nn.ConvTranspose2d(in_planes*2, in_planes, kernel_size=3, stride=2, output_padding=1),
            nn.LayerNorm([in_planes, 36, 36]),
            nn.Tanh(),
            nn.Conv2d(in_planes, in_planes, kernel_size=3, padding='same'),
            nn.LayerNorm([in_planes, 36, 36]),
            nn.Tanh()
        )
        self.Spatial_Attention = SpatialGate(17, 17)

    def forward(self, x):
        # X has shape B, T, C, H, W
        B, T, C, H, W = x.shape
        F1 = self.conv2d_1(x.reshape(B*T, C, H, W))
        _, C, H, W = F1.shape
        F1 = F1.view(B, T, C, H, W)
        F2 = torch.diff(F1, dim=1)   # -> B, T-1, C, H, W
        F3 = F2.view(B*(T-1), C, H, W)
        F4 = self.Spatial_Attention(F3)
        F5 = self.conv2d_2(F4)
        _, C, H, W = F5.shape
        return F5.view(B, T-1, C, H, W)


class STM(torch.nn.Module):
    def __init__(self, in_planes, channel_reductor, no_groups):  # -> in_channels = 32 channels
        super().__init__()
        # self.eca = ECA_Block(in_planes)
        self.conv2d_start = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=int(in_planes/channel_reductor), kernel_size=(1, 1),
                      padding='same'),
            nn.Tanh()
        )

        self.conv2d_end = nn.Sequential(
            nn.Conv2d(in_channels=int(in_planes/channel_reductor), out_channels=in_planes, kernel_size=(1, 1),
                      padding='same'),
            nn.Tanh()
        )

        self.cstm = CSTM(int(in_planes/channel_reductor))
        self.cmm = CMM(int(in_planes/channel_reductor), no_groups)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        # A0 = self.eca(x)
        A1 = self.conv2d_start(x)

        _, C, H, W = A1.shape
        A1 = A1.view(B, T, C, H, W)
        # last_frame = torch.unsqueeze(A1[:, -1, :, :, :], dim=1)  # -> B, C, H, W
        # A2 = A1[:, :-1, :, :, :]  # -> B, T-1, C, H, W

        cstm = self.cstm(A1)
        motion = self.cmm(A1)
        A3 = torch.tanh(cstm + motion)

        # last_frame_feature = torch.unsqueeze(last_frame_feature, dim=1)
        # A3 = torch.cat((A3, last_frame_feature), dim=1)
        A4 = self.conv2d_end(A3)  # -> BT, C, H, W
        _, C, H, W = A4.shape
        return A4.view(B, T, C, H, W)


class TSM_CSTM(torch.nn.Module):
    def __init__(self, in_planes, frame_depth, shift_factor, group_on, h, w, am = False):  # -> in_channels = 32 channels
        super().__init__()
        self.tsm = TSM_Block_Adv(in_planes, in_planes, frame_depth, True, shift_factor, group_on, on=False)
        self.enhancer_1 = Local_Tempo_Enhancer(in_planes, frame_depth)
        # self.enhancer_2 = Local_Tempo_Enhancer(in_planes, frame_depth)
        if am:
            self.new_am = True
            self.am = AttentionModule(in_planes, in_planes, size1=(h,w), size2=(math.ceil(h/2),math.ceil(w/2)), size3=(math.ceil(h/4),math.ceil(w/4)))
        else:
            self.new_am = False
        if frame_depth % 2 == 0:
            kernel_size = frame_depth - 3
        else:
            kernel_size = frame_depth - 2

        self.conv1d = nn.Sequential(
            nn.Conv1d(in_planes, in_planes, kernel_size=kernel_size, padding='same', groups=in_planes),
            # nn.LayerNorm([in_planes, frame_depth]),
            nn.Tanh()
        )

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=(3, 3), padding='same'),
            # nn.LayerNorm([in_planes, size, size]),
            nn.Tanh()
        )

    def forward(self, x):
        # x has shape B, T, C, H, W
        F1 = self.enhancer_1(x) # tam
        F2 = F1.permute(0, 3, 4, 2, 1)
        I1 = self.tsm(F1)
        
        B, H, W, C, T = F2.shape
        # print(B,H,W,C,T)
        
        F2 = F2.reshape(B*H*W, C, T)
        F3 = self.conv1d(F2)
        F3 = F3.view(B, H, W, C, T)
        F3 = F3.permute(0, 4, 3, 1, 2)
        F3 = F3.reshape(B*T, C, H, W)
        F4 = self.conv2d(F3)
        F4 = F4.view(B, T, C, H, W)
        F5 = torch.add(I1, F4)
        if self.new_am:
            F0 = F1.reshape(B*T, C, H, W)
            I2 = self.am(F0)
            I2 = I2.view(B, T, C, H, W)
            F6 = torch.add(I2, F5)
        else: F6 = F5
        # F6 = self.enhancer_2(F5)
        #return F5.view(B*T, C, H, W)
        return F6.view(B*T, C, H, W)


class TSM_CSTM_Serial(torch.nn.Module):
    def __init__(self, in_planes, frame_depth, c, h, w):  # -> in_channels = 32 channels
        super().__init__()
        if frame_depth <= 10:
            kernel_size = 7
        else:
            kernel_size = 19

        self.tsm = TSM_Block(in_planes, in_planes, True)
        # self.L = nn.Sequential(
        #     nn.Conv1d(in_planes, in_planes, kernel_size=kernel_size, padding='same', bias=False),
        #     nn.LayerNorm([c, 10]),
        #     nn.Tanh(),
        #     nn.Conv1d(in_planes, in_planes, 1, bias=False),
        #     nn.Sigmoid())

        self.frame_depth = frame_depth
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=in_planes, out_channels=in_planes, kernel_size=kernel_size, padding='same', groups=in_planes),
            nn.LayerNorm([c, 10]),
            nn.Tanh()
        )

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=(3, 3), padding='same'),
            nn.LayerNorm([c, h, w]),
            nn.Tanh()
        )

    def forward(self, x):
        # x has shape B, T, C, H, W
        I1 = self.tsm(x)

        F1 = x.permute(0, 3, 4, 2, 1)
        b, h, w, c, t = F1.shape
        F1 = F1.reshape(b*h*w, c, t)
        F2 = self.conv1d(F1)
        F2 = F2.view(b, h, w, c, t)
        F2 = F2.permute(0, 4, 3, 1, 2)
        F3 = F2.reshape(b*t, c, h, w)
        F4 = self.conv2d(F3)
        _, c, h, w = F4.shape

        F5 = x + F4.view(b, t, c, h, w)
        F6 = torch.mul(I1, F5)
        b, t, c, h, w = F6.shape
        return F6.reshape(b*t, c, h, w)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Channel_Gate(torch.nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super().__init__()

        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.Tanh(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = ['avg', 'max']

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class SpatialGate(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=(kernel_size-1) // 2, bias=False),
            nn.LayerNorm([1, h, w]),
            nn.ReLU(True),
            nn.Sigmoid()) 

    def forward(self, x):
        x_compress = self.compress(x)
        scale = self.spatial(x_compress)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):  # Accepting 4D input
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class Local_Tempo_Enhancer(torch.nn.Module):
    def __init__(self, in_channels, frame_depth, reductor=1):
        super().__init__()
        if frame_depth % 2 == 0:
            kernel_size = frame_depth - 3
        else:
            kernel_size = frame_depth - 2

        self.L = nn.Sequential(
            # nn.Linear(10, 10),
            nn.Conv1d(in_channels, in_channels // reductor, kernel_size, padding='same', bias=False),
            nn.LayerNorm([in_channels // reductor, frame_depth]),
            nn.Tanh(),
            nn.Conv1d(in_channels // reductor, in_channels, 1, bias=False),
            # nn.Linear(10, 10),
            nn.Sigmoid())

    def forward(self, x):
        # X -> B, T, C, H, W
        b, t, c, h, w = x.shape
        new_x = x.permute(0, 2, 1, 3, 4).contiguous()  # B, T, C, H, W -> B, C, T, H, W
        out = F.adaptive_avg_pool2d(new_x.view(b * c, t, h, w), (1, 1))   # Total number of channels, compress all spatial info
        out = out.view(b, c, t)
        local_activation = self.L(out).view(b, c, t, 1, 1)
        new_x = new_x * local_activation

        return new_x.view(b, t, c, h, w)


class TAM(torch.nn.Module):
    def __init__(self, in_channels, n_segment, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.n_segment = n_segment
        self.kernel_size = 7
        self.padding = 3
        self.stride = stride

        self.G = nn.Sequential(
            nn.Linear(n_segment, n_segment * 2, bias=False),
            nn.Tanh(),
            nn.Linear(n_segment * 2, self.kernel_size, bias=False),
            nn.Softmax(-1))

        self.L = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 4, self.kernel_size, stride=1, padding=self.kernel_size // 2, bias=False),
            nn.Tanh(),
            nn.Conv1d(in_channels // 4, in_channels, 1, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        # X -> B, T, C, H, W
        b, t, c, h, w = x.shape
        new_x = x.permute(0, 2, 1, 3, 4). vcontiguous()  # B, T, C, H, W -> B, C, T, H, W
        out = F.adaptive_avg_pool2d(new_x.view(b * c, t, h, w), (1, 1))   # Total number of channels, compress all spatial info
        out = out.view(-1, t)
        conv_kernel = self.G(out).view(b * c, 1, -1, 1)
        local_activation = self.L(out.view(b, c, t)).view(b, c, t, 1, 1)
        new_x = new_x * local_activation
        out = F.conv2d(new_x.view(1, b * c, t, h * w), conv_kernel, bias=None, stride=(self.stride, 1),
                       padding=(self.padding, 0), groups=b * c)
        out = out.view(b, c, t, h, w)
        out = out.permute(0, 2, 1, 3, 4).contiguous()

        return out.view(b*t, c, h, w)


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, h, w, stride=1):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = nn.LayerNorm([input_channels, h, w])
        # self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, output_channels//4, 1, 1, bias = False)
        # self.bn2 = nn.BatchNorm2d(output_channels//4)
        self.bn2 = nn.LayerNorm([output_channels//4, h, w])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channels//4, output_channels//4, 3, stride, padding = 1, bias = False)
        # self.bn3 = nn.BatchNorm2d(output_channels//4)
        self.bn3 = nn.LayerNorm([output_channels//4, h, w])
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_channels//4, output_channels, 1, 1, bias = False)
        self.conv4 = nn.Conv2d(input_channels, output_channels , 1, stride, bias = False)
        
    def forward(self, x):
        residual = x
        # print("ShapeL: ", x.shape)
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.input_channels != self.output_channels) or (self.stride !=1 ):
            residual = self.conv4(out1)
        out += residual
        return out

# got from github  -  Anh
class AttentionModule(nn.Module):

    #input size is 36x36
    def __init__(self, in_channels, out_channels, size1, size2, size3):
        super(AttentionModule, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels, size1[0], size1[1])

        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels, size1[0], size1[1]),
            ResidualBlock(in_channels, out_channels, size1[0], size1[1])
         )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax1_blocks = ResidualBlock(in_channels, out_channels, size2[0], size2[1])

        self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels, size2[0], size2[1])

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax2_blocks = ResidualBlock(in_channels, out_channels, size3[0], size3[1])

        self.skip2_connection_residual_block = ResidualBlock(in_channels, out_channels, size3[0], size3[1])

        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax3_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels, size3[0]//2+1, size3[1]//2+1),
            ResidualBlock(in_channels, out_channels, size3[0]//2+1, size3[1]//2+1)
        )

        self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)

        self.softmax4_blocks = ResidualBlock(in_channels, out_channels, size3[0], size3[1])

        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)

        self.softmax5_blocks = ResidualBlock(in_channels, out_channels, size2[0], size2[1])

        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.softmax6_blocks = nn.Sequential(
            # nn.BatchNorm2d(out_channels),
            nn.LayerNorm([out_channels,size1[0], size1[1]]),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels , kernel_size = 1, stride = 1, bias = False),
            # nn.BatchNorm2d(out_channels),
            nn.LayerNorm([out_channels,size1[0], size1[1]]),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels , kernel_size = 1, stride = 1, bias = False),
            nn.Sigmoid()
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels, size1[0], size1[1])

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
        out_mpool3 = self.mpool3(out_softmax2)
        out_softmax3 = self.softmax3_blocks(out_mpool3)
        #
        out_interp3 = self.interpolation3(out_softmax3)
        # print(out_skip2_connection.data)
        # print(out_interp3.data)
        out = out_interp3 + out_skip2_connection
        out_softmax4 = self.softmax4_blocks(out)
        out_interp2 = self.interpolation2(out_softmax4)
        out = out_interp2 + out_skip1_connection
        out_softmax5 = self.softmax5_blocks(out)
        out_interp1 = self.interpolation1(out_softmax5)
        out_softmax6 = self.softmax6_blocks(out_interp1)
        out = (1 + out_softmax6) * out_trunk
        out_last = self.last_blocks(out)

        return out_last
class ResidualAttentionModel(nn.Module):
    # for input size 224
    def __init__(self):
        super(ResidualAttentionModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(64, 256)
        self.attention_module1 = AttentionModule(256, 256)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule(512, 512)
        self.attention_module2_2 = AttentionModule(512, 512)  # tbq add
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule(1024, 1024)
        self.attention_module3_2 = AttentionModule(1024, 1024)  # tbq add
        self.attention_module3_3 = AttentionModule(1024, 1024)  # tbq add
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )
        self.fc = nn.Linear(2048,10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        # print(out.data)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out