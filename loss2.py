import torch
import torch.nn as nn
import torch.nn.modules.loss as loss
from log import log_warning
from scipy.signal import welch, periodogram, resample
from utils.funcs import plot_graph, plot_loss_graph, BPF_signal, normalize
import cusignal
import cupy
import numpy as np
def loss_fn(loss_fn):
    if loss_fn == "mse":
        return loss.MSELoss(reduction='mean')
    elif loss_fn == "L1":
        return loss.L1Loss()
    elif loss_fn == "neg_pearson":
        return NegPearsonLoss_MTTS()
    elif loss_fn == "combined_loss":
        return Combined_Loss()
    elif loss_fn == "snr":
        return NewCombinedLoss()
    else:
        log_warning("use implemented loss functions")
        raise NotImplementedError("implement a custom function(%s) in loss.py" % loss_fn)


def neg_Pearson_Loss_MTTS(predictions, targets):
    # Pearson correlation can be performed on the premise of normalization of input data
    predictions = torch.squeeze(predictions)
    targets = torch.squeeze(targets)
    if len(predictions.shape) >= 2:
        predictions = predictions.view(-1)
    if len(targets.shape) >= 2:
        targets = targets.view(-1)
    sum_x = torch.sum(predictions)  # x
    sum_y = torch.sum(targets)  # y
    sum_xy = torch.sum(predictions * targets)  # xy
    sum_x2 = torch.sum(torch.pow(predictions, 2))  # x^2
    sum_y2 = torch.sum(torch.pow(targets, 2))  # y^2
    t = len(predictions)
    pearson = (t * sum_xy - (sum_x * sum_y)) / (torch.sqrt((t * sum_x2 - torch.pow(sum_x, 2)) * (t * sum_y2 - torch.pow(sum_y, 2))))

    return 1 - pearson

def Welch(bvps, fps, minHz=0.65, maxHz=4.0, nfft=2048):
    """
    This function computes Welch'method for spectral density estimation.

    Args:
        bvps(flaot32 numpy.ndarray): BVP signal as float32 Numpy.ndarray with shape [num_estimators, num_frames].
        fps (float): frames per seconds.
        minHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        maxHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        nfft (int): number of DFT points, specified as a positive integer.
    Returns:
        Sample frequencies as float32 numpy.ndarray, and Power spectral density or power spectrum as float32 numpy.ndarray.
    """
    n = bvps.shape[0]
    if n < 256:
        seglength = n
        overlap = int(0.8*n)  # fixed overlapping
    else:
        seglength = 256
        overlap = 200
    # -- periodogram by Welch
    F, P = welch(bvps, nperseg=seglength, noverlap=overlap, fs=fps, nfft=nfft)
    F = F.astype(np.float32)
    P = P.astype(np.float32)
    # -- freq subband (0.65 Hz - 4.0 Hz)
    band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
    Pfreqs = F[band] # 60*F[band]
    Power = P[band]
    return Pfreqs, Power



def calculate_PSD(pred, f_range, fs):
    T = pred.shape[0] # window length
    # print("T = ", T)
    
    psd = torch.zeros_like(f_range)
    for i, f in enumerate(f_range):
        cos_vals = torch.cos(2 * torch.pi * f * torch.arange(T).cuda() / fs)
        sin_vals = torch.sin(2 * torch.pi * f * torch.arange(T).cuda() / fs)

        psd[i] = 1/T *(torch.sum(pred * cos_vals) ** 2 + \
                 torch.sum(pred * sin_vals) ** 2)
    # print("Shape before: ", psd.shape)
    psd = torch.tensor(resample(psd.cpu().detach().numpy(),64))
    # print("Shape after: ", psd.shape)
    return psd 

#MMSE: 0.4 - 2.67, UBFC: 0.4 - 4, PURE: 0.5 - 2.67
 # MANHOB_HCI: 0.67 - 2
 # also snr but implemneted by Anh from scratch based on the formula in the paper
def snr_loss(pred, target, fs):
    B, T = pred.shape # number of frames, 32x10 batch_size, window_size
    f_range = torch.linspace(0.67, 4, T).cuda()
    snr = torch.zeros(B).cuda()

    NyquistF = fs/2.;
    FResBPM = 0.5
    nfft = np.ceil((60*2*NyquistF)/FResBPM)   
    for i,ti in enumerate(target):
        # calculate HR ground truth
        gt = BPF_signal(normalize(ti.cpu().detach().numpy()), fs, 0.67, 4)
        # gt_fft = np.square(np.abs(np.fft.rfft(gt))) # assume 10 # use direct fourier transform
        freqs, psd = Welch(gt, fs, minHz=0.4, maxHz=2.67, nfft=nfft) # use welch algorithm
         # range of HR frequencies to consider
        
        max_power_idx = np.argmax(psd)
        fT = torch.tensor([freqs[max_power_idx]]).cuda()

        # compute power spectral density
        psd_all = calculate_PSD(pred[i], f_range, fs)
        psd_all_sum = torch.sum(psd_all)

        psd_fT = calculate_PSD(pred[i], fT, fs)
        psd_fT_sum = torch.sum(psd_fT)
        psd_other_sum = psd_all_sum - psd_fT_sum

        # snr[i] = 10*torch.log10(psd_fT_sum/(psd_other_sum))
        snr[i] = (psd_fT_sum - torch.log(torch.sum(torch.exp(psd_all))))  #cross entropy
        # snr[i] = cross_entropy(psd_fT_sum, psd_all)
    
    
    loss = torch.mean(snr) 
    # loss = torch.mean(psd_fT_gt) - torch.log(torch.sum(torch.exp(psd_all)))
    return - loss # return negative SNR as a loss to minimize
def Welch_cuda(bvps, fps, minHz=0.67, maxHz=4.0, nfft=2048):
    # print("Signal to Noise Loss Function!")
    """
    This function computes Welch'method for spectral density estimation on CUDA GPU.

    Args:
        bvps(float32 cupy.ndarray): BVP signal as float32 Numpy.ndarray with shape [num_estimators, num_frames].
        fps (cupy.float32): frames per seconds.
        minHz (cupy.float32): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        maxHz (cupy.float32): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        nfft (cupy.int32): number of DFT points, specified as a positive integer.
    Returns:
        Sample frequencies as float32 cupy.ndarray, and Power spectral density or power spectrum as float32 cupy.ndarray.
    """
    n = bvps.shape[0]
    if n < 256:
        seglength = n
        overlap = int(0.8*n)  # fixed overlapping
    else:
        seglength = 256
        overlap = 200  
    # -- periodogram by Welch
    F, P = cusignal.welch(bvps, nperseg=seglength,
                            noverlap=overlap, fs=fps, nfft=nfft)
    # print(F.shape, P.shape)
    # -- freq subband (0.65 Hz - 4.0 Hz)
    band = cupy.argwhere((F > minHz) & (F < maxHz)).flatten()
    Pfreqs = F[band]
    Power = P[band]
    return Pfreqs, Power

#  SNR loss function  -  Anh modified from other student
def get_SNR(predictions, targets, fps):
    predictions = torch.squeeze(predictions)
    targets = torch.squeeze(targets)
    '''Computes the signal-to-noise ratio of the BVP
    signals according to the method by -- de Haan G. et al., IEEE Transactions on Biomedical Engineering (2013).
    SNR calculated as the ratio (in dB) of power contained within +/- 0.1 Hz
    of the reference heart rate frequency and +/- 0.2 of its first
    harmonic and sum of all other power between 0.5 and 4 Hz.
    Adapted from https://github.com/danmcduff/iphys-toolbox/blob/master/tools/bvpsnr.m
    '''
    # upsampling  -  Anh
    predictions = predictions.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    # print("Shape before upsampling: ", predictions.shape, targets.shape)
    # up sampling the output signal
    # predictions = resample(predictions.cpu().detach().numpy(),256 , axis = 1)
    # targets = resample(targets.cpu().detach().numpy(), 256, axis = 1)
    # print("Shape after upsampling: ", predictions.shape, targets.shape)
    # generate groundtruth frequency
    interv1 = 0.1
    interv2 = 0.2
    if type(fps) != int:
        NyquistF = int(fps[0])/2.;
    else:
        NyquistF = fps/2.;
    FResBPM = 0.5
    nfft = np.ceil((60*2*NyquistF)/FResBPM)
    SNR = 0
    # fT = torch.zeros(len(targets))
    fT = np.zeros(len(targets))
    # get the ground truth frequency  - Anh
    for i,ti in enumerate(targets):
        # print(ti)
        # gt = BPF_signal(ti.cpu().detach().numpy(), fps, 0.67, 4)
        # gt = ti.cpu().detach().numpy()
        gt = ti
        # use welch algorithm
        if type(fps) != int:
            freqs, psd = Welch(gt, int(fps[0]), minHz=0.5, maxHz=2.67, nfft=nfft)
        else:
            freqs, psd = Welch(gt, fps, minHz=0.5, maxHz=2.67, nfft=nfft)
         # range of HR frequencies to consider
        max_power_idx = np.argmax(psd)
        fT[i] = freqs[max_power_idx]   
        
    for idx, bvp in enumerate(predictions): #MMSE: 0.4 - 2.67, UBFC: 0.4 - 4, PURE: 0.5 - 2.67
        curr_ref = fT[idx]                  # MANHOB_HCI: 0.67 - 2
        if type(fps) != int:
            pfreqs, power = Welch_cuda(bvp, int(fps[0]), minHz=0.5, maxHz=2.67, nfft=nfft)
        else:
            pfreqs, power = Welch_cuda(bvp, fps, minHz=0.5, maxHz=2.67, nfft=nfft)
        pfreqs = cupy.asnumpy(pfreqs)
        power = cupy.asnumpy(power)
        # print("frequency: ", pfreqs.shape)
        # print("power: ", power.shape)

        # upsampling the power spectral
        # pfreqs = resample(pfreqs,256)
        # power = resample(power, 256)

        GTMask1 = np.logical_and(pfreqs>=curr_ref-interv1, pfreqs<=curr_ref+interv1)
        GTMask2 = np.logical_and(pfreqs>=(curr_ref*2)-interv2, pfreqs<=(curr_ref*2)+interv2)
        GTMask = np.logical_or(GTMask1, GTMask2)
        FMask = np.logical_not(GTMask)

        p = power
        SPower = torch.tensor(np.sum(p[GTMask]))
        allPower = torch.tensor(np.sum(p[FMask]))
        
        if allPower == 0:
            print("Zero here: ", idx, curr_ref, p[FMask])
        snr = torch.log10(SPower/allPower)
        # print(snr)
        if snr == 0: print(snr)

        SNR += snr

    return - SNR/len(predictions)

#

class SNRLoss(nn.Module):
    def __init__(self):
        super(SNRLoss, self).__init__()

    def forward(self, predictions, targets, fps):
        # return snr_loss(predictions, targets, fps)
        return get_SNR(predictions, targets, fps)

class NegPearsonLoss_MTTS(nn.Module):
    def __init__(self):
        super(NegPearsonLoss_MTTS, self).__init__()

    def forward(self, predictions, targets):
        return neg_Pearson_Loss_MTTS(predictions, targets)


class Combined_Loss(nn.Module):
    def __init__(self):
        super(Combined_Loss, self).__init__()
        self.mse = loss.MSELoss(reduction='mean')
        self.pearson = NegPearsonLoss_MTTS()

    def forward(self, predictions, targets):
        mse = self.mse(predictions, targets)
        pearson = self.pearson(predictions, targets)
        total = mse + pearson     # change this
        # total = mse
        return total
class NewCombinedLoss(nn.Module):
    def __init__(self):
        super(NewCombinedLoss, self).__init__()
        self.mse = loss.MSELoss(reduction='mean')
        self.snr = SNRLoss()
        self.pearson = NegPearsonLoss_MTTS()

    # combined loss function about snr, mse, pearson  -  Anh
    def forward(self, predictions, targets, fps):
        mse = self.mse(predictions, targets)
        pearson = self.pearson(predictions, targets)
        snr = self.snr(predictions, targets, fps)
        # print(f"snr = {snr}, mse = {mse}, pearson: {pearson}")
        total =mse + snr + pearson
        # print(f"snr = {snr}, mse = {mse}")
        # total =0.1*mse + 0.9*snr
        # total = snr
        return total
        # return get_snr(predictions, targets, fps)