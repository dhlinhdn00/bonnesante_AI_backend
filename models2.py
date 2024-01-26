# import torchinfo
# import torchsummary

from log import log_warning, log_info
# from nets.models.DeepPhys import DeepPhys
from nets.models.MTTS import MTTS
# from nets.models.STM_Phys import STM_Phys
from nets.models.MTTS_CSTM import MTTS_CSTM
from nets.models.New import New
from nets.models.SlowFast_FD import SlowFast_FD
from nets.models.SlowFast_AM import SlowFast_AM


def get_model(model_name, pop_mean, pop_std, frame_depth, skip, shift_factor, group_on):
    if skip == 1:
        skip = True
    else:
        skip = False
    if model_name == "MTTS":
        return MTTS(pop_mean, pop_std, False, frame_depth, shift_factor, group_on=group_on)
    elif model_name == "New":
        return New(frame_depth, pop_mean, pop_std, eca=False)
    # TS_CST  -  Hao
    elif model_name == "MTTS_CSTM":
        eca = False
        return MTTS_CSTM(frame_depth, pop_mean, pop_std, eca, shift_factor, group_on=group_on)
    elif model_name == "TSDAN":
        return MTTS(pop_mean, pop_std, True, frame_depth, shift_factor, group_on=group_on)
    # elif model_name == 'STM_Phys':
    #     return STM_Phys(32, pop_mean, pop_std)
    # SlowFast_CST  -  Anh
    elif model_name == "SlowFast_FD":
        eca = False
        return SlowFast_FD(frame_depth, pop_mean, pop_std, eca, shift_factor, group_on)
    # elif model_name == "DeepPhys":
    #     return DeepPhys()
    elif model_name == "SlowFast_AM": # attempt to add the Residual Attention module - Anh
        eca = False
        return SlowFast_AM(frame_depth, pop_mean, pop_std, eca, shift_factor, group_on)
    else:
        log_warning("use implemented model")
        raise NotImplementedError("implement a custom model(%s) in /nets/models/" % model_name)


def is_model_support(model_name, model_list):
    """
    :param model_name: model name
    :param model_list: implemented model list
    :return: model
    """
    if not (model_name in model_list):
        log_warning("use implemented model")
        raise NotImplementedError("implement a custom model(%s) in /nets/models/" % model_name)


# def summary(model, model_name):
#     """
#     :param model: torch.nn.module class
#     :param model_name: implemented model name
#     :return: model
#     """
#     log_info("=========================================")
#     log_info(model_name)
#     log_info("=========================================")
#     if model_name == "DeepPhys" or model_name == "DeepPhys_DA":
#         torchsummary.summary(model, (2, 3, 36, 36))
#     elif model_name == "PhysNet" or model_name == "PhysNet_LSTM":
#         torchinfo.summary(model, (1, 3, 32, 128, 128))
#     elif model_name in "PPNet":
#         torchinfo.summary(model, (1, 1, 250))
#     else:
#         log_warning("use implemented model")
#         raise NotImplementedError("implement a custom model(%s) in /nets/models/" % model_name)