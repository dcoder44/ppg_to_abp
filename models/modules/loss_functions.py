import torch.nn as nn
import torch

mae_loss = nn.L1Loss()
mse_loss = nn.MSELoss()
huber_loss_function = nn.SmoothL1Loss(beta=4.0)

def mean_error(estimated_abp, true_abp):
    return torch.mean(estimated_abp - true_abp, dim=1)

def mean_absolute_error(estimated_abp, true_abp):
    return torch.mean(torch.abs(estimated_abp - true_abp), dim=1)

def huber_loss(estimated_abp, true_abp):
    sbp_loss = huber_loss_function(estimated_abp[0], true_abp[0])
    dbp_loss = huber_loss_function(estimated_abp[1], true_abp[1])
    return sbp_loss + dbp_loss
