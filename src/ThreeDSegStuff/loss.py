
import torch.nn as nn

def loss(
        gt_affs,
        pred_affs,
        affs_weights, # from mask
):
    loss = nn.MSELoss()
    loss_MSE = loss(pred_affs, gt_affs)
    loss_weighted = loss_MSE * affs_weights
    return loss_weighted