import torch

def loss(
        gt_affs,
        pred_affs,
        affs_weights, # from mask
):
    
    loss_weighted = MSE_loss(gt_affs, pred_affs, affs_weights)

    return loss_weighted

def MSE_loss(
        gt_affs,
        pred_affs,
        affs_weights,
): 

    pixelwise_loss = affs_weights * (pred_affs - gt_affs)**2
    count_affs = torch.count_nonzero(affs_weights)
    loss = pixelwise_loss.sum() / (count_affs+1e-8)

    return loss