from torch import Tensor, nn, sum
import torch

class weighted_MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_affs: Tensor, gt_affs: Tensor, affs_weights) -> Tensor:
        """
        Runs the forward pass.
        """

        pixelwise_loss = ((pred_affs - gt_affs)**2)*affs_weights
        count_affinities = torch.count_nonzero(affs_weights)+1e-8

        return sum(pixelwise_loss)/count_affinities


