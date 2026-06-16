# IMPORTS 
# import numpy as np
# import torch

# import gunpowder as gp

# from torch.utils.data import DataLoader
# from torchvision.transforms import v2 as transforms_v2
# from skimage.morphology import remove_small_objects

import mwatershed as mws
import numpy as np
import zarr
from blah.blah import write_labels

# ====== Load an OME-Zarr file ======
pred_affs = "/mnt/efs/dl_jrc/student_data/S-JM/train/processed_zarr/2026-06-16_20-31-33/snapshots/batch_1.zarr/pred_affs"
affs_array = zarr.open(pred_affs)[0]

# ====== Generate Instance Segmentations ====== 
# Set of offsets
neighborhood = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
], dtype=np.int64)

# Set mutex watershed biases
bias_short = -0.9

# Generate instance segmentation from affinities
biased_affs = np.array(
        [
            affs_array[0] + bias_short,
            affs_array[1] + bias_short,
            affs_array[2] + bias_short,
        ]
    ).astype(np.float64)

pred_labels = mws.agglom(biased_affs, neighborhood)

# Save instance segmentations into the OME-Zarr file
