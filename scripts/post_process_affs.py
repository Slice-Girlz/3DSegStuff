from pathlib import Path

import mwatershed as mws
import numpy as np
import zarr

from ome_zarr.writer import write_image
from skimage.morphology import remove_small_objects

# ====== Load an OME-Zarr file ======
pred_affs = "/mnt/efs/dl_jrc/student_data/S-JM/train/processed_zarr/2026-06-17_02-29-36/snapshots/batch_9001.zarr/gt_affs"
affs_array = zarr.open(pred_affs)[0] # Check if you have channel dim

# ====== Generate Instance Segmentations ====== 
# Set of offsets
neighborhood = np.array([
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
], dtype=np.int64)

# Set mutex watershed biases
bias_short = -0.8 # Originally at -0.9

# Generate instance segmentation from affinities
biased_affs = np.array(
        [
            affs_array[0] + bias_short,
            affs_array[1] + bias_short,
            affs_array[2] + bias_short,
        ]
    ).astype(np.float64)

pred_labels = mws.agglom(biased_affs, neighborhood)

# Filter out small objects
pred_labels = remove_small_objects(
    pred_labels.astype(np.uint64), min_size=50, connectivity=3
)

# ====== Save instance segmentations into the OME-Zarr file ======
root = zarr.open_group(str(Path(pred_affs).parent), mode="r+")
pred_labels_group = root.require_group("pred_labels")

write_image(
    image=pred_labels,
    group=pred_labels_group,
    axes=["z", "y", "x"], # Check if you have channel dim
    scaler=None,
)