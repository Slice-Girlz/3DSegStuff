from pathlib import Path

import mwatershed as mws
import numpy as np
import zarr
from ome_zarr.writer import write_labels
from ome_zarr.writer import write_image


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
print(pred_labels.shape)

# Save instance segmentations into the OME-Zarr file
root = zarr.open_group(str(Path(pred_affs).parent), mode="r+")
pred_labels_group = root.require_group("pred_labels")

write_image(
    image=pred_labels,
    group=pred_labels_group,
    axes=["z", "y", "x"],
    scaler=None,
)