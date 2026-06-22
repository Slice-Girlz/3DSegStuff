from pathlib import Path

import mwatershed as mws
import numpy as np
import zarr
from ome_zarr.writer import write_labels
from ome_zarr.writer import write_image
from skimage.morphology import remove_small_objects
from skimage.measure import label


# ====== Load an OME-Zarr file ======
pred_affs = "/mnt/efs/dl_jrc/student_data/S-MS/raw_data_omezarr/AR177_section2_1x1.ome.zarr/pred_affs_3d_long"
affs_array = zarr.open(pred_affs)

# ====== Generate Instance Segmentations ====== 
# Set of offsets
#neighborhood = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0], [0, 9, 0], [0, 0, 9]], dtype=np.int64)
neighborhood = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0], [0, 9, 0], [0, 0, 9]]

# Set mutex watershed biases
bias_short = -0.9
bias_long = -0.95

print(affs_array.shape)
# Generate instance segmentation from affinities
biased_affs = np.array(
        [
            affs_array[0] + bias_short,
            affs_array[1] + bias_short,
            affs_array[2] + bias_short, 
            affs_array[3] + bias_long,
            affs_array[4] + bias_long,
            affs_array[5] + bias_long
            ]
    ).astype(np.float64)

pred_labels = mws.agglom(biased_affs, neighborhood)
pred_labels = label(pred_labels, connectivity=1)
print(pred_labels.shape)

pred_labels = remove_small_objects(pred_labels.astype(np.int64), min_size=25, connectivity=1).astype(np.uint32)

# Save instance segmentations into the OME-Zarr file
root = zarr.open_group(str(Path(pred_affs).parent), mode="r+")
pred_labels_group = root.require_group("pred_labels_3d_long")

write_image(
    image=pred_labels,
    group=pred_labels_group,
    axes=["y", "x"],
    scaler=None,
)