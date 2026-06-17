import os, glob
import zarr
import numpy as np
from skimage.morphology import dilation, disk

def dilate_sparse_masks(input_dir, radius=6):
    footprint_2d = disk(radius)
    dst_label_name = f"sparse_label_masks_dilated_r{radius}"

    for f in sorted(glob.glob(os.path.join(input_dir, "*ome.zarr"))):
        src_path = os.path.join(f, "labels/sparse_label_masks/0")
        dst_path = os.path.join(f, f"labels/{dst_label_name}/0")

        src = zarr.open(src_path, mode="r")
        mask = src[:]

        print("mask shape:", mask.shape)

        # make footprint match mask ndim
        # only dilate last two dims: Y, X
        footprint = footprint_2d.reshape((1,) * (mask.ndim - 2) + footprint_2d.shape)

        dilated = dilation(mask > 0, footprint=footprint).astype(mask.dtype)

        dst = zarr.open(
            dst_path,
            mode="w",
            shape=dilated.shape,
            dtype=mask.dtype,
            chunks=src.chunks,
        )
        dst[:] = dilated

        print("wrote:", dst_path)

dilate_sparse_masks('/mnt/efs/dl_jrc/student_data/S-XZ/train_janelia/omezarr_split/train/', radius=6)
