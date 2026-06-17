

'''

Input a folder that contains all your ome.zarrs. 
This code reads your sparse label masks under each ome.zarr file
and create new dilated masks under each ome.zarr file.

Use it in command line by:
'python dilate_masks.py --input_dir /mnt/efs/dl_jrc/student_data/S-XX/data_train_omezarr --radius 5'

YC@260616

'''
import os
import glob
import argparse
import zarr
import numpy as np
from skimage import morphology


def dilate_masks(input_dir, dilation_steps=1):
    """
    Dilates sparse_label_masks for all .ome.zarr files in input_dir.
    Writes results to labels/sparse_label_masks_dilated/0, preserving
    the source's (1, Z, Y, X) shape, chunks, and dtype.
    """
    zarr_files = sorted(glob.glob(os.path.join(input_dir, "*ome.zarr")))
    selem = morphology.disk(dilation_steps)

    for f in zarr_files:
        src_path = os.path.join(f, "labels/sparse_label_masks/0")
        dst_path = os.path.join(f, f"labels/sparse_label_masks_dilated_{dilation_steps}/0")

        src = zarr.open(src_path, mode="r")
        src_array = src[:]
        original_shape = src_array.shape

        squeezed = np.squeeze(src_array)

        if squeezed.ndim == 2:
            dilated = morphology.dilation(squeezed, selem)
        elif squeezed.ndim == 3:
            dilated = np.empty_like(squeezed)
            for z in range(squeezed.shape[0]):
                dilated[z] = morphology.dilation(squeezed[z], selem)
        else:
            raise ValueError(
                f"Unexpected squeezed shape {squeezed.shape} for {f}"
            )

        out = dilated.reshape(original_shape).astype(src.dtype)

        dst = zarr.open(
            dst_path,
            mode="w",
            shape=src.shape,
            chunks=src.chunks,
            dtype=src.dtype,
        )
        dst[:] = out
        print(f"Done: {f}  shape={original_shape}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dilate sparse_label_masks in OME-zarr files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing *.ome.zarr files.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=1,
        help="Radius of the 2D disk structuring element (default: 1).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dilate_masks(input_dir=args.input_dir, dilation_steps=args.radius)