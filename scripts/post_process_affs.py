from pathlib import Path

import mwatershed as mws
import numpy as np
import zarr
from ome_zarr.writer import write_image
from scipy import ndimage as ndi
from ThreeDSegStuff.data.metadata import prepare_metadata


# ============================================================
# Paths
# ============================================================
sample_path = Path(
    "/mnt/efs/dl_jrc/student_data/S-XZ/train_janelia/omezarr_split/val/"
    "norm_1.ome.zarr"
)

pred_affs_path = sample_path / "pred_affs_r10"
out_name = "pred_labels_r10_2"


# ============================================================
# Parameters
# ============================================================

# Start with 0.30 because this was the setting you already tested.
# If still too fragmented after evaluation, try 0.40 or 0.50.
edge_threshold = 0.8

# Raw image foreground threshold.
# 98.0 gives about top 2% brightest voxels.
# Your previous output gave ~478k foreground voxels, reasonable to start.
fg_percentile = 96

# Cell diameter ~10 px.
# Compact sphere with radius 5 px is ~523 voxels,
# but because watershed is fragmented, use a low min_size first.
min_size = 2

# Remove very large merged blobs.
max_size = 10000

# GT has 196 cells.
# Keep slightly more than 196 so we do not delete true cells too early.
target_num_objects = 230


# ============================================================
# Helper functions
# ============================================================
def open_array(path):
    """
    Open either a direct zarr array or an OME-Zarr image group with /0.
    """
    obj = zarr.open(str(path), mode="r")

    if isinstance(obj, zarr.hierarchy.Group):
        return zarr.open(str(path / "0"), mode="r")

    return obj


def filter_and_keep_largest(
    labels,
    min_size=10,
    max_size=10000,
    target_num_objects=230,
):
    """
    Fast filtering for sparse mwatershed labels.

    Important:
    Do NOT use np.bincount directly here, because mwatershed labels can be
    sparse and have very large label IDs.

    Steps:
    1. Count only labels that actually exist.
    2. Remove tiny fragments.
    3. Remove huge merged blobs.
    4. Keep largest target_num_objects.
    5. Relabel consecutively.
    """

    labels = labels.astype(np.uint32, copy=False)

    unique_ids, counts = np.unique(labels, return_counts=True)

    # Remove background from counting
    nonzero = unique_ids != 0
    unique_ids = unique_ids[nonzero]
    counts = counts[nonzero]

    print("real objects before size filtering:", len(unique_ids))

    if len(unique_ids) == 0:
        print("No nonzero labels found.")
        return np.zeros_like(labels, dtype=np.uint32)

    # Size filtering
    valid = (counts >= min_size) & (counts <= max_size)

    valid_ids = unique_ids[valid]
    valid_counts = counts[valid]

    print("real objects after size filtering:", len(valid_ids))

    if len(valid_ids) == 0:
        print("Warning: no valid objects after size filtering.")
        return np.zeros_like(labels, dtype=np.uint32)

    # Keep largest objects
    if len(valid_ids) > target_num_objects:
        order = np.argsort(valid_counts)[::-1]
        keep_ids = valid_ids[order[:target_num_objects]]
        keep_sizes = valid_counts[order[:target_num_objects]]
    else:
        keep_ids = valid_ids
        keep_sizes = valid_counts

    print("objects kept:", len(keep_ids))
    print("kept object size range:", int(keep_sizes.min()), int(keep_sizes.max()))

    # Fast relabeling.
    # Use lookup table if label IDs are not too huge.
    max_id = int(labels.max())

    if max_id < 100_000_000:
        lut = np.zeros(max_id + 1, dtype=np.uint32)
        lut[keep_ids] = np.arange(1, len(keep_ids) + 1, dtype=np.uint32)
        out = lut[labels]
    else:
        # Fallback if label IDs are extremely large.
        out = np.zeros_like(labels, dtype=np.uint32)
        for new_id, old_id in enumerate(keep_ids, start=1):
            out[labels == old_id] = new_id

    return out.astype(np.uint32)


# ============================================================
# Load predicted affinities
# ============================================================
affs_zarr = open_array(pred_affs_path)
print("affinity shape:", affs_zarr.shape)

if affs_zarr.ndim != 4 or affs_zarr.shape[0] != 3:
    raise ValueError(
        "Expected pred_affs shape (3, z, y, x), got "
        f"{affs_zarr.shape}."
    )

affs = affs_zarr[:].astype(np.float32)
affs = np.clip(affs, 0.0, 1.0)


# ============================================================
# Same neighborhood as training
# ============================================================
neighborhood = np.array(
    [
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ],
    dtype=np.int64,
)


# ============================================================
# Run mutex watershed
# ============================================================
biased_affs = affs - edge_threshold
biased_affs = biased_affs.astype(np.float64)

print("biased affs range:", biased_affs.min(), biased_affs.max())

pred_labels = mws.agglom(biased_affs, neighborhood)

print("raw labels:", len(np.unique(pred_labels)) - 1)

pred_labels = pred_labels.astype(np.uint32)


# ============================================================
# Build foreground mask from raw image
# ============================================================
raw_zarr = open_array(sample_path)
raw = raw_zarr[:]

# Usually raw image is either (c, z, y, x) or (z, y, x)
if raw.ndim == 4:
    raw = raw[0]
elif raw.ndim != 3:
    raise ValueError(f"Expected raw image shape (z,y,x) or (c,z,y,x), got {raw.shape}")

raw = raw.astype(np.float32)

# Smooth to make foreground mask cleaner
raw_smooth = ndi.gaussian_filter(raw, sigma=1)

threshold = np.percentile(raw_smooth, fg_percentile)
fg = raw_smooth > threshold

# Gentle XY cleanup.
# Use (1,3,3) so we do not aggressively erode through z.
fg = ndi.binary_opening(fg, structure=np.ones((1, 3, 3)))
fg = ndi.binary_closing(fg, structure=np.ones((1, 3, 3)))

print("foreground threshold:", float(threshold))
print("foreground voxels:", int(fg.sum()))


# ============================================================
# Apply foreground mask
# ============================================================
pred_labels[~fg] = 0

labels_after_fg = len(np.unique(pred_labels)) - 1
print("labels after fg mask:", labels_after_fg)


# ============================================================
# Filter fragments and keep largest candidate cells
# ============================================================
pred_labels = filter_and_keep_largest(
    pred_labels,
    min_size=min_size,
    max_size=max_size,
    target_num_objects=target_num_objects,
)

print("final labels:", len(np.unique(pred_labels)) - 1)
print("final shape:", pred_labels.shape)


# ============================================================
# Save to OME-Zarr
# ============================================================
voxel_size_zyx_nm = [1900, 540, 540]  # Z, Y, X in nm

root = zarr.open_group(str(sample_path), mode="r+")

# Remove old output if it already exists
if out_name in root:
    del root[out_name]

# Make prediction labels match raw image dimensionality: (C, Z, Y, X)
if pred_labels.ndim == 3:
    pred_labels_to_save = pred_labels[np.newaxis, ...]  # (Z,Y,X) -> (1,Z,Y,X)
elif pred_labels.ndim == 4:
    pred_labels_to_save = pred_labels                 # already (C,Z,Y,X)
else:
    raise ValueError(
        f"Expected pred_labels to be 3D (Z,Y,X) or 4D (C,Z,Y,X), "
        f"got shape {pred_labels.shape}"
    )

# Create output group, e.g. root["pred_labels_r10_1"]
pred_labels_group = root.create_group(out_name)

write_image(
    image=pred_labels_to_save,
    group=pred_labels_group,
    axes=["c", "z", "y", "x"],  # OME-Zarr axis names
    scaler=None,
)

# Stamp funlib/Gunpowder-style metadata onto level 0.
# Important:
# - axis_names has 4 entries because the array is (C,Z,Y,X)
# - c^ marks channel as non-spatial
# - voxel_size/resolution/offset/units have only 3 entries for Z,Y,X
pred_labels_group["0"].attrs.update(
    {
        "axis_names": ["c^", "z", "y", "x"],
        "types": ["channel", "space", "space", "space"],
        "offset": [0, 0, 0],
        "resolution": voxel_size_zyx_nm,
        "voxel_size": voxel_size_zyx_nm,
        "units": ["nm", "nm", "nm"],
    }
)

print("saved to:", sample_path / out_name / "0")
print("shape:", pred_labels_group["0"].shape)
print("metadata:", dict(pred_labels_group["0"].attrs))