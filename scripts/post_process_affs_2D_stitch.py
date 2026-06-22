import mwatershed as mws
import numpy as np
import zarr
from ome_zarr.writer import write_image


# ====== Config ======
container = "/mnt/efs/dl_jrc/student_data/S-MS/annotations_omezarr/test/AR177_section4_3x2__A3.ome_crop02_z0000-0028_y0075-0203_x1511-1639_img.ome.zarr/0"
affs_path = f"{container}/pred_affs_short"   # array shape (c=2, z, y, x)
out_name = "pred_labels"

bias_short = -0.9        # mutex watershed bias added to the affinities
bias_long = -0.95
iou_threshold = 0.25      # min IoU to link a 2D object across consecutive z-slices
min_size = 10           # remove 3D objects smaller than this many voxels (0 = keep all)
voxel_size = (900, 260, 260)  # (z, y, x) in nm, written to the label metadata for viewing

# 2D neighborhood: short-range affinities along y and x
neighborhood = np.array([
    [1, 0],
    [0, 1]
], dtype=np.int64)


# ====== 1. Per-slice 2D instance segmentation ======
affs = zarr.open(affs_path)            # (2, z, y, x)
n_channels, nz, ny, nx = affs.shape
assert n_channels == len(neighborhood), (
    f"affs has {n_channels} channels but neighborhood has {len(neighborhood)} offsets"
)

seg = np.zeros((nz, ny, nx), dtype=np.uint64)
max_id = 0
for z in range(nz):
    biased_affs = np.stack([
        affs[0, z] + bias_short,
        affs[1, z] + bias_short,
    ]).astype(np.float64)

    labels = mws.agglom(biased_affs, neighborhood).astype(np.uint64)

    # Offset this slice's ids so they are globally unique (keep 0 as background).
    fg = labels != 0
    labels[fg] += max_id
    seg[z] = labels
    if fg.any():
        max_id = int(labels.max())
    print(f"z={z}: {len(np.unique(labels[fg]))} objects")


# ====== 2. Link objects across z by IoU (union-find) ======
parent = {}

def find(x):
    parent.setdefault(x, x)
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:        # path compression
        parent[x], x = root, parent[x]
    return root

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

for z in range(nz - 1):
    a, b = seg[z], seg[z + 1]
    both = (a != 0) & (b != 0)
    if not both.any():
        continue

    av, bv = a[both], b[both]
    # Encode (a, b) pairs into a single int for fast overlap counting.
    key = av.astype(np.int64) * (max_id + 1) + bv.astype(np.int64)
    uniq_key, inter = np.unique(key, return_counts=True)
    pair_a = (uniq_key // (max_id + 1)).astype(np.uint64)
    pair_b = (uniq_key % (max_id + 1)).astype(np.uint64)

    ids_a, areas_a = np.unique(a[a != 0], return_counts=True)
    ids_b, areas_b = np.unique(b[b != 0], return_counts=True)
    area_a = dict(zip(ids_a.tolist(), areas_a.tolist()))
    area_b = dict(zip(ids_b.tolist(), areas_b.tolist()))

    for la, lb, i in zip(pair_a.tolist(), pair_b.tolist(), inter.tolist()):
        iou = i / (area_a[la] + area_b[lb] - i)
        if iou >= iou_threshold:
            union(la, lb)


# ====== 3. Relabel to consecutive global ids ======
unique_ids = np.unique(seg)
unique_ids = unique_ids[unique_ids != 0]

root_to_new = {}
lut = np.zeros(max_id + 1, dtype=np.uint64)
next_id = 1
for uid in unique_ids.tolist():
    r = find(uid)
    if r not in root_to_new:
        root_to_new[r] = next_id
        next_id += 1
    lut[uid] = root_to_new[r]

seg = lut[seg]
print(f"3D objects after stitching: {next_id - 1}")


# ====== 4. Remove small objects (by total 3D voxel count) ======
if min_size > 0:
    ids, sizes = np.unique(seg, return_counts=True)
    keep = (ids != 0) & (sizes >= min_size)

    # Build a LUT that drops small ids and relabels the rest consecutively.
    relabel = np.zeros(int(ids.max()) + 1, dtype=np.uint64)
    relabel[ids[keep]] = np.arange(1, keep.sum() + 1, dtype=np.uint64)
    seg = relabel[seg]
    print(f"Removed {(~keep & (ids != 0)).sum()} objects smaller than {min_size} voxels")

n_final = int(seg.max())
print(f"Final 3D objects: {n_final}")


# ====== 5. Write out as a 3D OME-Zarr label volume ======
root = zarr.open_group(container, mode="r+")
if out_name in root:           # overwrite a previous run
    del root[out_name]
out_group = root.require_group(out_name)
write_image(
    image=seg,
    group=out_group,
    axes=["z", "y", "x"],
    scaler=None,
)

# funlib's neuroglancer CLI reads axes from these array attrs (not OME metadata).
# Mirror the raw's metadata so the labels share its z,y,x coordinate space.
out_group["0"].attrs.update({
    "axis_names": ["z", "y", "x"],
    "offset": [0, 0, 0],
    "resolution": list(voxel_size),
    "types": ["space", "space", "space"],
    "units": ["nm", "nm", "nm"],
    "voxel_size": list(voxel_size),
})
print(f"Wrote {out_name} with shape {seg.shape}")