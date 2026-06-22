import neuroglancer
import numpy as np
import zarr

base = "/mnt/efs/dl_jrc/student_data/S-MS/raw_data_omezarr/AR177_section2_1x1.ome.zarr"

raw = zarr.open(f"{base}/0")[:]                  # (c=2, z, y, x) float32
pred_affs = zarr.open(f"{base}/pred_affs_3d_long")[:]
labels = zarr.open(f"{base}/masks_cellpose/0")[:]  # (z, y, x) uint64

print("raw", raw.shape, raw.dtype)
print("labels", labels.shape, labels.dtype, "n_ids", len(np.unique(labels)))

neuroglancer.set_server_bind_address("127.0.0.1")  # localhost; tunnel the port if remote
viewer = neuroglancer.Viewer()

# Shared spatial coordinate space. Both layers live in z, y, x so they overlay.
# Voxel size (z, y, x) in nm from the acquisition metadata (0.9, 0.26, 0.26 um).
zyx = neuroglancer.CoordinateSpace(
    names=["z", "y", "x"], units="nm", scales=[900, 260, 260]
)

# Raw keeps its channel as a *local* dimension (c^) so it is NOT a shared axis.
raw_dims = neuroglancer.CoordinateSpace(
    names=["c^", "z", "y", "x"], units=["", "nm", "nm", "nm"], scales=[1, 900, 260, 260]
)

with viewer.txn() as s:
    s.layers["raw"] = neuroglancer.ImageLayer(
        source=neuroglancer.LocalVolume(data=raw, dimensions=raw_dims),
        shader="""
        void main() {
            emitRGB(vec3(
                0.0,
                toNormalized(getDataValue(1)),
                toNormalized(getDataValue(0))
                ));
        }
        """,
    )
    s.layers["pred_affs"] = neuroglancer.ImageLayer(
        source=neuroglancer.LocalVolume(data=pred_affs, dimensions=raw_dims))
    s.layers["pred_labels_3d"] = neuroglancer.SegmentationLayer(
        source=neuroglancer.LocalVolume(data=labels.astype(np.uint64), dimensions=zyx),
    )

print(viewer)
input("Press Enter to exit...")