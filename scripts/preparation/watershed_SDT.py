#!/usr/bin/env python3
"""
Convert semantic segmentation masks to instance segmentation masks using SDT watershed.

Pipeline:
  1. Load each semantic mask (tif/tiff) from --input-dir.
  2. Compute a signed distance transform (SDT): positive inside foreground,
     negative in background, and close to 0 near boundaries.
  3. Seed the watershed at local maxima of the SDT inside the foreground.
  4. Run skimage.segmentation.watershed on the negated SDT,
     restricted to the foreground.
  5. Filter tiny fragments with skimage.morphology.remove_small_objects.
  6. Save the result as a uint32 tif in --output-dir.

Usage:
    python watershed_sdt.py --output-dir /path/to/instance_masks
    python watershed_sdt.py --input-dir /custom/semantic --output-dir /out \
        --min-seed-distance 8 --min-size 100 --sdt-scale 5
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt, label, map_coordinates, maximum_filter
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

INPUT_DIR = "/mnt/efs/dl_jrc/student_data/S-HH/CellPoseSAM/Cell6/Masks"


def compute_sdt(labels: np.ndarray, scale: int = 5) -> np.ndarray:
    """Compute a signed distance transform from a semantic label image.

    Positive values are inside foreground objects, negative values are in the
    background, and values close to 0 are near object boundaries.

    Args:
        labels: Integer array where 0 = background and >0 = foreground/object.
        scale: Controls tanh normalization. Larger values preserve a broader
            distance range; smaller values saturate the SDT faster.

    Returns:
        float32 array with values approximately between -1 and 1.
    """
    dims = labels.ndim

    # Placeholder array of infinite distances.
    distances = np.ones(labels.shape, dtype=np.float32) * np.inf

    for axis in range(dims):
        # Compute boundary locations by shifting labels along this axis and
        # checking whether neighbouring pixels/voxels are the same label.
        slices_forward = tuple(
            slice(None) if a != axis else slice(1, None) for a in range(dims)
        )
        slices_backward = tuple(
            slice(None) if a != axis else slice(None, -1) for a in range(dims)
        )

        bounds = labels[slices_forward] == labels[slices_backward]

        # Pad to account for the pixel/voxel lost by shifting.
        bounds = np.pad(
            bounds,
            [(1, 1) if a == axis else (0, 0) for a in range(dims)],
            mode="constant",
            constant_values=1,
        )

        # Distance to boundary mask for this axis.
        axis_distances = distance_transform_edt(bounds)

        # Coordinates of original pixels relative to padded boundary mask.
        # This is a half-pixel shift along the axis we used for boundaries.
        coordinates = np.meshgrid(
            *[
                (
                    np.arange(axis_distances.shape[a])
                    if a != axis
                    else np.linspace(0.5, axis_distances.shape[a] - 1.5, labels.shape[a])
                )
                for a in range(dims)
            ],
            indexing="ij",
        )
        coordinates = np.stack(coordinates)

        # Interpolate distances back onto original pixel/voxel coordinates.
        sampled = map_coordinates(
            axis_distances,
            coordinates=coordinates,
            order=3,
            mode="nearest",
        )

        # Keep the minimum distance to a boundary across all axes.
        distances = np.minimum(distances, sampled.astype(np.float32))

    # Normalize distances to approximately [-1, 1].
    distances = np.tanh(distances / scale).astype(np.float32)

    # Make background negative.
    distances[labels == 0] *= -1

    return distances


def semantic_to_instance(
    semantic: np.ndarray,
    min_seed_distance: int = 10,
    min_size: int = 64,
    sdt_scale: int = 5,
) -> np.ndarray:
    """Convert a semantic mask to a unique-label instance mask using SDT watershed.

    Args:
        semantic: Integer array where 0 = background and >0 = foreground.
        min_seed_distance: Side length of the maximum-filter window used to
            find seed points. Smaller values create more seeds/splitting;
            larger values create fewer seeds/merging.
        min_size: Objects smaller than this many pixels/voxels are discarded.
        sdt_scale: Scale used by tanh normalization in compute_sdt.

    Returns:
        uint32 array with the same shape as ``semantic``.
    """
    foreground = semantic > 0

    if not np.any(foreground):
        return np.zeros(semantic.shape, dtype=np.uint32)

    # SDT is highest near object centres and lowest near boundaries/background.
    sdt = compute_sdt(semantic, scale=sdt_scale)

    # Seeds: local maxima of the SDT restricted to foreground.
    max_filtered = maximum_filter(sdt, size=min_seed_distance)
    maxima = (max_filtered == sdt) & foreground
    seeds, n_seeds = label(maxima)

    if n_seeds == 0:
        return np.zeros(semantic.shape, dtype=np.uint32)

    # Watershed floods from centres toward boundaries, so use the negative SDT.
    instance_mask = watershed(-sdt, markers=seeds, mask=foreground)

    instance_mask = remove_small_objects(
        instance_mask.astype(np.int64), min_size=min_size, connectivity=1
    )

    return instance_mask.astype(np.uint32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert semantic segmentation tifs to instance segmentation tifs using SDT watershed."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=INPUT_DIR,
        help=f"Directory containing semantic mask tif files. Default: {INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write instance mask tif files.",
    )
    parser.add_argument(
        "--min-seed-distance",
        type=int,
        default=10,
        help="Maximum-filter window size for seeding in pixels/voxels. Default: 10.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=64,
        help="Minimum object size in pixels/voxels; smaller fragments are removed. Default: 64.",
    )
    parser.add_argument(
        "--sdt-scale",
        type=int,
        default=5,
        help="Scale for tanh normalization of the SDT. Default: 5.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".tif", ".tiff")
    )
    if not tif_files:
        raise FileNotFoundError(f"No tif files found in {input_dir}")

    print(f"Found {len(tif_files)} file(s) in {input_dir}")

    for path in tif_files:
        print(f"  {path.name} ...", end=" ", flush=True)
        semantic = tifffile.imread(str(path))
        instance = semantic_to_instance(
            semantic,
            min_seed_distance=args.min_seed_distance,
            min_size=args.min_size,
            sdt_scale=args.sdt_scale,
        )
        out_path = output_dir / path.name
        tifffile.imwrite(str(out_path), instance)
        n_instances = int((np.unique(instance) != 0).sum())
        print(f"{n_instances} instances -> {out_path.name}")

    print(f"Done. Wrote {len(tif_files)} file(s) to {output_dir}")


if __name__ == "__main__":
    main()
