#!/usr/bin/env python3
"""
Convert semantic segmentation masks to instance segmentation masks.

Pipeline:
  1. Load each semantic mask (tif/tiff) from --input-dir.
  2. Compute short-range affinities: pixels inside an object have affinity 1
     with all their neighbours; boundary pixels have lower mean affinity.
  3. Seed the watershed at the local maxima of the mean affinity map.
  4. Run skimage.segmentation.watershed on the negated affinity map,
     restricted to the foreground.
  5. Filter tiny fragments with skimage.morphology.remove_small_objects.
  6. Save the result as a uint32 tif in --output-dir.

Usage:
    python watershed.py --output-dir /path/to/instance_masks
    python watershed.py --input-dir /custom/semantic --output-dir /out \
        --min-seed-distance 15 --min-size 100
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import label, maximum_filter
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

INPUT_DIR = "/mnt/efs/dl_jrc/student_data/S-JM/train/labels/curated_tissue_masks"


def compute_affinities(seg: np.ndarray, nhood: list) -> np.ndarray:
    """Binary affinities for each offset in nhood.

    Returns float32 array of shape (n_edges, *seg.shape).
    Affinity is 1 where both pixels share the same non-zero label.
    """
    nhood = np.array(nhood)
    shape = seg.shape
    n_edges = nhood.shape[0]
    dims = nhood.shape[1]
    affinity = np.zeros((n_edges,) + shape, dtype=np.float32)

    for e in range(n_edges):
        slices_src = tuple(
            slice(max(0, -nhood[e, d]), min(shape[d], shape[d] - nhood[e, d]))
            for d in range(dims)
        )
        slices_dst = tuple(
            slice(max(0, nhood[e, d]), min(shape[d], shape[d] + nhood[e, d]))
            for d in range(dims)
        )
        affinity[(e,) + slices_src] = (
            (seg[slices_src] == seg[slices_dst])
            * (seg[slices_src] > 0)
            * (seg[slices_dst] > 0)
        ).astype(np.float32)

    return affinity


def semantic_to_instance(
    semantic: np.ndarray,
    min_seed_distance: int = 10,
    min_size: int = 64,
) -> np.ndarray:
    """Convert a semantic mask to a unique-label instance mask.

    Args:
        semantic: Integer array where 0 = background and >0 = foreground.
        min_seed_distance: Side length of the maximum-filter window used to
            find seed points. Larger values merge nearby seeds.
        min_size: Objects smaller than this many voxels are discarded.

    Returns:
        uint32 array with the same shape as ``semantic``.
    """
    ndim = semantic.ndim
    nhood = [[0, 0, 1], [0, 1, 0], [1, 0, 0]] if ndim == 3 else [[0, 1], [1, 0]]

    foreground = semantic > 0

    # Affinities: 1 inside objects, 0 at object-background boundaries
    affs = compute_affinities(semantic, nhood)
    # Mean affinity is high at object interiors, lower near edges
    mean_aff = affs.mean(axis=0)

    # Seeds: local maxima of the affinity map restricted to foreground
    max_filtered = maximum_filter(mean_aff, size=min_seed_distance)
    maxima = (max_filtered == mean_aff) & foreground
    seeds, n_seeds = label(maxima)

    if n_seeds == 0:
        return np.zeros(semantic.shape, dtype=np.uint32)

    # Watershed floods from seeds toward the boundaries (hence negated height)
    instance_mask = watershed(-mean_aff, markers=seeds, mask=foreground)

    instance_mask = remove_small_objects(
        instance_mask.astype(np.int64), min_size=min_size, connectivity=1
    )

    return instance_mask.astype(np.uint32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert semantic segmentation tifs to instance segmentation tifs."
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
        help="Maximum-filter window size for seeding (pixels). Default: 10.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=64,
        help="Minimum object size in voxels; smaller fragments are removed. Default: 64.",
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
        )
        out_path = output_dir / path.name
        tifffile.imwrite(str(out_path), instance)
        n_instances = int((np.unique(instance) != 0).sum())
        print(f"{n_instances} instances -> {out_path.name}")

    print(f"Done. Wrote {len(tif_files)} file(s) to {output_dir}")


if __name__ == "__main__":
    main()
