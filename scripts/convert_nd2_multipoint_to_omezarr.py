"""
Batch convert multipoint ND2 files to OME-Zarr v0.5.
- One OME-Zarr per scene (e.g., 'A1', 'A2', ...)
- Lazy loading via dask (no full-file load into RAM)
- Pixel sizes read from ND2 metadata
- Squeezes T (assumed 1), keeps C, Z, Y, X
- Fixed pyramid depth (default 4 downsamples = 5 levels) — suited to 2048x2048
- Skips outputs that already exist
"""
import argparse
import math
import time
import traceback
from pathlib import Path

import numpy as np
import zarr
import glob as glob_mod
import dask.array as da
from bioio import BioImage
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image

# ===== Defaults (override via CLI args) =====
DEFAULT_INPUT_GLOB = "/mnt/efs/dl_jrc/student_data/S-MS/raw_data_nd2/*.nd2"
DEFAULT_PYRAMID_DEPTH = 4         # 4 downsamples for 2048 → smallest = 128
DEFAULT_CHUNKS = (1, 8, 256, 256) # (C, Z, Y, X)


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB,
                   help="Glob pattern for ND2 input files")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default: same as input)")
    p.add_argument("--pyramid-depth", type=int, default=DEFAULT_PYRAMID_DEPTH,
                   help="Number of pyramid downsamples (levels = depth + 1)")
    p.add_argument("--chunks", nargs=4, type=int, default=list(DEFAULT_CHUNKS),
                   metavar=("C", "Z", "Y", "X"),
                   help="Chunk size in (C, Z, Y, X)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be converted without doing it")
    p.add_argument("--limit-files", type=int, default=None,
                   help="Process only the first N files (for testing)")
    p.add_argument("--limit-scenes", type=int, default=None,
                   help="Process only the first N scenes per file (for testing)")
    return p.parse_args()


def get_channel_color(channel_name: str) -> str:
    """Heuristic mapping from channel name to OMERO hex color."""
    name = channel_name.lower()
    mapping = [
        ("dapi", "0000FF"),
        ("gfp", "00FF00"),
        ("cy3", "FF8000"),
        ("cy5", "FF0000"),
        ("cy7", "FFFFFF"),  # near-IR, just use white
    ]
    for keyword, color in mapping:
        if keyword in name:
            return color
    return "FFFFFF"


def convert_one_scene(img: BioImage, scene_name: str, output_path: Path,
                      pyramid_depth: int, chunks: tuple) -> dict:
    """Convert one scene of an open BioImage to OME-Zarr."""
    status = {"scene": scene_name, "output": output_path.name,
              "skipped": False, "error": None, "duration_s": None}

    if output_path.exists():
        print(f"    SKIP: {output_path.name} already exists")
        status["skipped"] = True
        return status

    t0 = time.time()
    try:
        img.set_scene(scene_name)

        # Lazy dask load: shape (T, C, Z, Y, X), not yet in RAM
        ddata = img.dask_data
        if ddata.shape[0] != 1:
            raise ValueError(f"Expected T=1 in scene '{scene_name}', got T={ddata.shape[0]}")
        ddata = ddata[0]  # (C, Z, Y, X), still lazy
        print(f"    scene='{scene_name}': shape={ddata.shape}, "
              f"dtype={ddata.dtype}, est_size={ddata.nbytes / 1e9:.2f} GB (lazy)")

        # Rechunk to match desired OME-Zarr chunks (helps efficient writing)
        ddata = ddata.rechunk(chunks)

        # Pixel sizes
        ps = img.physical_pixel_sizes
        if ps.Z is None or ps.Y is None or ps.X is None:
            raise ValueError(f"Missing pixel size in metadata: {ps}")

        # Channel names
        channel_names = [str(name) for name in img.channel_names]

        # Scale factors (cumulative) and coordinate transformations
        scale_factors = [
            {"c": 1, "z": 1, "y": 2 ** (level + 1), "x": 2 ** (level + 1)}
            for level in range(pyramid_depth)
        ]
        coord_transforms = []
        for level in range(pyramid_depth + 1):
            yx_factor = 2 ** level
            coord_transforms.append([
                {"type": "scale",
                 "scale": [1.0, ps.Z, ps.Y * yx_factor, ps.X * yx_factor]},
            ])

        axes = [
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]

        # Write
        store = parse_url(output_path, mode="w").store
        root = zarr.group(store=store)
        write_image(
            image=ddata,
            group=root,
            axes=axes,
            coordinate_transformations=coord_transforms,
            scale_factors=scale_factors if scale_factors else None,
            storage_options={"chunks": chunks},
        )

        # OMERO channel metadata for napari coloring
        ome_channels = [
            {"label": name,
             "color": get_channel_color(name),
             "active": True,
             "window": {"start": 0, "end": 65535, "min": 0, "max": 65535}}
            for name in channel_names
        ]
        attrs = dict(root.attrs)
        ome_attrs = attrs.get("ome", {})
        ome_attrs["omero"] = {"channels": ome_channels}
        attrs["ome"] = ome_attrs
        root.attrs.update(attrs)

        status["duration_s"] = time.time() - t0
        print(f"    Done in {status['duration_s']:.1f} s "
              f"({status['duration_s'] / 60:.2f} min)")

    except Exception as e:
        status["error"] = f"{type(e).__name__}: {e}"
        status["duration_s"] = time.time() - t0
        print(f"    ERROR after {status['duration_s']:.1f} s: {status['error']}")
        traceback.print_exc()

    return status


def convert_one_file(nd2_path: Path, args) -> list:
    print(f"  Reading {nd2_path.name}...")
    img = BioImage(nd2_path)
    scenes = list(img.scenes)
    print(f"  Found {len(scenes)} scenes: {scenes[:5]}"
          + (f" ... (and {len(scenes) - 5} more)" if len(scenes) > 5 else ""))

    if args.limit_scenes is not None:
        scenes = scenes[:args.limit_scenes]
        print(f"  Limiting to first {len(scenes)} scenes (--limit-scenes)")

    output_dir = Path(args.output_dir) if args.output_dir else nd2_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, scene in enumerate(scenes, 1):
        # Safe filename: replace any awkward characters in scene name
        safe_scene = scene.replace("/", "_").replace(":", "_")
        output_path = output_dir / f"{nd2_path.stem}__{safe_scene}.ome.zarr"
        print(f"  [scene {i}/{len(scenes)}] '{scene}' → {output_path.name}")

        if args.dry_run:
            print("    (dry run, skipping actual conversion)")
            continue

        results.append(convert_one_scene(
            img, scene, output_path,
            pyramid_depth=args.pyramid_depth,
            chunks=tuple(args.chunks),
        ))
    return results


def main():
    args = get_args()

    nd2_files = [Path(p) for p in sorted(glob_mod.glob(args.input_glob))]

    if args.limit_files is not None:
        nd2_files = nd2_files[:args.limit_files]

    print(f"Found {len(nd2_files)} ND2 file(s) matching {args.input_glob}")
    for f in nd2_files:
        print(f"  {f.name} ({f.stat().st_size / 1e9:.2f} GB)")
    print()
    if args.dry_run:
        print("=== DRY RUN ===")
    print()

    all_results = []
    t0 = time.time()
    for i, nd2 in enumerate(nd2_files, 1):
        print(f"[file {i}/{len(nd2_files)}] {nd2.name}")
        try:
            all_results.extend(convert_one_file(nd2, args))
        except Exception as e:
            print(f"  ERROR opening file: {e}")
            traceback.print_exc()
        print()

    # Summary
    elapsed = time.time() - t0
    print("=" * 60)
    print(f"SUMMARY (elapsed: {elapsed / 60:.1f} min)")
    print("=" * 60)
    if args.dry_run:
        print("(dry run — no conversions performed)")
        return
    n_ok = sum(1 for r in all_results if r["error"] is None and not r["skipped"])
    n_skip = sum(1 for r in all_results if r["skipped"])
    n_err = sum(1 for r in all_results if r["error"] is not None)
    print(f"Scenes converted: {n_ok}")
    print(f"Scenes skipped (already existed): {n_skip}")
    print(f"Scenes errored: {n_err}")
    if n_err:
        print("\nErrors:")
        for r in all_results:
            if r["error"]:
                print(f"  {r['scene']}: {r['error']}")


if __name__ == "__main__":
    main()