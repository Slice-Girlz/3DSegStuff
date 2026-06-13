#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 17:19:50 2026

@author: hoblos.h
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert paired microscopy images + ground-truth masks into an OME-Zarr dataset.

Expected folder layout
-----------------------
    <root>/
        images/   raw images                          (.tif, .tiff, .czi, .nd2)
        masks/    ground-truth segmentation masks      (.tif, .tiff, .czi, .nd2)

Each file is written to its OWN .ome.zarr (no stacking): one input file ->
one output dataset.

Implemented here
----------------
    Step 1  list_files()  -> read a folder into a sorted list[str] of file paths
    Step 2  load_array()  -> load one file (any supported extension) to np.ndarray
    Step 3  save_to_zarr() -> write one array to an .ome.zarr (write_image)

Install (per format you actually use)
--------------------------------------
    python -m pip install numpy tifffile ome-zarr zarr   # always
    python -m pip install czifile                        # for .czi
    python -m pip install nd2                             # for .nd2
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np


# File types this pipeline understands. Add an extension here and a matching
# branch in load_array() to support a new format.
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".tif", ".tiff", ".czi", ".nd2")


# ---------------------------------------------------------------------------
# Sorting helper
# ---------------------------------------------------------------------------
def natural_key(path: Path):
    """Sort 1.tif, 2.tif, ..., 10.tif in human order instead of 1, 10, 11, 2."""
    parts = re.split(r"(\d+)", path.stem)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


# ---------------------------------------------------------------------------
# Step 1: read a directory into a sorted list[str]
# ---------------------------------------------------------------------------
def list_files(directory: str | Path) -> list[str]:
    """
    Return a naturally-sorted list of supported file paths (as strings).

    Used for both the images folder and the masks folder, because both can
    contain any of the supported extensions.
    """
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    files = sorted(
        (
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=natural_key,
    )
    if not files:
        raise FileNotFoundError(
            f"No supported files {SUPPORTED_EXTENSIONS} found in {directory}"
        )
    return [str(p) for p in files]


# ---------------------------------------------------------------------------
# Step 2: load one file (multiple extensions) into np.ndarray
# ---------------------------------------------------------------------------
def load_array(path: str | Path) -> np.ndarray:
    """
    Load a single microscopy file into a NumPy array.

    Dispatches on the file extension (if/else). Reader libraries are imported
    lazily inside the helpers, so you only need the library for the formats you
    actually open.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in (".tif", ".tiff"):
        arr = _load_tiff(path)
    elif ext == ".czi":
        arr = _load_czi(path)
    elif ext == ".nd2":
        arr = _load_nd2(path)
    else:
        raise ValueError(
            f"Unsupported extension {ext!r} for {path.name}. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return np.asarray(arr)


def _load_tiff(path: Path) -> np.ndarray:
    try:
        import tifffile
    except ImportError as e:
        raise ImportError(
            "Reading .tif/.tiff needs `tifffile` (pip install tifffile)."
        ) from e
    return tifffile.imread(str(path))


def _load_czi(path: Path) -> np.ndarray:
    """
    Read a Zeiss .czi file.

    czifile returns an array padded with several length-1 dimensions
    (e.g. B, V, C, T, Z, Y, X, 0). We squeeze the singletons so the result is a
    clean array comparable to a tif read (e.g. ZYX or CZYX).
    """
    try:
        import czifile
    except ImportError as e:
        raise ImportError("Reading .czi needs `czifile` (pip install czifile).") from e
    return np.squeeze(czifile.imread(str(path)))


def _load_nd2(path: Path) -> np.ndarray:
    """Read a Nikon .nd2 file with the `nd2` package (returns a NumPy array)."""
    try:
        import nd2
    except ImportError as e:
        raise ImportError("Reading .nd2 needs `nd2` (pip install nd2).") from e
    return nd2.imread(str(path))


# ---------------------------------------------------------------------------
# Step 3: write one array to an .ome.zarr  (colleague's save_to_zarr, debugged)
# ---------------------------------------------------------------------------
def save_to_zarr(
                arrays,
                path,
                **zarr_parameters):

    """ Saves the chunked arrays in an ome-zarr file.

    Args:
        Arrays: a list of chunked arrays
        Path: file path of where to save the data

        zarr_parameters: flexible arguments that you will pass in as metadata. If they are the specific

    Returns:
        none, saves an ome-zarr object at specified path.

    Raises
    """
    assert (arrays is not None), "arrays are missing"
    assert (path is not None), "path is missing"

    try:
        zarr_parameters
        print("Metadata detected. \n")
        print(*zarr_parameters)

    except NameError:
        print("No metadata detected for ", path)

    from ome_zarr.writer import write_image

    write_image(
        image=arrays,
        group=path,
        **zarr_parameters
    )


# ---------------------------------------------------------------------------
# CLI / driver
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read paired image/mask folders and write one .ome.zarr per file."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Folder of raw images (.tif/.tiff/.czi/.nd2).",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        required=True,
        help="Folder of ground-truth masks (.tif/.tiff/.czi/.nd2).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the .ome.zarr files. Default: <images-dir parent>/ome_zarr",
    )
    parser.add_argument(
        "--axes",
        default="zyx",
        help="Axis order of each input array. Default: zyx (your data is 3D Z,Y,X).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Step 1 -- read both directories into list[str]
    image_files = list_files(args.images_dir)
    mask_files = list_files(args.masks_dir)

    print(f"Images dir: {args.images_dir}  ->  {len(image_files)} files")
    print(f"Masks  dir: {args.masks_dir}  ->  {len(mask_files)} files")
    print("First images:", [Path(p).name for p in image_files[:5]])
    print("First masks: ", [Path(p).name for p in mask_files[:5]])

    if len(image_files) != len(mask_files):
        print(
            f"WARNING: image count ({len(image_files)}) "
            f"!= mask count ({len(mask_files)}). They should pair 1:1."
        )

    # Step 2 -- load the first of each as a sanity check
    img0 = load_array(image_files[0])
    msk0 = load_array(mask_files[0])
    print(f"First image: {Path(image_files[0]).name}  shape={img0.shape}  dtype={img0.dtype}")
    print(f"First mask:  {Path(mask_files[0]).name}  shape={msk0.shape}  dtype={msk0.dtype}")

    # Step 3 -- write each image to its OWN .ome.zarr (no stacking)
    out_dir = args.output_dir or (args.images_dir.expanduser().resolve().parent / "ome_zarr")
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    for f in image_files:
        data = load_array(f)
        out_path = out_dir / f"{Path(f).stem}.ome.zarr"
        if out_path.exists():
            shutil.rmtree(out_path)
        save_to_zarr(data, str(out_path), axes=args.axes)
        print(f"Wrote {out_path.name}  shape={data.shape}  dtype={data.dtype}")

    print(f"Done. Wrote {len(image_files)} .ome.zarr files to {out_dir}")


if __name__ == "__main__":
    main()