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
    python -m pip install aicspylibczi                        # for .czi
    python -m pip install nd2                             # for .nd2
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image, write_labels


# File types this pipeline understands. Add an extension here and a matching
# branch in load_array() to support a new format.
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".tif", ".tiff", ".czi", ".nd2")


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

    aicspylibczi returns an array padded with several length-1 dimensions
    (e.g. B, V, C, T, Z, Y, X, 0). We squeeze the singletons so the result is a
    clean array comparable to a tif read (e.g. ZYX or CZYX).
    """
    try:
        from aicspylibczi import CziFile
    except ImportError as e:
        raise ImportError("Reading .czi needs `aicspylibczi` (pip install aicspylibczi).") from e
    

    #return np.squeeze(CziFile(str(path)).read_image()), CziFile(str(path)).metadata()
    return np.squeeze(CziFile.imread(str(path))), CziFile(str(path)).metadata()
    


def _load_nd2(path: Path) -> np.ndarray:
    """Read a Nikon .nd2 file with the `nd2` package (returns a NumPy array)."""
    try:
        import nd2
    except ImportError as e:
        raise ImportError("Reading .nd2 needs `nd2` (pip install nd2).") from e
    r#eturn nd2.imread(str(path))
    return nd2.ND2File(str(path)).imread(), nd2.ND2File(str(path)).metadata()

##TODO - META DATA LOADING.

##PRE-PROCESSING OUTPUT in T, C, Z, Y, X
##PREPROCESSING ENSURES LABEL AND IMAGE HAS SAME SHAPE

# ---------------------------------------------------------------------------
# Step 3: write one array to one .ome.zarr that includes images and masks
# ---------------------------------------------------------------------------
def save_to_zarr(
    image,
    label,
    sample_id,
    chunk_size, #(,,,)
    save_path = "./dataset.zarr",
    axes = "tcxyz",
):
    # if axes != "tczyx":
    #     print(axes)
    #     raise ValueError("Expected axes T C Z Y X")
    
    if len(chunk_size) != 5:
        raise ValueError("Expected chunk_size must have 5 items")
    
    store = parse_url(save_path, mode="w").store
    root = zarr.group(store=store)
    
    grp = root.create_group(sample_id)
    
    write_image(
        image=image,
        group=grp,
        axes=axes,
        storage_options={"chunks": chunk_size},
        scaler= None,
    )
    
    write_labels(
        labels=label,
        group=grp,
        name="labels", # appears under labels/cells
        axes=axes,
        storage_options={"chunks": chunk_size},
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
        default=["t", "c", "z", "y", "x"],
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
    
    # tmp, hacky
    img0 = img0[np.newaxis, np.newaxis, ...]
    msk0 = msk0[np.newaxis, np.newaxis, ...]
    
    print(f"First image: {Path(image_files[0]).name}  shape={img0.shape}  dtype={img0.dtype}")
    print(f"First mask:  {Path(mask_files[0]).name}  shape={msk0.shape}  dtype={msk0.dtype}")

    # Step 3 -- write each image to its OWN .ome.zarr (no stacking)
    out_dir = args.output_dir or (args.images_dir.expanduser().resolve().parent / "ome_zarr")
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")
    
    chunk_size = (1,1,64,64,64) #T,C,Z,Y,X ~0.5MB per chunk

    #for f in image_files:
    for i in range(len(image_files)):
        image = load_array(image_files[i])
        mask = load_array(mask_files[i])
        save_path = out_dir / f"{Path(image_files[i]).stem}.ome.zarr"
        if save_path.exists():
            shutil.rmtree(save_path)
        save_to_zarr(image=image, label=mask, sample_id=Path(image_files[i]).stem, chunk_size=chunk_size, save_path=save_path, axes=args.axes
                     #str(out_path), axes=args.axes, 
                     )
        print(f"Wrote {save_path.name}  shape={image.shape}  dtype={mask.dtype}")

    print(f"Done. Wrote {len(image_files)} .ome.zarr files to {out_dir}")


if __name__ == "__main__":
    main()