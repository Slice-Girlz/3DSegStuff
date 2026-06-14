import argparse
import shutil
from pathlib import Path

import numpy as np

from ThreeDSegStuff.data.io import list_files, load_array, save_to_zarr


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
        "--chunk-size",
        default=(1, 64, 64, 64),
        help="Image chunk size (C, Z, Y, X). Default: (1, 64, 64, 64). "
        "Label chunks are derived as (Z, Y, X).",
    )
    return parser.parse_args()


def _as_image(arr: np.ndarray) -> np.ndarray:
    """Normalize an input image to (C, Z, Y, X)."""
    if arr.ndim == 3:  # (Z, Y, X) -> add a single channel
        arr = arr[np.newaxis, ...]
    if arr.ndim != 4:
        raise ValueError(f"Expected image with 3 (ZYX) or 4 (CZYX) dims, got shape {arr.shape}")
    return arr


def _as_label(arr: np.ndarray) -> np.ndarray:
    """Normalize an input label to (Z, Y, X), dropping leading singleton axes."""
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected label reducible to 3 dims (ZYX), got shape {arr.shape}")
    return arr


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
    img0 = _as_image(load_array(image_files[0]))
    msk0 = _as_label(load_array(mask_files[0]))

    print(f"First image: {Path(image_files[0]).name}  shape={img0.shape}  dtype={img0.dtype}")
    print(f"First mask:  {Path(mask_files[0]).name}  shape={msk0.shape}  dtype={msk0.dtype}")

    # Step 3 -- write each volume to its OWN .ome.zarr (one sample, one frame)
    out_dir = args.output_dir or (args.images_dir.expanduser().resolve().parent / "ome_zarr")
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    image_chunks = tuple(args.chunk_size)            # (C, Z, Y, X)
    label_chunks = tuple(args.chunk_size)[1:]        # (Z, Y, X)

    for i in range(len(image_files)):
        image = _as_image(load_array(image_files[i]))
        mask = _as_label(load_array(mask_files[i]))
        save_path = out_dir / f"{Path(image_files[i]).stem}.ome.zarr"
        if save_path.exists():
            shutil.rmtree(save_path)
        save_to_zarr(
            image=image,
            label=mask,
            save_path=save_path,
            image_chunks=image_chunks,
            label_chunks=label_chunks,
            image_axes="czyx",
            label_axes="zyx",
        )
        print(f"Wrote {save_path.name}  image={image.shape}  label={mask.shape}  dtype={mask.dtype}")

    print(f"Done. Wrote {len(image_files)} .ome.zarr files to {out_dir}")


if __name__ == "__main__":
    main()