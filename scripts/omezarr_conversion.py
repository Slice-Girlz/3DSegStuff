import argparse
import shutil
from pathlib import Path

from ThreeDSegStuff.data.io import list_files, load_array, save_to_zarr
from ThreeDSegStuff.data.preprocess import preprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read paired image/mask folders and write one .ome.zarr per file."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Folder of raw images (.tif/.tiff/.czi/.nd2).",
    )
    parser.add_argument(
        "--masks-dir",
        type=str,
        required=True,
        help="Folder of ground-truth masks (.tif/.tiff/.czi/.nd2).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Where to write the .ome.zarr files. Default: <images-dir parent>/ome_zarr",
    )
    parser.add_argument(
        "--chunk-size",
        nargs=4,
        default=(1, 64, 64, 64),
        help="Chunk size (C, Z, Y, X) for both image and label arrays. "
        "Default: (1, 64, 64, 64).",
    )
    parser.add_argument(
        "--input-dims",
        type=str,
        required=True,
        help="Axis layout of the raw loaded arrays (e.g. 'zyx' or 'czyx').",
    )
    parser.add_argument(
        "--label-dims",
        type=str,
        required=True,
        help="Axis layout of the raw loaded label arrays (e.g. 'zyx' or 'czyx').",
    )
    parser.add_argument(
        "--normalize",
        choices=["percentile", "min_max", "none"],
        default="percentile",
        help="Image intensity normalization. Default: percentile.",
    )

    parser.add_argument(
        "--voxel_size",
        nargs=3,
        type=int,
        default=None,
        help="Voxel size (Z, Y, X) to stamp on the output, overriding any file "
        "metadata and the (100, 100, 100) default in metadata.py.",
    )
    parser.add_argument(
        "--unit",
        type=str,
        default=None,
        help="Physical unit for the voxel size, overriding the 'nm' default in "
        "metadata.py.",
    )

    parser.add_argument(
        "--make_sparse_mask",
        type=bool,
        required=False,
        help="Create a boolean mask from sparse labels. Save an extra folder named 'sparse_masks' in .ome.zarr "
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Step 1 -- read both directories into list[str]
    image_files = list_files(Path(args.images_dir))
    mask_files = list_files(Path(args.masks_dir))

    print(f"Images dir: {args.images_dir}  ->  {len(image_files)} files")
    print(f"Masks  dir: {args.masks_dir}  ->  {len(mask_files)} files")
    print("First images:", [Path(p).name for p in image_files[:5]])
    print("First masks: ", [Path(p).name for p in mask_files[:5]])

    if len(image_files) != len(mask_files):
        print(
            f"WARNING: image count ({len(image_files)}) "
            f"!= mask count ({len(mask_files)}). They should pair 1:1."
        )

    # Step 2 -- write each volume to its OWN .ome.zarr (one sample, one frame)
    out_dir = Path(args.output_dir)
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    image_chunks = tuple(args.chunk_size)  # (C, Z, Y, X)
    label_chunks = tuple(args.chunk_size)  # (C, Z, Y, X)

    for i in range(len(image_files)):
        image, image_meta = load_array(image_files[i])
        mask, _ = load_array(mask_files[i])
        
        # Homogenize dims to (C, Z, Y, X) and normalize the image
        image, mask = preprocess(
            image_array=image,
            label_array=mask,
            image_dims=args.input_dims,
            label_dims=args.label_dims,
            normalize="percentile",
        )
        
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
            label_axes="czyx",
            image_metadata=image_meta,
            make_sparse_mask = args.make_sparse_mask,
            voxel_size=args.voxel_size,
            unit=args.unit,
        )
        print(f"Wrote {save_path.name}  image={image.shape}  label={mask.shape}  dtype={mask.dtype}")

    print(f"Done. Wrote {len(image_files)} .ome.zarr files to {out_dir}")


if __name__ == "__main__":
    main()