import argparse
import shutil
from pathlib import Path


from 3DSegStuff.data.io import list_files, load_array, save_to_zarr


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