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
    #eturn nd2.imread(str(path))
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