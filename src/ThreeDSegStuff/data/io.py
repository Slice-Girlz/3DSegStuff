import json
from pathlib import Path

import numpy as np
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image, write_labels

from ThreeDSegStuff.data.metadata import prepare_metadata


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
# Step 2: load one file (multiple extensions) into (np.ndarray, metadata)
# ---------------------------------------------------------------------------
def load_array(path: str | Path) -> tuple[np.ndarray, object | None]:
    """
    Load a single microscopy file into ``(array, metadata)``.

    Dispatches on the file extension (if/else). Reader libraries are imported
    lazily inside the helpers, so you only need the library for the formats you
    actually open. ``metadata`` is the reader-native object (or ``None`` for
    formats we don't extract it from yet) and is passed through untouched.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in (".tif", ".tiff"):
        arr, meta = _load_tiff(path)
    elif ext == ".czi":
        arr, meta = _load_czi(path)
    elif ext == ".nd2":
        arr, meta = _load_nd2(path)
    else:
        raise ValueError(
            f"Unsupported extension {ext!r} for {path.name}. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return np.asarray(arr), meta


def _load_tiff(path: Path) -> tuple[np.ndarray, object | None]:
    try:
        import tifffile
    except ImportError as e:
        raise ImportError(
            "Reading .tif/.tiff needs `tifffile` (pip install tifffile)."
        ) from e
    # tifffile metadata is format-dependent; not extracted yet.
    return tifffile.imread(str(path)), None


def _load_czi(path: Path) -> tuple[np.ndarray, object | None]:
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

    czi = CziFile(str(path))
    # read_image() returns (data, shape_dims); we only want the array.
    data, _ = czi.read_image()
    return np.squeeze(data), czi.meta  # czi.meta is the metadata XML element


def _load_nd2(path: Path) -> tuple[np.ndarray, object | None]:
    """Read a Nikon .nd2 file with the `nd2` package."""
    try:
        import nd2
    except ImportError as e:
        raise ImportError("Reading .nd2 needs `nd2` (pip install nd2).") from e

    # `metadata` is a cached_property (attribute, not a method); `asarray()`
    # returns the NumPy array. Use a context manager so the handle is closed.
    with nd2.ND2File(str(path)) as f:
        return f.asarray(), f.metadata

##PRE-PROCESSING OUTPUT in C, Z, Y, X
##PREPROCESSING ENSURES LABEL AND IMAGE HAS SAME SHAPE


def _jsonify_metadata(meta: object | None) -> object | None:
    """
    Coerce reader-native metadata into something zarr can store in attrs (JSON).

    Returns the object untouched if already JSON-serializable, an XML string for
    lxml elements (e.g. .czi metadata), or ``str(meta)`` as a last resort.
    """
    if meta is None:
        return None
    try:
        json.dumps(meta)
        return meta
    except (TypeError, ValueError):
        pass
    
    try:
        from lxml import etree
        if isinstance(meta, etree._Element):
            return etree.tostring(meta, pretty_print=True).decode()
    except ImportError:
        pass
    return str(meta)


# ---------------------------------------------------------------------------
# Step 3: write one volume (image + label) to its OWN .ome.zarr
# ---------------------------------------------------------------------------
def save_to_zarr(
    image: np.ndarray, # (C, Z, Y, X)
    label: np.ndarray, # (C, Z, Y, X)
    save_path="./volume.ome.zarr", # path to ONE .ome.zarr (one sample, one frame)
    image_chunks=(1, 64, 64, 64), # C, Z, Y, X
    label_chunks=(1, 64, 64, 64), # C, Z, Y, X
    image_axes="czyx",
    label_axes="czyx",
    label_name="labels",
    image_metadata=None,
):
    """
    Write a single volume to its own OME-Zarr store at the store root.

    Each .ome.zarr holds exactly one sample at one frame: the image lives at the
    root as a (C, Z, Y, X) multiscale, and the label lives under
    labels/<label_name> as a (C, Z, Y, X) multiscale. Reader-native metadata,
    when provided, is stored on the root group attrs (JSON-coerced).

    funlib.persistence-style spatial metadata (voxel_size, offset, units, ...) is
    derived from ``image_metadata`` and stamped onto the image and label arrays so
    the store opens directly in funlib.persistence / funlib.show.neuroglancer /
    gunpowder. See :mod:`ThreeDSegStuff.data.metadata`.
    """
    if image.ndim != 4:
        raise ValueError(f"Expected image with 4 dims (C, Z, Y, X), got shape {image.shape}")
    if label.ndim != 4:
        raise ValueError(f"Expected label with 4 dims (C, Z, Y, X), got shape {label.shape}")
    if len(image_chunks) != 4:
        raise ValueError("Expected image_chunks must have 4 items (C, Z, Y, X)")
    if len(label_chunks) != 4:
        raise ValueError("Expected label_chunks must have 4 items (C, Z, Y, X)")

    store = parse_url(save_path, mode="w").store
    root = zarr.group(store=store)

    write_image(
        image=image,
        group=root,
        axes=image_axes,
        storage_options={"chunks": image_chunks},
        scaler=None,
    )

    write_labels(
        labels=label,
        group=root,
        name=label_name,  # appears under labels/<label_name>
        axes=label_axes,
        storage_options={"chunks": label_chunks},
        scaler=None,
    )

    if image_metadata is not None:
        root.attrs["native_metadata"] = _jsonify_metadata(image_metadata)

    # Stamp funlib-style metadata onto the image and label arrays (level 0). The
    # data lives at "0" (image multiscale) and "labels/<name>/0" (label multiscale).
    funlib_metadata = prepare_metadata(image_metadata)
    funlib_metadata["resolution"] = funlib_metadata["voxel_size"]  # gunpowder reads `resolution`
    root["0"].attrs.update(funlib_metadata)
    root[f"labels/{label_name}/0"].attrs.update(funlib_metadata)

def save_to_zarr_noLabel(
    image: np.ndarray, # (C, Z, Y, X)
    save_path="./volume.ome.zarr", # path to ONE .ome.zarr (one sample, one frame)
    image_chunks=(1, 64, 64, 64), # C, Z, Y, X
    image_axes="czyx",
    image_metadata=None,
):
    """
    Write a single volume to its own OME-Zarr store at the store root.

    Each .ome.zarr holds exactly one sample at one frame: the image lives at the
    root as a (C, Z, Y, X) multiscale, and the label lives under
    labels/<label_name> as a (C, Z, Y, X) multiscale. Reader-native metadata,
    when provided, is stored on the root group attrs (JSON-coerced).

    funlib.persistence-style spatial metadata (voxel_size, offset, units, ...) is
    derived from ``image_metadata`` and stamped onto the image and label arrays so
    the store opens directly in funlib.persistence / funlib.show.neuroglancer /
    gunpowder. See :mod:`ThreeDSegStuff.data.metadata`.
    """
    if image.ndim != 4:
        raise ValueError(f"Expected image with 4 dims (C, Z, Y, X), got shape {image.shape}")
    # if label.ndim != 4:
    #     raise ValueError(f"Expected label with 4 dims (C, Z, Y, X), got shape {label.shape}")
    if len(image_chunks) != 4:
        raise ValueError("Expected image_chunks must have 4 items (C, Z, Y, X)")
    # if len(label_chunks) != 4:
    #     raise ValueError("Expected label_chunks must have 4 items (C, Z, Y, X)")

    store = parse_url(save_path, mode="w").store
    root = zarr.group(store=store)

    write_image(
        image=image,
        group=root,
        axes=image_axes,
        storage_options={"chunks": image_chunks},
        scaler=None,
    )

    # write_labels(
    #     labels=label,
    #     group=root,
    #     name=label_name,  # appears under labels/<label_name>
    #     axes=label_axes,
    #     storage_options={"chunks": label_chunks},
    #     scaler=None,
    # )

    if image_metadata is not None:
        root.attrs["native_metadata"] = _jsonify_metadata(image_metadata)

    # Stamp funlib-style metadata onto the image and label arrays (level 0). The
    # data lives at "0" (image multiscale) and "labels/<name>/0" (label multiscale).
    funlib_metadata = prepare_metadata(image_metadata)
    funlib_metadata["resolution"] = funlib_metadata["voxel_size"]  # gunpowder reads `resolution`
    root["0"].attrs.update(funlib_metadata)
    # root[f"labels/{label_name}/0"].attrs.update(funlib_metadata)