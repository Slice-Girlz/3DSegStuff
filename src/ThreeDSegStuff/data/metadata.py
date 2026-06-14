"""Prepare funlib.persistence-compatible metadata for our (C, Z, Y, X) volumes.

funlib.persistence — and the tools that share its layout (funlib.show.neuroglancer
and gunpowder) — read a few fields from a Zarr array's attributes to understand
an array physically:

    voxel_size  size of one voxel,        spatial axes only -> (z, y, x)
    offset      position of the corner,   spatial axes only -> (z, y, x)
    units       physical unit per axis,    spatial axes only -> (z, y, x)
    axis_names  name of every axis (channel axes end in "^") -> (c, z, y, x)
    types       "channel" / "space" per axis                 -> (c, z, y, x)

`prepare_metadata` reads what it can from the microscope file's own metadata (the
object `io.load_array` returns) and fills everything else with the defaults below.

Note: funlib stores world coordinates as *integers*, so extracted voxel sizes are
expressed in nanometres (e.g. 0.2 um -> 200 nm). Microns would round to zero.
"""

# Defaults for a (C, Z, Y, X) volume, used when the file has no usable metadata.
# voxel_size / offset / units are spatial only (z, y, x); axis_names / types
# cover every axis (the channel axis is non-physical, hence the "^").
DEFAULT_VOXEL_SIZE = (1, 1, 1)
DEFAULT_OFFSET = (0, 0, 0)
DEFAULT_UNITS = ("um", "um", "um")
DEFAULT_AXIS_NAMES = ("c^", "z", "y", "x")
DEFAULT_TYPES = ("channel", "space", "space", "space")


def prepare_metadata(native_metadata=None) -> dict:
    """Build the funlib metadata dict for one (C, Z, Y, X) volume.

    Reads the voxel size (and its unit) from `native_metadata` when we recognise
    the format; anything missing falls back to the module defaults.
    """
    derived = _derive_funlib_from_metadata(native_metadata)
    return {
        "voxel_size": list(derived.get("voxel_size", DEFAULT_VOXEL_SIZE)),
        "offset": list(DEFAULT_OFFSET),
        "units": list(derived.get("units", DEFAULT_UNITS)),
        "axis_names": list(DEFAULT_AXIS_NAMES),
        "types": list(DEFAULT_TYPES),
    }


def _derive_funlib_from_metadata(native_metadata) -> dict:
    """Best-effort read of the (z, y, x) voxel size from native reader metadata.

    Returns a partial dict ({"voxel_size": ..., "units": ...}) when we recognise
    the metadata, or {} so `prepare_metadata` falls back to the defaults.
    """
    if native_metadata is None:
        return {}

    voxel_xyz_um = _nd2_voxel_size_um(native_metadata) or _czi_voxel_size_um(native_metadata)
    if voxel_xyz_um is None:
        return {}

    # microns -> integer nanometres, reordered (x, y, z) -> (z, y, x) for czyx
    x, y, z = voxel_xyz_um
    voxel_size_zyx = [int(round(v * 1000)) for v in (z, y, x)]
    if any(v <= 0 for v in voxel_size_zyx):
        return {}  # incomplete calibration -> use defaults
    return {"voxel_size": voxel_size_zyx, "units": ["nm", "nm", "nm"]}


def _nd2_voxel_size_um(meta):
    """Nikon .nd2 metadata: per-axis calibration in microns, ordered (x, y, z)."""
    x, y, z = (float(v) for v in meta.channels[0].volume.axesCalibration)
    return (x, y, z)


def _czi_voxel_size_um(meta):
    """Zeiss .czi metadata XML: <Distance Id="X|Y|Z"><Value> in metres."""
    from lxml import etree

    if not isinstance(meta, etree._Element):
        return None
    microns = []
    for axis in ("X", "Y", "Z"):
        metres = meta.findtext(f".//Distance[@Id='{axis}']/Value")
        microns.append(float(metres) * 1e6)  # metres -> microns
    return tuple(microns)

