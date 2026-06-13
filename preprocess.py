import numpy as np

def preprocess(
    image_array: np.ndarray,
    label_array: np.ndarray,
    image_dims=("t", "c", "z", "y", "x"),
    normalize=True,
    normalize_method="percentile",
    return_mask=False,
):
    """
    Preprocess microscopy image and label data.

    Output:
        image_out: float32
        label_out: uint16
        optional mask: bool
    """

    image_array, label_array = check_image_label_input(
        image_array=image,
        label_array=label,
        image_dims=("t", "c", "z", "y", "x"),
    )

    image_out = image_array.astype(np.float32)
    label_out = label_array.astype(np.uint16)

    if normalize:
        pass

    if return_mask:
        pass


def check_image_label_input(
    image_array: np.ndarray,
    label_array: np.ndarray,
    image_dims=("t", "c", "z", "y", "x"),
    allowed_image_dtypes=(
        "float64",
        "float32",
        "float16",
        "uint8",
        "uint16",
        "uint32",
        "int8",
        "int16",
        "int32",
        "int64",
    ),
    allowed_label_dtype="uint16",
):
    """
    Check:
    1. image and label follow standard (t, c, z, y, x)
    2. image and label have the same shape
    3. image dtype is allowed
    4. label dtype is uint16
    """

    image_array = np.asarray(image_array)
    label_array = np.asarray(label_array)

    expected_dims = ("t", "c", "z", "y", "x")

    # Check dimension order
    if tuple(image_dims) != expected_dims:
        raise ValueError(
            f"Expected image_dims to be {expected_dims}, "
            f"but got {tuple(image_dims)}."
        )

    # Check both arrays are 5D
    if image_array.ndim != 5:
        raise ValueError(
            f"image_array must be 5D with order (t, c, z, y, x), "
            f"but got shape {image_array.shape}."
        )

    if label_array.ndim != 5:
        raise ValueError(
            f"label_array must be 5D with order (t, c, z, y, x), "
            f"but got shape {label_array.shape}."
        )

    # Check same shape
    if image_array.shape != label_array.shape:
        raise ValueError(
            f"image_array and label_array must have the same shape.\n"
            f"image_array shape: {image_array.shape}\n"
            f"label_array shape: {label_array.shape}"
        )

    # Check image dtype
    image_dtype = str(image_array.dtype)

    if image_dtype == "unit16":
        raise TypeError("Invalid image dtype 'unit16'. Did you mean 'uint16'?")

    if image_dtype not in allowed_image_dtypes:
        raise TypeError(
            f"image_array dtype must be one of {allowed_image_dtypes}, "
            f"but got {image_dtype}."
        )

    # Check label dtype
    label_dtype = str(label_array.dtype)

    if label_dtype != allowed_label_dtype:
        raise TypeError(
            f"label_array dtype must be {allowed_label_dtype}, "
            f"but got {label_dtype}."
        )

    return image_array, label_array