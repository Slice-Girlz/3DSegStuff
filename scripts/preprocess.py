import numpy as np
from typing import Literal



def preprocess(
    image_array: np.ndarray, 
    label_array: np.ndarray, 
    image_dims, 
    normalize : Literal['min_max','percentile'] | None = 'percentile'
):
    # Check dtype
    image_array, label_array = check_dtype(
        image_array=image_array,
        label_array=label_array,
    )

    # Fix dims
    image_array = fix_dims(image_dims, image_array)
    label_array = fix_dims(image_dims, label_array)

    # Normalize
    if normalize is not None:    
        if normalize == 'min_max':
            image_array = min_max_normalize(image_array)
        elif normalize == 'percentile':
            image_array = percentile_normalize(image_array)

    return image_array, label_array
    

# ==========================
def check_dtype(
    image_array: np.ndarray,
    label_array: np.ndarray,
    allowed_image_dtypes = [
        "float64",
        "float32",
        "float16",
        "uint8",
        "uint16",
        "uint32",
    ],
    allowed_label_dtypes = [
        "uint8",
        "uint16",
        "uint32",
    ],
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

    # Check same shape
    if image_array.shape != label_array.shape:
        raise ValueError(
            f"image_array and label_array must have the same shape.\n"
            f"image_array shape: {image_array.shape}\n"
            f"label_array shape: {label_array.shape}"
        )

    # Check image dtype
    image_dtype = str(image_array.dtype)

    if image_dtype not in allowed_image_dtypes:
        raise TypeError(
            f"image_array dtype must be one of {allowed_image_dtypes}, "
            f"but got {image_dtype}."
        )

    # Check label dtype
    label_dtype = str(label_array.dtype)

    if label_dtype not in allowed_label_dtypes:
        raise TypeError(
            f"label_array dtype must be one of {allowed_label_dtypes}, "
            f"but got {label_dtype}."
        )


def fix_dims(image_dims, array):
    expected_dims=("t" , "c", "z", "y", "x")
    temp_dims = image_dims
    for i in range(len(expected_dims)):
        dims=expected_dims[i]
        is_present = (dims in temp_dims)
        if not is_present:
            array = np.expand_dims(array, i)
    return array 


# min_max_normalization
def min_max_normalize(image_array):
    norm_array = np.zeros_like(image_array, dtype=np.float32) #create a empty zero array
    
    for t in range(image_array.shape[0]):
        for c in range(image_array.shape[1]):
            volume = image_array[t, c]      # -> volume.shape=(z, y, x)
            v_min = volume.min()
            v_max = volume.max()
            norm_array[t, c] = ((volume - v_min) / (v_max - v_min + 1e-8))
    return norm_array


# percentile normalization 
def percentile_normalize(image_array, pmin=0.5, pmax=99.5):
    # image_array.shape
    assert image_array.ndim == 5, (f"Expected image_array with 5 dims [t,c,z,y,x], got shape {image_array.shape}")

    norm_array = np.zeros_like(image_array, dtype=np.float32) #create a empty zero array
    
    for t in range(image_array.shape[0]):
        for c in range(image_array.shape[1]):
            volume = image_array[t, c]
            v_min = np.percentile(volume, pmin)
            v_max = np.percentile(volume, pmax)
            norm_array[t, c] = np.clip((volume - v_min) / (v_max - v_min + 1e-8), 0, 1)

    return norm_array



