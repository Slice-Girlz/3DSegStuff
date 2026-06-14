import numpy as np
from typing import Literal


def preprocess(
    image_array: np.ndarray, 
    label_array: np.ndarray, 
    image_dims: str = "czyx",
    label_dims: str = "czyx",
    normalize : Literal['min_max','percentile'] | None = 'percentile'
):
    # Check dtype
    # check_dtype(image_array=image_array, label_array=label_array)

    check_dtype(input_array=image_array,allowed_input_dtypes = [
        "float64",
        "float32",
        "float16",
        "uint8",
        "uint16",
        "uint32",
    ])

    check_dtype(input_array=label_array, allowed_input_dtypes = [
        "uint8",
        "uint16",
        "uint32",
    ])

    check_dshape(image_array=image_array, label_array=label_array)

    # Fix dims
    image_array = fix_dims(image_array, input_dims=image_dims)
    label_array = fix_dims(label_array, input_dims=label_dims).astype(np.uint32)
    

    # Normalize
    if normalize is not None:    
        if normalize == 'min_max':
            image_array = min_max_normalize(image_array)
        elif normalize == 'percentile':
            image_array = percentile_normalize(image_array)

    return image_array, label_array




def preprocess_noLabel(
    image_array: np.ndarray, 
    image_dims: str = "czyx",
    normalize : Literal['min_max','percentile'] | None = 'percentile'
):
    # Check dtype
    check_dtype(input_array=image_array, allowed_input_dtypes=[
        "float64",
        "float32",
        "float16",
        "uint8",
        "uint16",
        "uint32",
    ])

    # Fix dims
    image_array = fix_dims(image_array, input_dims=image_dims)

    # Normalize
    if normalize is not None:    
        if normalize == 'min_max':
            image_array = min_max_normalize(image_array)
        elif normalize == 'percentile':
            image_array = percentile_normalize(image_array)

    return image_array


# ==========================
def check_dtype(
    input_array: np.ndarray,

    allowed_dtypes: list
):

    # allowed_image_dtypes = [
    #     "float64",
    #     "float32",
    #     "float16",
    #     "uint8",
    #     "uint16",
    #     "uint32",
    # ],
    # allowed_label_dtypes = [
    #     "uint8",
    #     "uint16",
    #     "uint32",
    # ],

    """
    Check:
    1. image and label follow standard (c, z, y, x)
    2. image and label have the same shape
    3. image dtype is allowed
    4. label dtype is uint16
    """
    input_array = np.asarray(input_array)

    # Check input dtype
    input_dtype = str(input_array.dtype)

    if input_dtype not in allowed_dtypes:
        raise TypeError(
            f"input_array dtype must be one of {allowed_dtypes}, "
            f"but got {input_dtype}."
        )

    # if input_array.shape != label_array.shape:
    #     raise ValueError(
    #         f"image_array and label_array must have the same shape.\n"
    #         f"image_array shape: {input_array.shape}\n"
    #         f"label_array shape: {label_array.shape}"
    #     )

def check_dshape(
    image_array: np.ndarray,
    label_array: np.ndarray,
):
    image_array = np.asarray(image_array)
    label_array = np.asarray(label_array)

    # Check same shape
    if image_array.shape != label_array.shape:
        raise ValueError(
            f"image_array and label_array must have the same shape.\n"
            f"image_array shape: {image_array.shape}\n"
            f"label_array shape: {label_array.shape}"
        )


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
    1. image and label follow standard (c, z, y, x)
    2. image and label have the same shape
    3. image dtype is allowed
    4. label dtype is uint16
    """
    image_array = np.asarray(image_array)
    label_array = np.asarray(label_array).astype(np.uint32)
    

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
    
    



def fix_dims(array, input_dims):
    """
    Reorder (and pad) the axes of `array` so it follows the standard ``czyx`` layout.

    Existing axes named in `input_dims` are transposed into ``czyx`` order, and any
    of ``c``, ``z``, ``y``, ``x`` that are missing are inserted as singleton axes.

    Ex: ``input_dims="xycz"`` with an array of shape ``(X, Y, C, Z)`` returns an
    array of shape ``(C, Z, Y, X)``.
    """
    target_dims = "czyx"
    input_dims = input_dims.lower()

    # Validate input dims string
    if len(set(input_dims)) != len(input_dims):
        raise ValueError(f"input_dims must not contain duplicate axes, got '{input_dims}'.")
    
    unknown = set(input_dims) - set(target_dims)
    if unknown:
        raise ValueError(
            f"input_dims contains unknown axes {sorted(unknown)}, "
            f"allowed axes are {sorted(target_dims)}."
        )
    if len(input_dims) != array.ndim:
        raise ValueError(
            f"input_dims '{input_dims}' has {len(input_dims)} axes, "
            f"but array has {array.ndim} dimensions."
        )

    # Insert missing axes as singletons (prepended), tracking their names
    current_dims = list(input_dims)
    for d in target_dims:
        if d not in current_dims:
            array = np.expand_dims(array, axis=0)
            current_dims.insert(0, d)

    # Transpose into czyx order
    order = [current_dims.index(d) for d in target_dims]
    array = np.transpose(array, order)
    return array


# min_max_normalization
def min_max_normalize(image_array):
    
    assert image_array.ndim == 4, (f"Expected image_array with 4 dims [c,z,y,x], got shape {image_array.shape}")
    
    norm_array = np.zeros_like(image_array, dtype=np.float32) #create a empty zero array
    for c in range(image_array.shape[0]):
        volume = image_array[c] # -> volume.shape=(z, y, x)
        v_min = volume.min()
        v_max = volume.max()
        norm_array[c] = ((volume - v_min) / (v_max - v_min + 1e-8))
    return norm_array


# percentile normalization 
def percentile_normalize(image_array, pmin=0.5, pmax=99.5):
    
    assert image_array.ndim == 4, (f"Expected image_array with 4 dims [c,z,y,x], got shape {image_array.shape}")

    norm_array = np.zeros_like(image_array, dtype=np.float32) #create a empty zero array
    for c in range(image_array.shape[0]):
        volume = image_array[c]
        v_min = np.percentile(volume, pmin)
        v_max = np.percentile(volume, pmax)
        norm_array[c] = np.clip((volume - v_min) / (v_max - v_min + 1e-8), 0, 1)

    return norm_array
