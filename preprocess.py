#
import numpy as np


preprocessed_array, preprocessed_label_array = preprocess(image_array, label_array, image_dims=(100, ))

def preprocess(image_array: np.ndarray, 
               label_array: np.ndarray, 
               image_dims=("t" , "c", "z", "y", "x"), 
               d_type=['float64', 'float32', 'float16', 'uint16'], 
               return_mask=True', 'False', 
               normalize=['None', 'min_max','percentile'] 
               ):

    # Normalize
    if normalize == 'None':
        pass
    if normalize == 'min_max':
        image_array = min_max_normalize(image_array, d_type)
    elif normalize == 'max':
        image_array = max_normalize(image_array, d_type)

    # Create Mask
    mask = (label_array > 0).astype(np.uint8)
    =1

# min_max_normalization
def min_max_normalize(image_array, d_type):
    # image_array.shape
    assert image_array.ndim == 5, (f"Expected image_array with 5 dims [t,c,z,y,x], got shape {image_array.shape}")

    norm_array = np.zeros_like(image_array, dtype=d_type) #create a empty zero array
    
    for t in range(image_array.shape[0]):
        for c in range(image_array.shape[1]):
            volume = image_array[t, c]      # -> volume.shape=(z, y, x)
            v_min = volume.min()
            v_max = volume.max()
            if v_max == v_min:
                norm_array[t, c] = np.zeros_like(volume, dtype=d_type)
            else:
                norm_array[t, c] = ((volume - v_min) / (v_max - v_min)).astype(d_type)
    return norm_array


# percentile normalization 
def percentile_normalize(image_array, d_type, pmin, pmax=99):
    # image_array.shape
    assert image_array.ndim == 5, (f"Expected image_array with 5 dims [t,c,z,y,x], got shape {image_array.shape}")

    norm_array = np.zeros_like(image_array, dtype=d_type) #create a empty zero array
    
    for t in range(image_array.shape[0]):
        for c in range(image_array.shape[1]):
            volume = image_array[t, c]
            v_min = np.percentile(volume, pmin)
            v_max = np.percentile(volume, pmax)
            norm_array[t, c] = np.clip((volume - v_min) / (v_max - v_min), 0, 1).astype(d_type)

    return norm_array
    
