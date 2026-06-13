# %%

def preprocess(image_array, image_dims=[t,c,z,y,x], label_array=['uint16'], d_type=['float64', 'float32', 'float16', 'unit16'], return_mask=['True', 'False'], normalize ):l

    def make_mask(label_array):

    def min_max_normalize(volume: np.ndarray) -> np.ndarray:
        v_min = volume.min()
        v_max = volume.max()
        if v_max == v_min:
            return np.zeros_like(volume, dtype=np.float32)
        return ((volume - v_min) / (v_max - v_min)).astype(np.float32)

  
    

    

    
