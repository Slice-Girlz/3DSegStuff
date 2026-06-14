# A script to run Gunpowder pipeline
# 
# Input: zarr file

# Imports
import gunpowder as gp
import matplotlib.pyplot as plt
from funlib.geometry import Roi 
from funlib.persistence import open_ds, Array


# PARAMETERS
ZARR_PATH = '/mnt/efs/dl_jrc/student_data/S-MS/raw_data_omezarr/AR163_section1_1x1__XYPos_0.ome.zarr/0'
ZARR_LEVEL = '0'        # This is the name of your lowest resolution zarr folder
CHANNEL = 0             # This is the channel that you want to do your segmentations in
XY_PATCH_SIZE = 256     # XY size of patch
Z_PATCH_SIZE = 1        # Z size of patch
Z_PLANE = 15            # REPLACE BY 0 LATER

# Declare data key
raw = gp.ArrayKey('RAW')

# Declare data source
raw_array = Array(open_ds(ZARR_PATH)[CHANNEL:CHANNEL+1, ...])
vs = raw_array.voxel_size
source = gp.ArraySource(
  raw, raw_array, True
)

# Pick a random sample
random_location  = gp.RandomLocation()

# Normalize 
normalize = gp.Normalize(raw)

# Augmentations
simple_augment = gp.SimpleAugment(
  transpose_only = (2, 3)   # Only transpose XY
)
elastic_augment = gp.DeformAugment(
  control_point_spacing = (10 * vs[-2], 10 * vs[-1]),
  jitter_sigma = (2 * vs[-2], 2 * vs[-1]),
  rotate = True, 
  spatial_dims = 2            # Only deform XY
)
intensity_augment = gp.IntensityAugment(
  raw,
  scale_min=0.8,
  scale_max=1.2,
  shift_min=0.2,
  shift_max=0.2)
noise_augment = gp.NoiseAugment(raw, mode='gaussian', p=1, sigma=(0, 1, 1, 1), channel_axis=0) # Probability of noise defaults to 1

# Pipeline = sequence of nodes:
pipeline = source + normalize + random_location + simple_augment + elastic_augment # + noise_augment

##########################################################

# Request a batch 
request = gp.BatchRequest()
request[raw] = Roi(
  offset = (CHANNEL, Z_PLANE, 0, 0),   # This is a placeholder = overwritten by RandomLocation
  shape = (1, Z_PATCH_SIZE, XY_PATCH_SIZE, XY_PATCH_SIZE))

# Build the pipeline
with gp.build(pipeline):
  # Request a batch 
  batch = pipeline.request_batch(request)

# Visualize (2D only)
print(batch[raw].data.shape)
plt.imshow(batch[raw].data.squeeze(0).squeeze(0))
print(batch[raw].data.squeeze(0).squeeze(0).shape)
# Should compare to original to make sure it works