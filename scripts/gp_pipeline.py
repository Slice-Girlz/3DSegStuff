# A script to run Gunpowder pipeline
# 
# Input: zarr file

# Imports
import gunpowder as gp
from funlib.geometry import Roi 
from funlib.persistence import open_ds, Array
from smooth_augment import SmoothAugment
from helper_imshow_gp import imshow

# PARAMETERS
ZARR_PATH = '/mnt/efs/dl_jrc/student_data/S-MS/raw_data_omezarr/AR163_section1_1x1__XYPos_0.ome.zarr/0'
CHANNEL = 1             # This is the channel that you want to do your segmentations in
XY_PATCH_SIZE = 256     # XY size of patch
Z_PATCH_SIZE = 26       # Z size of patch
BATCH_SIZE = 4          # Choose batch size appropriately
PROB_AUGMENT = 0.3      # Probability of noise augments (shared by gaussian and poisson)
VAR_NOISE = 10e-5       # Variance of gaussian noise

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
  jitter_sigma = (1.5 * vs[-2], 1.5 * vs[-1]),
  rotate = True, 
  spatial_dims = 2            # Only deform XY
)
intensity_augment = gp.IntensityAugment(
  raw,
  scale_min=0.8,
  scale_max=1.2,
  shift_min=0.2,
  shift_max=0.2)
gaussian_noise_augment = gp.NoiseAugment(raw, mode='gaussian', p=PROB_AUGMENT, var=VAR_NOISE, clip=True)
poisson_noise_augment = gp.NoiseAugment(raw, mode='poisson', p=PROB_AUGMENT, clip=True)
smooth_augment = SmoothAugment(raw, p=PROB_AUGMENT)

# Batch 
stack = gp.Stack(BATCH_SIZE)

# Pipeline = sequence of nodes:
pipeline = (
    source + 
    normalize + 
    random_location + 
    simple_augment + 
    elastic_augment + 
    intensity_augment + 
    gaussian_noise_augment + 
    poisson_noise_augment +
    smooth_augment +
    stack)

##########################################################

# Request a batch   
request = gp.BatchRequest()
request[raw] = Roi(
  offset = (CHANNEL, 0, 0, 0),
  shape = (1, Z_PATCH_SIZE, XY_PATCH_SIZE, XY_PATCH_SIZE))

# Build the pipeline
with gp.build(pipeline):
  # Request a batch 
  batch = pipeline.request_batch(request)

# Visualize (2D only)
print(batch[raw].data.shape)
imshow(batch[raw].data, z_plane=13)
# Should compare to original to make sure it works

# Do same ish for mask.omezarr