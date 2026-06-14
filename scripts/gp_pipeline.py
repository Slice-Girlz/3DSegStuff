# A script to run Gunpowder pipeline
# 
# Input: zarr file

# Imports
import gunpowder as gp
from funlib.geometry import Roi 
from funlib.persistence import open_ds, Array
from smooth_augment import SmoothAugment
from helper_imshow_gp import imshow
import os

def train(
    samples,
    n_training_steps = 100,
    channel = 1,
    input_shape = [1, 16, 128, 128],
    output_shape = ...,
    voxel_size = ...,
    batch_size = 5, 
    prob_augment = 0.3, 
    var_noise = 10e-5,
    neighborhood = [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    save_checkpoints_every,
    save_snapshots_every, 
):
    """

    Inputs:
    - samples                     # List of directories to omezarr files
    - channel                     # This is the channel that you want to do your segmentations in
    - input_shape                 # Patch size
    - batch_size                  # Choose batch size appropriately
    - prob_augment                # Probability of noise augments (shared by gaussian and poisson)
    - var_noise                   # Variance of gaussian noise
    - neighborhood                # Neighborhoods to compute affinities from 
    - save_checkpoints_every      # 
    - save_snapshots_every        #

    """

    # Declare array keys
    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    gt_affs = gp.ArrayKey("GT_AFFS")
    affs_weights = gp.ArrayKey("AFFS_WEIGHTS")
    gt_affs_mask = gp.ArrayKey("AFFS_MASK")
    pred_affs = gp.ArrayKey("PRED_AFFS")

    # Model training setup
    # model = ...
    # model.train()
    # loss = ...
    # optimizer = ...

    # Prepare request
    input_size = gp.Coordinate(input_shape) * voxel_size
    output_size = gp.Coordinate(output_shape) * voxel_size

    # Request a batch   
    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(gt_affs, output_size)
    request.add(affs_weights, output_size)
    request.add(pred_affs, output_size)
    
    # Declare data source
    for sample in samples:
       source = tuple(
          gp.ArraySource(raw, open_ds(sample["raw"]), True),
          gp.ArraySource(labels, open_ds(sample["labels"]), False)
       )
       + gp.Normalize(raw)
       + gp.Pad(raw, None)      # Add infinite padding?
       + gp.Pad(labels, None)   # Add infinite padding? 
       + gp.RandomLocation()

    # Prepare augmentations
    simple_augment = gp.SimpleAugment(
      transpose_only = (2, 3)   # Only transpose XY
    )
    elastic_augment = gp.DeformAugment(
      control_point_spacing = (10 * voxel_size[-2], 10 * voxel_size[-1]),
      jitter_sigma = (1.5 * voxel_size[-2], 1.5 * voxel_size[-1]),
      rotate = True, 
      spatial_dims = 2            # Only deform XY
    )
    intensity_augment = gp.IntensityAugment(
      raw,
      scale_min=0.8,
      scale_max=1.2,
      shift_min=0.2,
      shift_max=0.2)
    gaussian_noise_augment = gp.NoiseAugment(raw, mode='gaussian', p=prob_augment, var=var_noise, clip=True)
    poisson_noise_augment = gp.NoiseAugment(raw, mode='poisson', p=prob_augment, clip=True)
    smooth_augment = SmoothAugment(raw, p=prob_augment)

    # Prepare affinities
    affinities = gp.AddAffinities(
        affinity_neighborhood=neighborhood,
        labels=labels,
        affinities=gt_affs
    )

    # Prepare random provider
    random_provider = gp.RandomProvider()

    # Prepare batch 
    stack = gp.Stack(batch_size)

    # Prepare snapshot
    snapshot = gp.Snapshot(
        dataset_names={
            raw: "raw",
            gt_affs: "gt_affs",
            pred_affs: "pred_affs",
            affs_weights: "affs_weights",
        },
        output_filename="batch_{iteration}.zarr",
        output_dir=os.path.join(setup_dir, "snapshots"),
        every=save_snapshots_every,
    )

    ##########################################################

    # Set up pipeline = sequence of nodes in order:
    pipeline = (
        source +
        random_provider + 
        simple_augment + 
        elastic_augment + 
        intensity_augment + 
        gaussian_noise_augment + 
        poisson_noise_augment +
        smooth_augment +
        affinities + 
        stack)

    ##########################################################

    # Build the pipeline
    with gp.build(pipeline):
      for step in range(n_training_steps):
         pipeline.request_batch(request)

