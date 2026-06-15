# A script to run Gunpowder pipeline
# 
# Input: zarr file

# Imports
import gunpowder as gp
from funlib.geometry import Roi, Coordinate
from funlib.persistence import open_ds, Array
#from smooth_augment import SmoothAugment
import os
import logging 
import glob

logging.basicConfig(level=logging.INFO)

def train(
    input_dir,
    output_dir,
    n_training_steps = 10,
    channel = 1,
    input_shape = [1, 16, 128, 128],
    output_shape = [1, 14, 124, 124],
    batch_size = 1, 
    prob_augment = 0.3, 
    var_noise = 10e-5,
    neighborhood = [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    save_snapshots_every = 1
):
    """

    Inputs:
    - input_dir                   # Directory with omezarr files
    - output_dir                  # Directory to store outputs in 
    - channel                     # This is the channel that you want to do your segmentations in
    - input_shape                 # Patch size
    - batch_size                  # Choose batch size appropriately
    - prob_augment                # Probability of noise augments (shared by gaussian and poisson)
    - var_noise                   # Variance of gaussian noise
    - neighborhood                # Neighborhoods to compute affinities from 
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

    samples = [
      {"raw": os.path.join(f, "0"), "labels": os.path.join(f, "labels/labels/0")}
      for f in sorted(glob.glob(os.path.join(input_dir, "*ome.zarr")))
      ]
    voxel_size = open_ds(samples[0]["raw"]).voxel_size

    # Prepare request
    input_size = gp.Coordinate(input_shape[1:]) * voxel_size
    output_size = gp.Coordinate(output_shape[1:]) * voxel_size

    print(input_shape, input_size, output_shape, output_size)

    # Request a batch   
    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(gt_affs, output_size)
    #request.add(affs_weights, output_size)
    #request.add(pred_affs, output_size)
    
    # Get samples and declare data source
    source = tuple((
        gp.ArraySource(raw, open_ds(sample["raw"]), True),
        gp.ArraySource(labels, Array(open_ds(sample["labels"])[0], voxel_size=voxel_size), False)
      )
      + gp.MergeProvider()
      + gp.Normalize(raw)
      + gp.Pad(raw, output_size)      # Add infinite padding?
      + gp.Pad(labels, output_size)   # Add infinite padding? 
      + gp.RandomLocation()
      for sample in samples)

    # Prepare augmentations
    simple_augment = gp.SimpleAugment(
      transpose_only = (1, 2)   # Only transpose XY
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
    #smooth_augment = SmoothAugment(raw, p=prob_augment)

    # Prepare affinities
    affinities = gp.AddAffinities(
        affinity_neighborhood=neighborhood,
        labels=labels,
        affinities=gt_affs,
        dtype='float32'
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
            labels: "labels"
        },
        output_filename="batch_{iteration}.zarr",
        output_dir=os.path.join(output_dir, "snapshots"),
        every=save_snapshots_every
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
        #smooth_augment +
        affinities + 
        stack +
        snapshot)

    ##########################################################

    # Build the pipeline
    with gp.build(pipeline):
      for step in range(n_training_steps):
         pipeline.request_batch(request)