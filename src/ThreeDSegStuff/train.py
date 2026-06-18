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
import shutil
import torch
import datetime

logging.basicConfig(level=logging.INFO)

def train(
   model,
   loss,
   optimizer,
   input_dir,
   output_dir,
   config_path = None,
   n_training_steps = 10,
   input_shape = [1, 16, 128, 128],
   output_shape = [1, 14, 124, 124],
   batch_size = 1, 
   prob_augment = 0.3, 
   var_noise = 10e-5,
   neighborhood = [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
   save_snapshots_every = 1,
   save_checkpoints_every = 5,
   sparse_mask = False,
   rotate_aug = True,
   log_wandb = False,
   wandb_project = "3DSegStuff",
   wandb_run_name = None,
   log_every = 1,
   unet_config = None,
   boundary = 1,
   radius = 5,
):
   """

   Inputs:
   - input_dir                   # Directory with omezarr files
   - output_dir                  # Directory to store outputs in (a timestamped run subfolder is created inside it)
   - config_path                 # Path to the config file to copy into the run folder for provenance
   - n_training_steps            # how many batches?
   - channel                     # This is the channel that you want to do your segmentations in
   - input_shape                 # Input patch size: figure out correct based on model architecture
   - output_shape                # Output patch size: igure out correct based on model architecture
   - batch_size                  # Choose batch size appropriately: how many patches in a batch? 
   - prob_augment                # Probability of noise augments (shared by gaussian and poisson)
   - var_noise                   # Variance of gaussian noise
   - neighborhood                # Neighborhoods to compute affinities from 
   - save_snapshots_every        # How often to save snapshots (in training steps)?
   - save_checkpoints_every      # How often to save checkpoints (in training steps)?
   - log_wandb                   # Log loss to Weights & Biases?
   - wandb_project               # W&B project name
   - wandb_run_name              # W&B run name (None -> auto-generated)
   - log_every                   # Log to W&B every N training steps
   - unet_config                 # Dict of UNet architecture params (logged to W&B)
   """

   # Create a fresh timestamped run directory so previous runs are never overwritten
   run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
   run_dir = os.path.join(output_dir, run_name)
   os.makedirs(run_dir, exist_ok=True)
   logging.info(f"Saving outputs to {run_dir}")

   # Copy the config used for this run for reproducibility
   if config_path is not None:
      shutil.copy(config_path, os.path.join(run_dir, os.path.basename(config_path)))
   else:
      logging.warning("No config_path provided: config will not be copied to the run directory.")

   # Optionally set up Weights & Biases logging
   if log_wandb:
      import wandb
      wandb.init(
         project=wandb_project,
         name=wandb_run_name,
         config={
               "n_training_steps": n_training_steps,
               "input_shape": input_shape,
               "output_shape": output_shape,
               "batch_size": batch_size,
               "prob_augment": prob_augment,
               "var_noise": var_noise,
               "neighborhood": neighborhood,
               "sparse_mask": sparse_mask,
               "optimizer": type(optimizer).__name__,
               "lr": optimizer.param_groups[0]["lr"],
               "loss": type(loss).__name__,
               "unet_config": unet_config,
         },
      )

   # Declare array keys
   raw = gp.ArrayKey("RAW")
   labels = gp.ArrayKey("LABELS")
   unlabelled = gp.ArrayKey("MASK")
   gt_affs = gp.ArrayKey("GT_AFFS")
   affs_weights = gp.ArrayKey("AFFS_WEIGHTS")
   gt_affs_mask = gp.ArrayKey("AFFS_MASK")
   pred_affs = gp.ArrayKey("PRED_AFFS")

   # Model training setup
   model.train()
   optimizer = optimizer
   batch_size = batch_size
   
   # if sparse_mask==True:
   #    samples = [
   #       {"raw": os.path.join(f, "0"), 
   #       "labels": os.path.join(f, "labels/labels/0"),
   #       "unlabelled": os.path.join(f, "labels/sparse_label_masks/0"),
   #       } 
   #       for f in sorted(glob.glob(os.path.join(input_dir, "*ome.zarr")))
   #    ]  
   # else:
   #    samples = [
   #       {"raw": os.path.join(f, "0"), 
   #       "labels": os.path.join(f, "labels/labels/0")
   #       } 
   #       for f in sorted(glob.glob(os.path.join(input_dir, "*ome.zarr")))
   #    ]

   if sparse_mask==True and radius>1:
      samples = [
         {"raw": os.path.join(f, "0"), 
         "labels": os.path.join(f, "labels/labels/0"),
         "unlabelled": os.path.join(f, f"labels/sparse_label_masks_dilated_{radius}/0"),
         } 
         for f in sorted(glob.glob(os.path.join(input_dir, "*ome.zarr")))
      ]  
   elif sparse_mask==True and radius==1:
      samples = [
         {"raw": os.path.join(f, "0"), 
         "labels": os.path.join(f, "labels/labels/0"),
         "unlabelled": os.path.join(f, f"labels/sparse_label_masks/0"),
         } 
         for f in sorted(glob.glob(os.path.join(input_dir, "*ome.zarr")))
      ]     
   else:
      samples = [
         {"raw": os.path.join(f, "0"), 
         "labels": os.path.join(f, "labels/labels/0")
         } 
         for f in sorted(glob.glob(os.path.join(input_dir, "*ome.zarr")))
      ]
   
   # assuming same vs
   voxel_size = open_ds(samples[0]["raw"]).voxel_size # World coordinates: voxel coordinate * voxel_size = physical unit
   # voxel_size should be integers

   # Prepare size of requests
   input_size = gp.Coordinate(input_shape[1:]) * voxel_size
   output_size = gp.Coordinate(output_shape[1:]) * voxel_size

   # Request a batch   
   request = gp.BatchRequest()
   request.add(raw, input_size)
   request.add(labels, output_size)
   request.add(gt_affs, output_size)
   request.add(affs_weights, output_size)
   request.add(pred_affs, output_size)
   request.add(unlabelled, output_size)
   # Get samples and declare data source
   source = tuple(
      (
         (
            gp.ArraySource(raw, open_ds(sample["raw"]), True),
            gp.ArraySource(labels, Array(open_ds(sample["labels"])[0], voxel_size=voxel_size), False), # Labels from converter have channel dim? To check
            gp.ArraySource(unlabelled, Array(open_ds(sample["unlabelled"])[0], voxel_size=voxel_size), False)
         )
         + gp.MergeProvider() 
      )
      + gp.Normalize(raw)                # Convert to floats (should already be floats after converting to ome-zarr)
      + gp.Pad(raw, output_size//2)         # Set this appropriately
      + gp.Pad(labels, output_size//2)      # Set this appropriately
      + gp.Pad(unlabelled, output_size//2)      # Set this appropriately
      + gp.RandomLocation(mask=unlabelled, min_masked=0.07)         # Pick a random patch in that source
      for sample in samples) + gp.RandomProvider() # Picks a random source (= ome-zarr) every time

   # Prepare augmentations: tune these to make them likely microscope images for your case!
   simple_augment = gp.SimpleAugment(
   transpose_only = (1, 2)   # Only transpose XY
   )
   
   elastic_augment = gp.DeformAugment(
   control_point_spacing = (10 * voxel_size[-2], 10 * voxel_size[-1]),
   jitter_sigma = (1.5 * voxel_size[-2], 1.5 * voxel_size[-1]),
   rotate = rotate_aug, 
   spatial_dims = 2            # Only deform XY
   )
   intensity_augment = gp.IntensityAugment(
   raw,
   scale_min=0.8,
   scale_max=1.2,
   shift_min=0.2,
   shift_max=0.2) # Are we clipping to 0-1? To check
   gaussian_noise_augment = gp.NoiseAugment(raw, mode='gaussian', p=prob_augment, var=var_noise, clip=True)
   poisson_noise_augment = gp.NoiseAugment(raw, mode='poisson', p=prob_augment, clip=True)
   #smooth_augment = SmoothAugment(raw, p=prob_augment)

   grow_boundary = gp.GrowBoundary(labels, mask=unlabelled, steps=boundary, only_xy=True)

   # Prepare affinities
   affinities = gp.AddAffinities(
      affinity_neighborhood=neighborhood,
      labels=labels,
      unlabelled=unlabelled,          # Training mask, for sparse data
      affinities_mask=gt_affs_mask,   # Training mask but one per channel of the affinities
      affinities=gt_affs,   # New array key
      dtype='float32'
   )

   # Affinities weights are computed here, after masking labels with training mask
   balance_labels = gp.BalanceLabels(labels=gt_affs, scales=affs_weights, mask=gt_affs_mask)  

   train = gp.torch.Train(
      model,
      loss,
      optimizer,
      inputs={
         0: raw
      },
      outputs={
         0: pred_affs
      },
      loss_inputs={
         0: pred_affs,
         1: gt_affs,
         2: affs_weights
      },
      save_every=save_checkpoints_every,
      checkpoint_basename=os.path.join(run_dir, "model"),
      device='cuda'
   )

   # Prepare batch 
   stack = gp.Stack(batch_size)

   # Prepare snapshot = save the requests
   snapshot = gp.Snapshot(
      dataset_names={         # Specify which arrays to save
         raw: "raw",
         gt_affs: "gt_affs",
         pred_affs: "pred_affs",
         affs_weights: "affs_weights",
         labels: "labels",
         unlabelled: "unlabelled"
      },
      output_filename="batch_{iteration}.zarr",
      output_dir=os.path.join(run_dir, "snapshots"),
      every=save_snapshots_every
   )

   ##########################################################

   # Set up pipeline = sequence of nodes in order:
   pipeline = (
      source +
      simple_augment + 
      elastic_augment + 
      intensity_augment + 
      gaussian_noise_augment + 
      poisson_noise_augment +
      #smooth_augment +
      grow_boundary +
      affinities + 
      balance_labels +
      stack +
      train +
      snapshot
      # gp.PrintProfilingStats(every=10)
   )
   ##########################################################

   # Build the pipeline
   with gp.build(pipeline):
      for step in range(n_training_steps):
         batch = pipeline.request_batch(request)
         # gp.torch.Train attaches the loss and global iteration to the batch
         if log_wandb and batch.iteration % log_every == 0:
            wandb.log({"loss": float(batch.loss)}, step=batch.iteration)

   if log_wandb:
      wandb.finish()