from ThreeDSegStuff.predict import predict
from ThreeDSegStuff.unet import UNet ### Place holder for loading the new unet
import torch
import os
import json
import torch.nn as nn

# Read in parameters from a config file. 
# Load model parameters
setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))# I copied this line from Vijay's script, idk how it works, will check alter. 

config_path = os.path.join(setup_dir, "config_files/config_unet_XZ.json")
with open(config_path) as f:
    unet_config = json.load(f)

# depth = unet_config["depth"]
in_channels = unet_config["in_channels"]
# out_channels = unet_config["out_channels"]
final_activation = getattr(nn, unet_config["final_activation"])()
num_fmaps = unet_config["num_fmaps"]
fmap_inc_factor = unet_config["fmap_inc_factor"]

padding = unet_config["padding"]
upsample_mode = unet_config["upsample_mode"]
ndim = unet_config["ndim"]

kernel_size = unet_config["kernel_size"]
downsample_factor = unet_config["downsample_factor"]

#initialize the Unet with the correct parameters, from the config file. 
model = UNet(
    depth = depth, 
    in_channels=in_channels, 
    out_channels = out_channels, 
    final_activation=final_activation, 
    num_fmaps = num_fmaps, 
    fmap_inc_factor=fmap_inc_factor,

    padding = padding,
    upsample_mode= upsample_mode,
    ndim = ndim,

    kernel_size = kernel_size, 
    downsample_factor = downsample_factor
)

predict(
    model,
    input_dir = '/mnt/efs/dl_jrc/student_data/S-XZ/train_janelia/omezarr_split/val/norm_1.ome.zarr/0',
    output_dir = '/mnt/efs/dl_jrc/student_data/S-XZ/train_janelia/omezarr_split/val/norm_1.ome.zarr/pred_affs',
    config_file_path = '/home/S-YC/3DSegStuff/scripts/config_files/config_unet.json',
    checkpoint_file_path = '/mnt/efs/dl_jrc/student_data/S-XZ/train_janelia/train_results_r6_rotate/2026-06-17_15-36-31/model_checkpoint_10000',
    neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]], #should be same neighborhood as train
    input_shape = (1, 64, 160, 160), # should be same input_shape as train
    output_shape = (1, 44, 120, 120) # should be same output_shape as train
    )
