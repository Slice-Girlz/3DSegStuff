from ThreeDSegStuff.predict import predict
from ThreeDSegStuff.unet_new import UNet ### Place holder for loading the new unet
import torch
import os
import json

# Read in parameters from a config file. 
# Load model parameters
setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))# I copied this line from Vijay's script, idk how it works, will check alter. 

config_path = os.path.join(setup_dir, "config_files/config_unet.json")
with open(config_path) as f:
    unet_config = json.load(f)

depth = unet_config["depth"]
in_channels = unet_config["in_channels"]
out_channels = unet_config["out_channels"]
final_activation = eval(unet_config["final_activation"])
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
    input_dir = '/mnt/efs/dl_jrc/student_data/S-MS/123.ome.zarr/0',
    output_dir = '/mnt/efs/dl_jrc/student_data/S-MS/model_pred/',
    config_path = config_path,
    checkpoint_file_path = '/mnt/efs/dl_jrc/student_data/S-YC/model_outputs/2026-06-16_14-14-09/model_checkpoint_9900',
    neighborhood = [[1, 0, 0], [0, 1, 0], [0, 0, 1]], #should be same neighborhood as train
    input_shape = [1, 16, 128, 128], # should be same input_shape as train
    output_shape = [1, 14, 124, 124] # should be same output_shape as train
    )
