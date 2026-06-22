from ThreeDSegStuff.predict_2p5D import predict
from ThreeDSegStuff.unet_2p5D import UNet ### Place holder for loading the new unet
import torch
import os
import json

# Read in parameters from a config file. 
# Load model parameters
setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))# I copied this line from Vijay's script, idk how it works, will check alter. 


config_path = os.path.join(setup_dir, "config_files/config_unet_ms_short_range.json")
with open(config_path) as f:
    unet_config = json.load(f)

in_channels = unet_config["in_channels"]
num_fmaps = unet_config["num_fmaps"]
fmap_inc_factor = unet_config["fmap_inc_factor"]
downsample_factors = eval(
    repr(unet_config["downsample_factors"]).replace("[", "(").replace("]", ")")
)
kernel_size_down = eval(
    repr(unet_config["kernel_size_down"]).replace("[", "(").replace("]", ")")
)
kernel_size_up = eval(
    repr(unet_config["kernel_size_up"]).replace("[", "(").replace("]", ")")
)
activation = unet_config["activation"]
final_activation = unet_config["final_activation"]

ndims = unet_config["outputs"]["3d_affs"]["dims"]
padding = unet_config["padding"]
constant_upsample = unet_config["constant_upsample"]
neighborhood = unet_config["outputs"]["3d_affs"]["neighborhood"]
boundary = unet_config["outputs"]["3d_affs"]["grow_boundary"]

 
input_dir = unet_config['input_dir']
output_dir = unet_config['output_dir']
n_training_steps = unet_config['n_training_steps']

input_shapes = unet_config['input_shape'] #watch out that the global variable is plural but inside the json and fxn the arg is singular
output_shapes = unet_config['output_shape'] #watch out that the global variable is plural but inside the json and fxn the arg is singular
print(output_shapes)
batch_size = unet_config['batch_size']
prob_augment = unet_config['prob_augment']
var_noise = unet_config['var_noise']
save_checkpoints_every = unet_config['save_checkpoints_every']
save_snapshots_every = unet_config['save_snapshots_every']
sparse_mask = eval(unet_config['sparse_mask'])
log_wandb = eval(unet_config['log_wandb'])
wandb_project = unet_config['wandb_project']
lr = float(unet_config['learning_rate'])

wandb_run_name = unet_config['wandb_run_name']
log_every = unet_config['log_every']
stack_infer = eval(unet_config["stack_infer"])
adj_slices = unet_config["adj_slices"]
shape_increase  = unet_config["shape_increase"]

#initialize the Unet with the correct parameters, from the config file. 
model = UNet(
    in_channels=in_channels,
    num_fmaps=num_fmaps,
    fmap_inc_factor=fmap_inc_factor,
    ndims=ndims,
    downsample_factors=downsample_factors,
    kernel_size_down=kernel_size_down,
    kernel_size_up=kernel_size_up,
    activation=activation,
    constant_upsample=constant_upsample,
    padding=padding,
    final_activation=final_activation,
    stack_infer=stack_infer)

predict(
    model,
    input_dir = '/mnt/efs/dl_jrc/student_data/S-MS/annotations_omezarr/test/AR177_section4_3x2__A3.ome_crop02_z0000-0028_y0075-0203_x1511-1639_img.ome.zarr/0',
    output_dir = '/mnt/efs/dl_jrc/student_data/S-MS/annotations_omezarr/test/AR177_section4_3x2__A3.ome_crop02_z0000-0028_y0075-0203_x1511-1639_img.ome.zarr/pred_affs_short',
    config_file_path = '/home/S-MS/3DSegStuff/scripts/config_files/config_unet_ms_short_range.json',
    checkpoint_file_path = '/mnt/efs/dl_jrc/student_data/S-MS/model_outputs/2026-06-17_15-47-21/model_checkpoint_10000',
    neighborhood = neighborhood, #should be same neighborhood as train
    input_shape = input_shapes, # should be same input_shape as train
    output_shape = output_shapes, # should be same output_shape as train
    adj_slices = adj_slices,
    shape_increase = shape_increase
    )
