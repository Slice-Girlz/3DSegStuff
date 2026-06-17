from ThreeDSegStuff.predict import predict
from ThreeDSegStuff.unet import UNet ### Place holder for loading the new unet
import torch
import os
import json

# Read in parameters from a config file. 
# Load model parameters
setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))# I copied this line from Vijay's script, idk how it works, will check alter. 


config_path = os.path.join(setup_dir, "config_files/config_unet.json")
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
batch_size = unet_config['batch_size']
prob_augment = unet_config['prob_augment']
var_noise = unet_config['var_noise']
save_checkpoints_every = unet_config['save_checkpoints_every']
save_snapshots_every = unet_config['save_snapshots_every']
sparse_mask = eval(unet_config['sparse_mask'])
radius = unet_config['radius']
rotate_aug = eval(unet_config['rotate_aug'])
log_wandb = eval(unet_config['log_wandb'])
wandb_project = unet_config['wandb_project']
lr = float(unet_config['learning_rate'])

# if wandb_project is not None:
#     wandb_project == eval(wandb_project)
# else: 
#     pass
#currently expecting wandb_project to pass in None. 
# if you pass in a string instead, it will just pass so it stays a string. 

wandb_run_name = unet_config['wandb_run_name']
log_every = unet_config['log_every']


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
    final_activation=final_activation
)

predict(
    model,
    input_dir = "/mnt/efs/dl_jrc/student_data/S-JM/train/new_ome_zarr_SDT/2026_05_18_JMS_36A_TileScan_1_A1_R_3_Merged_Colony_0002_rf_preprocessed_uint8.ome.zarr/0",
    output_dir = '/mnt/efs/dl_jrc/student_data/S-JM/train/processed_zarr/test1.ome.zarr/pred_affs',
    config_file_path = 'scripts/config_files/config_unet.json',
    checkpoint_file_path = '/mnt/efs/dl_jrc/student_data/S-YC/model_outputs/2026-06-16_20-34-18/model_checkpoint_100',
    neighborhood = [[1, 0, 0], [0, 1, 0], [0, 0, 1]], #should be same neighborhood as train
    input_shape = (1, 16, 64, 64), # should be same input_shape as train
    output_shape = (1, 16, 64, 64) # should be same output_shape as train
    )
