from ThreeDSegStuff.train import train
from ThreeDSegStuff.loss import weighted_MSELoss
from ThreeDSegStuff.unet import UNet
import torch.optim
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
rotate_aug = eval(unet_config['rotate_aug'])
log_wandb = eval(unet_config['log_wandb'])
wandb_project = unet_config['wandb_project']

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

loss_fct = weighted_MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1e5)

train(
    model = model, 
    loss = loss_fct, 
    optimizer = optimizer,
    input_dir = '/mnt/efs/dl_jrc/student_data/S-EC/diam300/',
    output_dir = '/mnt/efs/dl_jrc/student_data/S-EC/diam300/model_outputs/',
    config_path = config_path,
    n_training_steps = n_training_steps, 
    input_shape = input_shapes,#[1, 64, 64, 64],
    output_shape = output_shapes,#[1, 44, 24, 24],
    batch_size = batch_size,#1,
    prob_augment = prob_augment,#0.3, 
    var_noise = var_noise,#10e-5,
    neighborhood = neighborhood,
    save_snapshots_every = save_snapshots_every, #100,
    save_checkpoints_every = save_checkpoints_every, #100,
    sparse_mask = sparse_mask, #True,
    rotate_aug = rotate_aug, #False,
    log_wandb = log_wandb, #True,
    wandb_project = wandb_project, #"3DSegStuff",
    wandb_run_name = wandb_run_name, #None,
    log_every = log_every,
    unet_config = unet_config,
    boundary = boundary) # contents of the config dict, for saving to WanDB