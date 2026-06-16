from ThreeDSegStuff.train import train
from ThreeDSegStuff.loss import weighted_MSELoss
from ThreeDSegStuff.unet_old import UNet
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
activation = eval(unet_config["activation"])

outputs = unet_config["outputs"]
padding = unet_config["padding"]
constant_upsample = unet_config["constant_upsample"]
boundary = unet_config["grow_boundary"]
neighborhood = unet_config["neighborhood"]


#initialize the Unet with the correct parameters, from the config file. 
model = UNet(
    in_channels=in_channels,
    num_fmaps=num_fmaps,
    fmap_inc_factor=fmap_inc_factor,
    outputs=outputs,
    downsample_factors=downsample_factors,
    kernel_size_down=kernel_size_down,
    kernel_size_up=kernel_size_up,
    activation=activation,
    constant_upsample=constant_upsample,
    padding=padding
)

loss_fct = weighted_MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train(
    model = model, 
    loss = loss_fct, 
    optimizer = optimizer,
    input_dir = '/mnt/efs/dl_jrc/student_data/S-MS/annotations_omezarr/train_val/',
    output_dir = '/mnt/efs/dl_jrc/student_data/S-MS/model_outputs/',
    config_path = config_path,
    n_training_steps = 10, 
    input_shape = [1, 16, 64, 64],
    output_shape = [1, 16, 64, 64],
    batch_size = 1,
    prob_augment = 0.3, 
    var_noise = 10e-5,
    neighborhood = neighborhood,
    save_snapshots_every = 1,
    save_checkpoints_every = 1,
    sparse_mask = True,
    rotate_aug = False,
    log_wandb = True,
    wandb_project = "3DSegStuff",
    wandb_run_name = None,
    log_every = 1,
    unet_config = unet_config,
    boundary = boundary) # contents of the config dict, for saving to WanDB