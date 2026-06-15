from ThreeDSegStuff.train import train
from ThreeDSegStuff.loss import weighted_MSEloss
from ThreeDSegStuff.unet import UNet
import torch.optim
import torch
import os
import json

# Read in parameters from a config file. 
# Load model parameters
setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))# I copied this line from Vijay's script, idk how it works, will check alter. 

with open(os.path.join(setup_dir, "config_files/config_unet.json")) as f:
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

loss_fct = weighted_MSEloss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train(
    model = model, 
    loss = loss_fct, 
    optimizer = optimizer,
    input_dir = '/mnt/efs/dl_jrc/student_data/S-MS/annotations_omezarr/',
    output_dir = '.',
    n_training_steps = 10,
    channel = 1,
    input_shape = [1, 16, 128, 128],
    output_shape = [1, 14, 124, 124],
    batch_size = 1, 
    prob_augment = 0.3, 
    var_noise = 10e-5,
    neighborhood = [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    save_snapshots_every = 1, 
    sparse_mask = False)

