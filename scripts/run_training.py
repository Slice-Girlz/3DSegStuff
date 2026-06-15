from ThreeDSegStuff.train import train
from ThreeDSegStuff.loss import loss
from ThreeDSegStuff.unet import UNet


import os
import json

loss  = ...
optimizer = ...



#TO DO: 

#read in parameters from a config file. 

### replace the following 
# load model parameters
setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))# I copied this line from Vijay's script, idk how it works, will check alter. 

print(str(setup_dir))



with open(os.path.join(setup_dir, "net_config.json")) as f:
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

#these parameters were more elaborate in Vijay's script
kernel_size = unet_config["kernel_size"]
downsample_factor = unet_config["downsample_factor"]
# they were split into kernel size up and down, 
# and also written as 
# downsample_factors = eval(
#     repr(net_config["downsample_factors"]).replace("[", "(").replace("]", ")")
# )


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



train(
    model = model, 
    # loss, 
    # optimizer,
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