from ThreeDSegStuff.predict import predict
from ThreeDSegStuff.unet import UNet
import os
import json


def str_to_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).lower() == "true"


def none_if_string_none(x):
    if x is None or x == "None":
        return None
    return x


# Read config file
setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

config_path = os.path.join(setup_dir, "config_files/config_unet_XZ.json")
with open(config_path) as f:
    unet_config = json.load(f)

in_channels = unet_config["in_channels"]
num_fmaps = unet_config["num_fmaps"]
fmap_inc_factor = unet_config["fmap_inc_factor"]
ndims = unet_config["outputs"]["3d_affs"]["dims"]

downsample_factors = [tuple(f) for f in unet_config["downsample_factors"]]
kernel_size_down = unet_config["kernel_size_down"]
kernel_size_up = unet_config["kernel_size_up"]

activation = unet_config["activation"]
final_activation = unet_config["final_activation"]

num_fmaps_out = none_if_string_none(unet_config["num_fmaps_out"])
num_heads = unet_config["num_heads"]

constant_upsample = str_to_bool(unet_config["constant_upsample"])
padding = unet_config["padding"]


# Initialize UNet using your current class signature
model = UNet(
    in_channels=in_channels,
    num_fmaps=num_fmaps,
    fmap_inc_factor=fmap_inc_factor,
    ndims=ndims,
    downsample_factors=downsample_factors,
    kernel_size_down=kernel_size_down,
    kernel_size_up=kernel_size_up,
    activation=activation,
    num_fmaps_out=num_fmaps_out,
    num_heads=num_heads,
    constant_upsample=constant_upsample,
    padding=padding,
    final_activation=final_activation,
)


predict(
    model,
    input_dir="/mnt/efs/dl_jrc/student_data/S-XZ/train_janelia/omezarr_split/val/norm_1.ome.zarr/0",
    output_dir="/mnt/efs/dl_jrc/student_data/S-XZ/train_janelia/omezarr_split/val/norm_1.ome.zarr/pred_affs",

    # use the same config file you loaded above
    config_file_path=config_path,

    checkpoint_file_path="/mnt/efs/dl_jrc/student_data/S-XZ/train_janelia/train_results_r6_rotate/2026-06-17_15-36-31/model_checkpoint_10000",

    neighborhood=unet_config["outputs"]["3d_affs"]["neighborhood"],

    input_shape=tuple(unet_config["input_shape"]),
    output_shape=tuple(unet_config["output_shape"]),
)