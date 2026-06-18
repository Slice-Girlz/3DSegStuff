from ThreeDSegStuff.predict import predict
from ThreeDSegStuff.unet import UNet

import os
import json
import shutil


def str_to_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).lower() == "true"


def none_if_string_none(x):
    if x is None or x == "None":
        return None
    return x


# -------------------------
# Paths
# -------------------------
setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

config_path = os.path.join(setup_dir, "config_files/config_unet_XZ.json")

base_input_dir = "/mnt/efs/dl_jrc/student_data/S-XZ/gfp_100_omezarr"

checkpoint_file_path = (
    "/mnt/efs/dl_jrc/student_data/S-XZ/train_janelia/"
    "train_results_r10_rotate/2026-06-17_18-19-56/"
    "model_checkpoint_16000"
)

output_name = "pred_affs"

# Set True if you want to overwrite old predictions
overwrite = True


# -------------------------
# Read config file
# -------------------------
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


# -------------------------
# Initialize UNet once
# -------------------------
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


# -------------------------
# Batch prediction
# -------------------------
for i in range(100):
    sample_name = f"{i:03d}.ome.zarr"
    sample_path = os.path.join(base_input_dir, sample_name)

    input_dir = os.path.join(sample_path, "0")
    output_dir = os.path.join(sample_path, output_name)

    if not os.path.exists(sample_path):
        print(f"[SKIP] Sample does not exist: {sample_path}")
        continue

    if not os.path.exists(input_dir):
        print(f"[SKIP] Input dataset does not exist: {input_dir}")
        continue

    if os.path.exists(output_dir):
        if overwrite:
            print(f"[REMOVE] Old prediction: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            print(f"[SKIP] Prediction already exists: {output_dir}")
            continue

    print("=" * 80)
    print(f"[PREDICT] {sample_name}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    predict(
        model,
        input_dir=input_dir,
        output_dir=output_dir,

        config_file_path=config_path,

        checkpoint_file_path=checkpoint_file_path,

        neighborhood=unet_config["outputs"]["3d_affs"]["neighborhood"],

        input_shape=tuple(unet_config["input_shape"]),
        output_shape=tuple(unet_config["output_shape"]),
    )

    print(f"[DONE] {sample_name}")


print("All predictions finished.")