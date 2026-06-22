# A script to run Gunpowder pipeline
# 
# Input: zarr file

# Imports

import gunpowder as gp
from funlib.geometry import Roi, Coordinate
from funlib.persistence import open_ds, Array, prepare_ds
from ThreeDSegStuff.invert import InvertIntensities
import os
import logging 
import glob
logging.basicConfig(level=logging.INFO)

import torch
import json


from ThreeDSegStuff.unet_2p5D import UNet ### Place holder for loading the new unet

def predict(
    model,
    input_dir,
    output_dir,
    config_file_path,
    checkpoint_file_path,
    neighborhood,
    input_shape,
    output_shape,
    adj_slices,
    shape_increase
    ):

    """

    Inputs:
    - input_dir                   # Directory with omezarr files
    - output_dir                  # Directory to store outputs in 
    - seg_channel                 # This is the channel that you want to do your segmentations in
    - input_shape                 # Patch size

    """
    print(output_shape)

    # === load net config ===
    with open(os.path.join(config_file_path)) as f:
        logging.info(
            "Reading setup config from %s" % os.path.join(config_file_path)
        )
        net_config = json.load(f)


    # === Load model checkpoint file ===
    checkpoint_file_path = checkpoint_file_path if os.path.exists(checkpoint_file_path) else f"{checkpoint_file_path}.ckpt"
    if not os.path.exists(checkpoint_file_path):
        raise FileNotFoundError(f"Neither {checkpoint_file_path} nor {checkpoint_file_path}.ckpt were found.")

    # === Model setup ===
    model.eval()

    # === Declare array keys ===
    raw = gp.ArrayKey("RAW")
    pred_affs = gp.ArrayKey("PRED_AFFS")

    input_arr = open_ds(input_dir)
    voxel_size = input_arr.voxel_size # World coordinates: voxel coordinate * voxel_size = physical unit
    # voxel_size should be integers

    # Preparing for output array -> makes an empty zarr with desired shape
    print(f'Output size: {output_shape}')
    output_arr = prepare_ds(
       output_dir,
       shape=(len(neighborhood), *input_arr.shape[1:]), #affinity has 3 channels
       voxel_size=input_arr.voxel_size, 
       offset=input_arr.offset,
       axis_names=input_arr.axis_names,
       units=input_arr.units,
       types=input_arr.types,
       chunk_shape=output_shape,
       dtype=input_arr.dtype
      )
    # For some reason this is adding xy dim at the end of output_shape: remove it
    output_shape = output_shape[:-1]
    print(f'Output size: {output_shape}')

    # Prepare request
    print(f'Voxel size: {voxel_size}')
    print(f'Voxel size: {voxel_size[1:]}')
    print(f'Output shape: {output_shape}')
    print(f'Output shape: {output_shape[1:]}')
    print(f'Output shape: {adj_slices, *input_shape[1:]}')
    print(f'Input shape: {input_shape}')
    print(f'Input shape: {input_shape[1:]}')
    print(f'Input shape: {1, *output_shape[1:]}')
    # input_size = gp.Coordinate(adj_slices, *input_shape[:-1]) * voxel_size
    # output_size = gp.Coordinate(*output_shape[:-1]) * voxel_size

    shape_increase = shape_increase
    input_shape = [x + y for x, y in zip(shape_increase, net_config["input_shape"])]
    output_shape = [x + y for x, y in zip(shape_increase, net_config["output_shape"])]

    input_size = gp.Coordinate(adj_slices, *input_shape[1:]) * voxel_size
    output_size = gp.Coordinate(1, *output_shape[1:]) * voxel_size

    # Request a final ROI request   
    request = gp.BatchRequest()
    #request.add(raw, input_arr.roi.shape)
    request.add(pred_affs, output_arr.roi.shape)

    # Prepare a chunk request
    chunk_request = gp.BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(pred_affs, output_size)
    
    # Get samples and declare data source
    source = (
        gp.ArraySource(raw, input_arr, True)
      + gp.Normalize(raw)
      + gp.Pad(raw, None)
    )

    # Prepare batch 
    stack = gp.Stack(1)

    predict = gp.torch.Predict(model, 
                               inputs = {0 : raw}, 
                               outputs = {0: pred_affs}, 
                               checkpoint=checkpoint_file_path,
                               device='cuda' 
                               ) 

    squeeze = gp.Squeeze([pred_affs])
    zarr_write = gp.ZarrWrite(
       {pred_affs : output_dir.split(".zarr")[-1]},
       store=output_dir.split(".zarr")[0] + ".zarr",
    )

    scan = gp.Scan(chunk_request)

    ##########################################################

    # Set up pipeline = sequence of nodes in order:
    pipeline = (
        source + 
        stack +
        predict +
        squeeze +
        zarr_write + 
        scan)

    ##########################################################

    # Build the pipeline
    with gp.build(pipeline):
       pipeline.request_batch(request)