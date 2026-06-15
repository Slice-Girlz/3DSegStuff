# A script to run Gunpowder pipeline
# 
# Input: zarr file

# Imports
import gunpowder as gp
from funlib.geometry import Roi, Coordinate
from funlib.persistence import open_ds, Array, prepare_ds
#from smooth_augment import SmoothAugment
from ThreeDSegStuff.invert import InvertIntensities
import os
import logging 
import glob

logging.basicConfig(level=logging.INFO)

def predict(
    input_array,
    output_array,
    channel = 1,
    input_shape = [1, 16, 128, 128],
    output_shape = [1, 14, 124, 124]
):
    """

    Inputs:
    - input_dir                   # Directory with omezarr files
    - output_dir                  # Directory to store outputs in 
    - channel                     # This is the channel that you want to do your segmentations in
    - input_shape                 # Patch size

    """

    # Declare array keys
    raw = gp.ArrayKey("RAW")
    pred_affs = gp.ArrayKey("PRED_AFFS")

    # Model training setup
    # model = ...
    # model.eval()
    

    input_arr = open_ds(input_array)
    output_arr = prepare_ds(
       output_array,
       shape=input_arr.shape, 
       voxel_size=input_arr.voxel_size, 
       offset=input_arr.offset,
       axis_names=input_arr.axis_names,
       units=input_arr.units,
       types=input_arr.types,
       chunk_shape=output_shape,
       dtype=input_arr.dtype
      )
    voxel_size = input_arr.voxel_size

    # Prepare request
    input_size = gp.Coordinate(input_shape[1:]) * voxel_size
    output_size = gp.Coordinate(output_shape[1:]) * voxel_size

    # Request a final ROI request   
    request = gp.BatchRequest()
    #request.add(raw, input_arr.roi.shape)
    request.add(pred_affs, output_arr.roi.shape)

    # Prepare a chunk request
    chunk_request = gp.BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(pred_affs, output_size)
    
    # Get samples and declare data source
    source = (gp.ArraySource(raw, input_arr, True)
      + gp.Normalize(raw)
      + gp.Pad(raw, output_size))

    # Prepare batch 
    #stack = gp.Stack(1)

    predict = InvertIntensities(raw, pred_affs)

    zarr_write = gp.ZarrWrite(
       {pred_affs : output_array.split(".zarr")[-1]},
       store=output_array.split(".zarr")[0] + ".zarr",
    )

    scan = gp.Scan(chunk_request)

    ##########################################################

    # Set up pipeline = sequence of nodes in order:
    pipeline = (
        source + 
        #stack +
        predict +
        zarr_write + 
        scan)

    ##########################################################

    # Build the pipeline
    with gp.build(pipeline):
       pipeline.request_batch(request)