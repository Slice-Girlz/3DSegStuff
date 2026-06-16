# A script to run a Gunpowder validation loop
#
# Computes the (no-augmentation, no-gradient) weighted-MSE loss on a held-out
# set of ome.zarr volumes. Mirrors the training pipeline in train.py but:
#   - drops every augmentation
#   - swaps gp.torch.Train (forward + loss + backward) for gp.torch.Predict
#     (forward only, under torch.no_grad)
#   - computes the same weighted_MSELoss manually from the requested batch
#
# Returns the mean validation loss over n_val_batches random patches. Writes
# nothing to disk (whole-volume scan-to-zarr lives in predict.py).

# Imports
import gunpowder as gp
from funlib.persistence import open_ds, Array
import torch
import logging

logging.basicConfig(level=logging.INFO)


def validate(
    model,
    loss,
    val_samples,                                # list of {"raw","labels","unlabelled"} dicts
    voxel_size,                                 # passed in from train() so geometry matches exactly
    input_shape = [1, 16, 64, 64],
    output_shape = [1, 16, 64, 64],
    neighborhood = [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    batch_size = 5,
    sparse_mask = True,
    n_val_batches = 10,
    grow_boundary = True,
    device = 'cuda',
):
    """
    Run a patch-based validation loop and return the mean weighted-MSE loss.

    Inputs:
    - model                       # Same model object trained by train() (weights are shared live)
    - loss                        # Same weighted_MSELoss(pred_affs, gt_affs, affs_weights) used in training
    - val_samples                 # Held-out samples (same dict layout train() builds)
    - voxel_size                  # World coordinates, passed in so val/train request geometry is identical
    - input_shape                 # Input patch size (must match training)
    - output_shape                # Output patch size (must match training)
    - neighborhood                # Neighborhoods to compute affinities from (must match training)
    - batch_size                  # Patches per request_batch
    - sparse_mask                 # Whether labels come with a sparse training mask
    - n_val_batches               # Number of random patches to average the loss over
    - grow_boundary               # Keep the GrowBoundary target transform (deterministic, recommended)
    - device                      # Device for the forward pass
    """

    if not sparse_mask:
        # The pipeline below requests `unlabelled`; supporting the non-sparse
        # case would need a separate source/request. Fail loudly for now.
        raise NotImplementedError("validate() currently assumes sparse_mask=True")

    if not val_samples:
        logging.warning("validate() called with no val_samples; returning nan.")
        return float("nan")

    # Declare array keys
    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    unlabelled = gp.ArrayKey("MASK")
    gt_affs = gp.ArrayKey("GT_AFFS")
    affs_weights = gp.ArrayKey("AFFS_WEIGHTS")
    gt_affs_mask = gp.ArrayKey("AFFS_MASK")
    pred_affs = gp.ArrayKey("PRED_AFFS")

    # Prepare size of requests
    input_size = gp.Coordinate(input_shape[1:]) * voxel_size
    output_size = gp.Coordinate(output_shape[1:]) * voxel_size

    # Request a batch (same keys/sizes as training)
    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(gt_affs, output_size)
    request.add(affs_weights, output_size)
    request.add(pred_affs, output_size)
    request.add(unlabelled, output_size)

    # Get samples and declare data source (mirror train.py, NO augmentations)
    source = tuple(
        (
            (
                gp.ArraySource(raw, open_ds(sample["raw"]), True),
                gp.ArraySource(labels, Array(open_ds(sample["labels"])[0], voxel_size=voxel_size), False),
                gp.ArraySource(unlabelled, Array(open_ds(sample["unlabelled"])[0], voxel_size=voxel_size), False)
            )
            + gp.MergeProvider()
        )
        + gp.Normalize(raw)
        + gp.Pad(raw, output_size//2)
        + gp.Pad(labels, output_size//2)
        + gp.Pad(unlabelled, output_size//2)
        + gp.RandomLocation(mask=unlabelled)
        for sample in val_samples) + gp.RandomProvider()

    grow_boundary_node = gp.GrowBoundary(labels, mask=unlabelled)

    # Prepare affinities (same as training)
    affinities = gp.AddAffinities(
        affinity_neighborhood=neighborhood,
        labels=labels,
        unlabelled=unlabelled,
        affinities_mask=gt_affs_mask,
        affinities=gt_affs,
        dtype='float32'
    )

    # Affinities weights, computed after masking labels with training mask
    balance_labels = gp.BalanceLabels(labels=gt_affs, scales=affs_weights, mask=gt_affs_mask)

    # Prepare batch
    stack = gp.Stack(batch_size)

    ##########################################################

    # Run validation: eval mode + no grad, restoring the previous mode robustly.
    # gp.torch.Predict will not flip the model into eval mode, and the training
    # Train node will not flip it back, so we manage it here. We switch to eval
    # BEFORE constructing the Predict node so it does not warn about predicting
    # in train mode.
    was_training = model.training
    model.eval()
    total = 0.0
    try:
        # Forward-pass node: the inference twin of gp.torch.Train, run under
        # torch.no_grad() with the current (shared) model weights.
        predict = gp.torch.Predict(
            model,
            inputs={
                0: raw
            },
            outputs={
                0: pred_affs
            },
            device=device
        )

        # Set up pipeline = sequence of nodes in order:
        pipeline = source
        if grow_boundary:
            pipeline = pipeline + grow_boundary_node
        pipeline = (
            pipeline +
            affinities +
            balance_labels +
            stack +
            predict
        )

        with gp.build(pipeline):
            for _ in range(n_val_batches):
                batch = pipeline.request_batch(request)
                pred = torch.as_tensor(batch[pred_affs].data)
                gt = torch.as_tensor(batch[gt_affs].data)
                w = torch.as_tensor(batch[affs_weights].data)
                with torch.no_grad():
                    total += float(loss(pred, gt, w))
    finally:
        model.train(was_training)

    return total / max(n_val_batches, 1)
