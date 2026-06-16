#!/usr/bin/env python3
"""Compute the receptive field of a 3D (ZYX) U-Net.

The architecture mirrors the funlib-style U-Net used in this repo
(``src/ThreeDSegStuff/unet.py``): per-level convolution passes with "valid"
convolutions, max-pool downsampling on the encoder, and transposed-conv /
trilinear upsampling on the decoder, followed by a 1x1x1 head.

The receptive field is computed per axis with the symmetric rule: each conv
pass adds sum(k-1), upsampling is treated as a no-op, and each downsampling
doubles (multiplies by the factor) the accumulated field. It does NOT depend on
the input image size; the ``image_dims`` argument is used only to additionally
report the resulting output shape and whether the image is large enough.

Usage:
    python compute_receptive_field.py --config config_files/config_unet.json \\
        --image-dims 48 256 256
"""

import argparse
import json
import math
from functools import reduce


def _as_kernel_list(kernels):
    """Normalize a convolution pass into a list of (z, y, x) kernel sizes."""
    norm = []
    for k in kernels:
        if isinstance(k, int):
            norm.append((k, k, k))
        else:
            norm.append(tuple(k))
    return norm


def receptive_field(num_levels, downsample_factors, kernel_size_down, kernel_size_up):
    """Receptive field (z, y, x), per axis, by the symmetric-upper-bound rule.

    Traced backward from the output to the input, the operations are:

      - conv pass (list of kernels k): rf += sum(k - 1)
      - upsample:                      no-op (ignored)
      - downsample (factor s):         rf *= s

    Because the downsample multiplications are applied *after* (i.e. closer to
    the input than) the decoder conv passes, the decoder contributions get
    scaled up by the encoder factors. This yields an UPPER BOUND on the true
    receptive field (here 64 vs. an exact ~41-44 measured by gradient support
    on the actual network). The upper bound is convenient/conservative for
    sizing input context and padding.
    """
    rf = [1, 1, 1]

    def add_convs(kernels):
        for k in kernels:
            for d in range(3):
                rf[d] += k[d] - 1

    # Decoder conv passes (levels 0 .. num_levels-2); upsamples ignored.
    for level in range(num_levels - 1):
        add_convs(kernel_size_up[level])

    # Bottleneck conv pass.
    add_convs(kernel_size_down[num_levels - 1])

    # Back down the encoder: each downsample multiplies, then its conv pass.
    for level in range(num_levels - 2, -1, -1):
        s = downsample_factors[level]
        for d in range(3):
            rf[d] *= s[d]
        add_convs(kernel_size_down[level])

    # Final 1x1x1 head adds nothing.
    return tuple(rf)


def output_shape(image_dims, num_levels, downsample_factors,
                 kernel_size_down, kernel_size_up):
    """Trace the spatial shape (z, y, x) through the U-Net (valid convs).

    Mirrors ``UNet.rec_forward`` including the ``crop_to_factor`` step used to
    guarantee translation equivariance. Returns (output_shape, warnings).
    """
    warnings = []

    # crop factors for translation equivariance (cumulative product of
    # downsample factors from the bottleneck up), as in unet.py.
    crop_factors = []
    factor_product = None
    for factor in downsample_factors[::-1]:
        if factor_product is None:
            factor_product = list(factor)
        else:
            factor_product = [f * ff for f, ff in zip(factor, factor_product)]
        crop_factors.append(factor_product)
    crop_factors = crop_factors[::-1]

    def conv_crop(kernels):
        return [sum(k[d] - 1 for k in kernels) for d in range(3)]

    def rec(level, shape):
        i = num_levels - level - 1
        # left convolution pass
        f_left = [shape[d] - conv_crop(kernel_size_down[i])[d] for d in range(3)]

        if level == 0:
            return f_left

        # downsample
        s = downsample_factors[i]
        g_in = []
        for d in range(3):
            if f_left[d] % s[d] != 0:
                warnings.append(
                    f"Level {i}: feature size {f_left[d]} not divisible by "
                    f"downsample factor {s[d]} on axis {d} (z/y/x) -> the real "
                    f"U-Net would raise a RuntimeError here.")
            g_in.append(f_left[d] // s[d])

        g_out = rec(level - 1, g_in)

        # upsample
        g_up = [g_out[d] * s[d] for d in range(3)]

        # crop_to_factor (valid padding)
        factor = crop_factors[i]
        conv_crop_up = conv_crop(kernel_size_up[i])
        g_cropped = []
        for d in range(3):
            n = math.floor((g_up[d] - conv_crop_up[d]) / factor[d])
            g_cropped.append(n * factor[d] + conv_crop_up[d])

        # concat with cropped f_left (spatial size = g_cropped), then conv pass
        f_out = [g_cropped[d] - conv_crop_up[d] for d in range(3)]
        return f_out

    out = rec(num_levels - 1, list(image_dims))
    # final 1x1x1 head: no spatial change
    if any(o <= 0 for o in out):
        warnings.append(
            f"Computed output shape {tuple(out)} has a non-positive dimension: "
            f"the input image {tuple(image_dims)} is too small for this U-Net.")
    return tuple(out), warnings


def main():
    parser = argparse.ArgumentParser(
        description="Compute the receptive field of a 3D (ZYX) U-Net.")
    parser.add_argument("--config", required=True,
                        help="Path to the U-Net config JSON.")
    parser.add_argument("--image-dims", type=int, nargs=3, metavar=("Z", "Y", "X"),
                        help="Input image size as Z Y X (optional; used to "
                             "report output shape and feasibility).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    downsample_factors = [tuple(d) for d in cfg["downsample_factors"]]
    kernel_size_down = [_as_kernel_list(p) for p in cfg["kernel_size_down"]]
    kernel_size_up = [_as_kernel_list(p) for p in cfg["kernel_size_up"]]

    num_levels = len(downsample_factors) + 1

    # sanity checks against the architecture in unet.py
    assert len(kernel_size_down) == num_levels, (
        f"kernel_size_down must have {num_levels} entries (one per level), "
        f"got {len(kernel_size_down)}.")
    assert len(kernel_size_up) == num_levels - 1, (
        f"kernel_size_up must have {num_levels - 1} entries, "
        f"got {len(kernel_size_up)}.")

    rf = receptive_field(num_levels, downsample_factors,
                         kernel_size_down, kernel_size_up)

    total_downsample = [reduce(lambda a, b: a * b, (s[d] for s in downsample_factors), 1)
                        for d in range(3)]

    print("=" * 60)
    print("U-Net receptive field")
    print("=" * 60)
    print(f"config              : {args.config}")
    print(f"levels              : {num_levels} "
          f"({num_levels - 1} down/up-sampling steps)")
    print(f"downsample factors  : {downsample_factors}")
    print(f"total downsampling  : (z, y, x) = {tuple(total_downsample)}")
    print(f"in_channels         : {cfg.get('in_channels')}")
    print("-" * 60)
    print(f"RECEPTIVE FIELD     : (z, y, x) = {rf}  voxels")
    print("-" * 60)

    if args.image_dims:
        out_shape, warnings = output_shape(
            args.image_dims, num_levels, downsample_factors,
            kernel_size_down, kernel_size_up)
        print(f"input image (z,y,x) : {tuple(args.image_dims)}")
        print(f"output shape (z,y,x): {out_shape}")
        for d, axis in enumerate("zyx"):
            if args.image_dims[d] < rf[d]:
                warnings.append(
                    f"Input {axis}={args.image_dims[d]} is smaller than the "
                    f"receptive field {rf[d]} on that axis.")
        if warnings:
            print("-" * 60)
            print("WARNINGS:")
            for w in warnings:
                print(f"  - {w}")
    print("=" * 60)


if __name__ == "__main__":
    main()
