#!/usr/bin/env python3
"""
Simple IoU evaluation for instance labels stored in OME-Zarr / Zarr.

Compares:
    label       = ground-truth instance labels
    pred_label  = predicted instance labels

Assumptions:
    0 = background
    1, 2, 3, ... = object / instance IDs
    input arrays are stored inside .ome.zarr or .zarr
    label and pred_label have the same shape

Metrics:
    - foreground IoU
    - object-level matching by IoU threshold
    - mean matched IoU
    - variation of information (VOI): voi_split, voi_merge, voi

Usage:
    python evaluation.py --label Cell6.ome.zarr/labels/labels/0 --pred-label prediction.ome.zarr/pred_labels --out --out ./cell6_t1000.json

    Required:
        --label        ground-truth labels inside the .ome.zarr / .zarr store (you define this)
        --pred-label   predicted labels inside the .ome.zarr / .zarr store (you define this)
        --out          path where the results .json will be written (you define this)
    Optional:
        --iou-threshold   IoU cutoff for object matching (default: 0.5)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import zarr


def load_zarr_array(path):
    """
    Load an array from a .ome.zarr or .zarr dataset path.

    Example paths:
        Cell1.ome.zarr/labels/labels/0
        prediction.ome.zarr/pred_labels/0
    """

    path = str(path)

    if ".ome.zarr" in path:
        marker = ".ome.zarr"
    elif ".zarr" in path:
        marker = ".zarr"
    else:
        raise ValueError(
            "Input must be a .ome.zarr or .zarr dataset path, for example: Cell1.ome.zarr/labels"
        )

    store_end = path.index(marker) + len(marker)
    store_path = path[:store_end]
    dataset_path = path[store_end:].strip("/")

    root = zarr.open(store_path, mode="r")

    if dataset_path == "":
        return np.asarray(root)

    return np.asarray(root[dataset_path])


def compute_iou(label, pred_label):
    """
    Compute foreground IoU and pairwise instance IoU.

    Parameters
    ----------
    label : np.ndarray
        Ground-truth instance labels. 0 is background.
    pred_label : np.ndarray
        Predicted instance labels. 0 is background.

    Returns
    -------
    dict
        IoU results.
    """

    label = np.asarray(label)
    pred_label = np.asarray(pred_label)

    if label.shape != pred_label.shape:
        raise ValueError(
            f"label and pred_label must have the same shape. Got {label.shape} and {pred_label.shape}."
        )

    gt_foreground = label > 0
    pred_foreground = pred_label > 0

    foreground_intersection = np.logical_and(gt_foreground, pred_foreground).sum()
    foreground_union = np.logical_or(gt_foreground, pred_foreground).sum()

    foreground_iou = (
        foreground_intersection / foreground_union
        if foreground_union > 0
        else 0.0
    )

    gt_ids = np.unique(label)
    pred_ids = np.unique(pred_label)

    gt_ids = gt_ids[gt_ids != 0]
    pred_ids = pred_ids[pred_ids != 0]

    iou_matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=float)

    for i, gt_id in enumerate(gt_ids):
        gt_mask = label == gt_id

        for j, pred_id in enumerate(pred_ids):
            pred_mask = pred_label == pred_id

            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()

            if union > 0:
                iou_matrix[i, j] = intersection / union

    return {
        "foreground_iou": float(foreground_iou),
        "gt_ids": [int(x) for x in gt_ids],
        "pred_ids": [int(x) for x in pred_ids],
        "iou_matrix": iou_matrix,
    }


def compute_voi(label, pred_label):
    """
    Variation of Information (VOI).

    Python port of the VOI part of funkelab/funlib.evaluate rand_voi.hpp.
    Matching that implementation, only voxels where the ground-truth label is
    non-zero are counted; prediction background is kept as label 0.

    Returns
    -------
    dict with:
        voi_split  = H(pred | gt)   penalizes splitting one GT object into many
        voi_merge  = H(gt | pred)   penalizes merging several GT objects into one
        voi        = voi_split + voi_merge   (lower is better, 0 = perfect)
    """

    a = np.asarray(label).ravel()
    b = np.asarray(pred_label).ravel()

    # only count ground-truth foreground voxels (C++ `if (a)` guard)
    mask = a != 0
    a = a[mask]
    b = b[mask]

    total = a.size

    if total == 0:
        return {"voi_split": 0.0, "voi_merge": 0.0, "voi": 0.0}

    # factorize labels so the joint histogram is fast and overflow-safe
    _, a_idx = np.unique(a, return_inverse=True)
    _, b_idx = np.unique(b, return_inverse=True)

    a_counts = np.bincount(a_idx)
    b_counts = np.bincount(b_idx)

    joint_key = a_idx.astype(np.int64) * b_counts.size + b_idx
    ab_counts = np.bincount(joint_key)
    ab_counts = ab_counts[ab_counts > 0]

    p_a = a_counts / total
    p_b = b_counts / total
    p_ab = ab_counts / total

    H_a = -np.sum(p_a * np.log2(p_a))
    H_b = -np.sum(p_b * np.log2(p_b))
    H_ab = -np.sum(p_ab * np.log2(p_ab))

    voi_split = float(H_ab - H_a)  # H(pred | gt)
    voi_merge = float(H_ab - H_b)  # H(gt | pred)

    return {
        "voi_split": voi_split,
        "voi_merge": voi_merge,
        "voi": voi_split + voi_merge,
    }


def evaluate(label, pred_label, iou_threshold=0.5):
    """
    Evaluate predicted labels against ground-truth labels using IoU.

    Matching is one-to-one: one predicted object can match only one ground-truth object. 
    Matches are chosen greedily from highest IoU to lowest IoU.
    """

    iou_results = compute_iou(label, pred_label)
    voi_results = compute_voi(label, pred_label)

    gt_ids = iou_results["gt_ids"]
    pred_ids = iou_results["pred_ids"]
    iou_matrix = iou_results["iou_matrix"]

    candidate_matches = []

    for i, gt_id in enumerate(gt_ids):
        for j, pred_id in enumerate(pred_ids):
            iou = iou_matrix[i, j]

            if iou >= iou_threshold:
                candidate_matches.append((iou, gt_id, pred_id))

    candidate_matches.sort(reverse=True, key=lambda x: x[0])

    matched_gt = set()
    matched_pred = set()
    matches = []

    for iou, gt_id, pred_id in candidate_matches:
        if gt_id in matched_gt:
            continue

        if pred_id in matched_pred:
            continue

        matched_gt.add(gt_id)
        matched_pred.add(pred_id)

        matches.append(
            {
                "gt_id": int(gt_id),
                "pred_id": int(pred_id),
                "iou": float(iou),
            }
        )

    num_matched = len(matches)

    mean_matched_iou = (
        float(np.mean([m["iou"] for m in matches]))
        if num_matched > 0
        else 0.0
    )

    return {
        "iou_threshold": float(iou_threshold),
        "foreground_iou": iou_results["foreground_iou"],
        "num_gt_objects": int(len(gt_ids)),
        "num_pred_objects": int(len(pred_ids)),
        "num_matched": int(num_matched),
        "mean_matched_iou": float(mean_matched_iou),
        "voi_split": voi_results["voi_split"],
        "voi_merge": voi_results["voi_merge"],
        "voi": voi_results["voi"],
        "matches": matches,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predicted instance labels against ground-truth labels using IoU."
    )

    parser.add_argument(
        "--label",
        required=True,
        help="Ground-truth label dataset inside .ome.zarr/.zarr, e.g. Cell1.ome.zarr/labels/labels/0",
    )

    parser.add_argument(
        "--pred-label",
        required=True,
        help="Predicted label dataset inside .ome.zarr/.zarr, e.g. prediction.ome.zarr/pred_labels",
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for object matching. Default: 0.5",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Path where the results JSON will be written, e.g. results/cell6_t179.json",
    )

    args = parser.parse_args()

    label = load_zarr_array(args.label)
    pred_label = load_zarr_array(args.pred_label)

    results = evaluate(
        label=label,
        pred_label=pred_label,
        iou_threshold=args.iou_threshold,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()