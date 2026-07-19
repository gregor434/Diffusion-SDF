#!/usr/bin/env python3

"""Per-shape reconstruction metrics used by all three training stages."""

import csv
import os

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree as KDTree


def mesh_validity(recon_mesh):
    path = recon_mesh if str(recon_mesh).endswith(".ply") else str(recon_mesh) + ".ply"
    try:
        reconstructed = trimesh.load(path)
        return float(
            isinstance(reconstructed, trimesh.Trimesh)
            and len(reconstructed.vertices) > 0
            and len(reconstructed.faces) > 0
            and np.isfinite(reconstructed.vertices).all()
        )
    except Exception:
        return 0.0


def mesh_metrics(gt_points, recon_mesh, gt_normals=None, fscore_threshold=0.01):
    if torch.is_tensor(gt_points):
        gt_points = gt_points.detach().cpu().numpy()
    gt_points = np.asarray(gt_points).reshape(-1, 3)
    if torch.is_tensor(gt_normals):
        gt_normals = gt_normals.detach().cpu().numpy()
    if gt_normals is not None:
        gt_normals = np.asarray(gt_normals).reshape(-1, 3)

    path = recon_mesh if str(recon_mesh).endswith(".ply") else str(recon_mesh) + ".ply"
    try:
        reconstructed = trimesh.load(path)
        valid = bool(mesh_validity(path))
        if not valid:
            raise ValueError("invalid mesh")
        recon_points, face_indices = trimesh.sample.sample_surface(
            reconstructed, len(gt_points)
        )
        recon_normals = reconstructed.face_normals[face_indices]
    except Exception:
        return {
            "chamfer_distance": float("nan"),
            "f_score": 0.0,
            "normal_consistency": float("nan"),
            "mesh_validity": 0.0,
        }

    recon_tree = KDTree(recon_points)
    gt_to_recon, gt_match = recon_tree.query(gt_points)
    gt_tree = KDTree(gt_points)
    recon_to_gt, recon_match = gt_tree.query(recon_points)
    chamfer = np.square(gt_to_recon).mean() + np.square(recon_to_gt).mean()
    recall = np.mean(gt_to_recon <= fscore_threshold)
    precision = np.mean(recon_to_gt <= fscore_threshold)
    f_score = 2 * precision * recall / max(precision + recall, 1e-12)

    normal_consistency = float("nan")
    if gt_normals is not None:
        first = np.abs(np.sum(gt_normals * recon_normals[gt_match], axis=-1)).mean()
        second = np.abs(np.sum(recon_normals * gt_normals[recon_match], axis=-1)).mean()
        normal_consistency = float(0.5 * (first + second))
    return {
        "chamfer_distance": float(chamfer),
        "f_score": float(f_score),
        "normal_consistency": normal_consistency,
        "mesh_validity": 1.0,
    }


def main(
    gt_pc,
    recon_mesh,
    out_file,
    mesh_name,
    return_value=False,
    return_sampled_pc=False,
    prioritize_cov=False,
    pc_size=None,
):
    metrics = mesh_metrics(gt_pc, recon_mesh)
    chamfer = metrics["chamfer_distance"]
    if return_value:
        return chamfer
    out_file = os.path.join(os.getcwd(), out_file)
    with open(out_file, "a", newline="") as output:
        csv.writer(output).writerow([mesh_name, chamfer])
    if return_sampled_pc:
        reconstructed = trimesh.load(str(recon_mesh) + ".ply")
        if torch.is_tensor(gt_pc):
            count = pc_size or gt_pc.shape[-2]
        else:
            count = pc_size or np.asarray(gt_pc).shape[-2]
        return trimesh.sample.sample_surface(reconstructed, count)[0], chamfer


def calc_cd(gt_pc, recon_pc):
    if torch.is_tensor(gt_pc):
        gt_pc = gt_pc.detach().cpu().numpy()
    gt_pc = np.asarray(gt_pc).reshape(-1, 3)
    recon_pc = np.asarray(recon_pc).reshape(-1, 3)
    gt_to_recon = KDTree(recon_pc).query(gt_pc)[0]
    recon_to_gt = KDTree(gt_pc).query(recon_pc)[0]
    return np.square(gt_to_recon).mean() + np.square(recon_to_gt).mean()
