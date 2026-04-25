#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI点云裁剪 + 主干优先去噪改标 + 表型输入文件准备工具

输入：ASCII PLY（需包含 x/y/z/object_value，建议包含rgb）
输出：
  1) point_cloud_main_clusters_merged.ply（ASCII，保留object_value）
  2) hybrid_filtered_final.ply（binary，供现有表型脚本高度解析）
  3) stem_only_cleaned.ply（可选查看主干效果）
  4) filtering_report.json（质量门控与诊断）
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d


INT_TYPES = {"char", "uchar", "short", "ushort", "int", "uint"}
FLOAT_TYPES = {"float", "double"}


@dataclass
class PlyData:
    properties: List[Tuple[str, str]]
    data: np.ndarray  # shape: [N, P], float64
    prop_index: Dict[str, int]


def parse_int_set(text: str) -> set:
    if not text:
        return set()
    return {int(v.strip()) for v in text.split(",") if v.strip()}


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b not in (0, 0.0) else 0.0


def read_ascii_ply(path: str) -> PlyData:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY头部读取失败")
            line = line.strip()
            header.append(line)
            if line == "end_header":
                break

        if header[0] != "ply":
            raise ValueError("不是合法PLY文件")

        fmt = [h for h in header if h.startswith("format ")]
        if not fmt or "ascii" not in fmt[0]:
            raise ValueError("当前仅支持ASCII PLY输入")

        vertex_count = 0
        properties: List[Tuple[str, str]] = []
        in_vertex = False
        for h in header:
            if h.startswith("element "):
                parts = h.split()
                in_vertex = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif in_vertex and h.startswith("property "):
                parts = h.split()
                if len(parts) >= 3:
                    properties.append((parts[2], parts[1]))

        if vertex_count <= 0 or not properties:
            raise ValueError("PLY顶点或属性定义无效")

        rows = []
        pnum = len(properties)
        for _ in range(vertex_count):
            line = f.readline()
            if not line:
                break
            vals = line.strip().split()
            if len(vals) < pnum:
                continue
            try:
                rows.append([float(v) for v in vals[:pnum]])
            except ValueError:
                continue

    data = np.asarray(rows, dtype=np.float64)
    if data.size == 0:
        raise ValueError("点云为空或读取失败")

    prop_index = {n: i for i, (n, _) in enumerate(properties)}
    required = {"x", "y", "z", "object_value"}
    miss = [k for k in required if k not in prop_index]
    if miss:
        raise ValueError(f"缺少必要属性: {miss}")

    return PlyData(properties=properties, data=data, prop_index=prop_index)


def write_ascii_ply(path: str, ply: PlyData, mask: np.ndarray, labels: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    selected = ply.data[mask].copy()
    selected[:, ply.prop_index["object_value"]] = labels[mask]

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {selected.shape[0]}\n")
        for name, ptype in ply.properties:
            f.write(f"property {ptype} {name}\n")
        f.write("end_header\n")

        for row in selected:
            out = []
            for (name, ptype), v in zip(ply.properties, row):
                if ptype in INT_TYPES:
                    out.append(str(int(round(v))))
                elif ptype in FLOAT_TYPES:
                    out.append(f"{float(v):.8f}")
                else:
                    out.append(str(v))
            f.write(" ".join(out) + "\n")


def write_binary_hybrid(path: str, ply: PlyData, mask: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    xyz = np.stack(
        [
            ply.data[mask, ply.prop_index["x"]],
            ply.data[mask, ply.prop_index["y"]],
            ply.data[mask, ply.prop_index["z"]],
        ],
        axis=1,
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))

    if all(c in ply.prop_index for c in ("red", "green", "blue")):
        rgb = np.stack(
            [
                ply.data[mask, ply.prop_index["red"]],
                ply.data[mask, ply.prop_index["green"]],
                ply.data[mask, ply.prop_index["blue"]],
            ],
            axis=1,
        )
        pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb / 255.0, 0.0, 1.0))

    o3d.io.write_point_cloud(path, pcd, write_ascii=False, compressed=False)


def run_dbscan_largest(points: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if labels.size == 0 or labels.max() < 0:
        return np.ones(points.shape[0], dtype=bool)
    valid = labels >= 0
    uniq, cnt = np.unique(labels[valid], return_counts=True)
    keep = uniq[np.argmax(cnt)]
    return labels == keep


def run_dbscan_labels(points: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False), dtype=np.int32)


def pca_main_axis(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 3:
        return np.array([0.0, 1.0, 0.0])
    c = points.mean(axis=0)
    x = points - c
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    axis = vt[0]
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return np.array([0.0, 1.0, 0.0])
    return axis / n


def point_line_dist(points: np.ndarray, center: np.ndarray, axis: np.ndarray) -> np.ndarray:
    d = points - center
    proj = (d @ axis)[:, None] * axis[None, :]
    perp = d - proj
    return np.linalg.norm(perp, axis=1)


def knn_indices(points: np.ndarray, idx: int, k: int) -> np.ndarray:
    d2 = np.sum((points - points[idx]) ** 2, axis=1)
    order = np.argsort(d2)
    order = order[order != idx]
    return order[:k]


def boundary_score(points: np.ndarray, labels: np.ndarray, k: int, sample_max: int = 3000) -> Tuple[int, int]:
    n = points.shape[0]
    if n <= 2:
        return 0, 0
    ids = np.arange(n)
    if n > sample_max:
        rng = np.random.default_rng(42)
        ids = rng.choice(ids, size=sample_max, replace=False)
    boundary = 0
    for i in ids:
        nbr = knn_indices(points, i, k=k)
        uniq = np.unique(labels[nbr])
        if uniq.size >= 2:
            boundary += 1
    return boundary, ids.size


def parse_args() -> argparse.Namespace:
    default_phenotype = str(Path(__file__).resolve().parent / "真实表型提取代码_0327.py")
    parser = argparse.ArgumentParser(description="ROI主干去噪改标并生成表型输入点云")
    parser.add_argument("--input_ply", required=True, help="输入ASCII PLY绝对路径")
    parser.add_argument("--output_scene_dir", required=True, help="输出场景目录绝对路径")
    parser.add_argument("--scene_name", default="filtered")
    parser.add_argument("--roi", nargs=6, type=float, metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"))
    parser.add_argument("--axis", choices=["x", "y", "z"], default="y")
    parser.add_argument("--stem_label", type=int, default=3)
    parser.add_argument("--root_labels", default="0,221")
    parser.add_argument("--special_labels", default="220,221,222,223,224,225,226")
    parser.add_argument("--main_mode", choices=["full", "stem-only"], default="full")

    parser.add_argument("--roi_dbscan_eps", type=float, default=0.08)
    parser.add_argument("--roi_dbscan_min_points", type=int, default=20)

    parser.add_argument("--stem_seed_radius", type=float, default=0.08)
    parser.add_argument("--stem_radius_min", type=float, default=0.04)
    parser.add_argument("--stem_radius_max", type=float, default=0.25)
    parser.add_argument("--stem_connect_eps", type=float, default=0.06)
    parser.add_argument("--stem_connect_min_points", type=int, default=12)
    parser.add_argument("--min_seed_points", type=int, default=15)

    parser.add_argument("--vote_k", type=int, default=24)
    parser.add_argument("--vote_majority", type=float, default=0.72)
    parser.add_argument("--vote_cluster_eps", type=float, default=0.03)
    parser.add_argument("--vote_small_cluster_max", type=int, default=40)
    parser.add_argument("--vote_boundary_nonmajor_max", type=int, default=6)

    parser.add_argument("--upper_eps", type=float, default=0.05)
    parser.add_argument("--upper_min_points", type=int, default=8)
    parser.add_argument("--upper_small_cluster_max", type=int, default=30)
    parser.add_argument("--upper_reassign_dist", type=float, default=0.08)

    parser.add_argument("--gate_min_continuity", type=float, default=0.88)
    parser.add_argument("--gate_max_breaks", type=int, default=2)
    parser.add_argument("--gate_max_radius_cv", type=float, default=0.55)
    parser.add_argument("--gate_max_relabel_ratio", type=float, default=0.08)
    parser.add_argument("--gate_min_relabel_consistency", type=float, default=0.70)
    parser.add_argument("--gate_min_boundary_retention", type=float, default=0.92)
    parser.add_argument("--gate_min_points_per_bin", type=int, default=3)

    parser.add_argument("--run_phenotype", action="store_true")
    parser.add_argument(
        "--phenotype_script",
        default=default_phenotype,
    )
    parser.add_argument("--phenotype_skip_skeleton", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = os.path.abspath(args.input_ply)
    scene_dir = os.path.abspath(args.output_scene_dir)
    out_dir = os.path.join(scene_dir, f"output_{args.scene_name}")
    train_dir = os.path.join(out_dir, "train")
    os.makedirs(train_dir, exist_ok=True)

    special_labels = parse_int_set(args.special_labels)
    root_labels = parse_int_set(args.root_labels)
    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_idx = axis_map[args.axis]

    ply = read_ascii_ply(input_path)
    xyz = np.stack(
        [
            ply.data[:, ply.prop_index["x"]],
            ply.data[:, ply.prop_index["y"]],
            ply.data[:, ply.prop_index["z"]],
        ],
        axis=1,
    )
    labels_orig = ply.data[:, ply.prop_index["object_value"]].astype(np.int32)
    labels = labels_orig.copy()

    total_n = xyz.shape[0]
    if args.roi is None:
        roi_mask = np.ones(total_n, dtype=bool)
    else:
        xmin, xmax, ymin, ymax, zmin, zmax = args.roi
        roi_mask = (
            (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) &
            (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax) &
            (xyz[:, 2] >= zmin) & (xyz[:, 2] <= zmax)
        )

    xyz_roi = xyz[roi_mask]
    roi_largest_local = run_dbscan_largest(xyz_roi, args.roi_dbscan_eps, args.roi_dbscan_min_points)
    roi_largest_mask = np.zeros(total_n, dtype=bool)
    roi_indices = np.where(roi_mask)[0]
    roi_largest_mask[roi_indices[roi_largest_local]] = True
    keep_special_mask = roi_mask & np.isin(labels, list(special_labels))
    region_mask = roi_largest_mask | keep_special_mask

    region_idx = np.where(region_mask)[0]
    xyz_region = xyz[region_mask]
    lbl_region = labels[region_mask]

    if xyz_region.shape[0] < 20:
        raise RuntimeError("ROI/LCC filtered points are too few to continue (ROI/连通域过滤后点过少)")

    root_mask_region = np.isin(lbl_region, list(root_labels))
    if np.any(root_mask_region):
        root_centroid = xyz_region[root_mask_region].mean(axis=0)
        root_axis_val = float(np.min(xyz_region[root_mask_region, axis_idx]))
    else:
        q = np.percentile(xyz_region[:, axis_idx], 5)
        low = xyz_region[:, axis_idx] <= q
        root_centroid = xyz_region[low].mean(axis=0)
        root_axis_val = float(np.min(xyz_region[low, axis_idx]))

    special_exclude = set(special_labels) - {args.stem_label} - root_labels
    stem_candidate_region = ~np.isin(lbl_region, list(special_exclude))

    seed_region = stem_candidate_region & (lbl_region == args.stem_label)
    if np.count_nonzero(seed_region) < args.min_seed_points:
        d_root = np.linalg.norm(xyz_region - root_centroid[None, :], axis=1)
        seed_region = stem_candidate_region & (d_root <= args.stem_seed_radius)
    if np.count_nonzero(seed_region) < args.min_seed_points:
        seed_region = stem_candidate_region

    axis_vec = pca_main_axis(xyz_region[seed_region])
    if axis_vec[axis_idx] < 0:
        axis_vec = -axis_vec
    center = xyz_region[seed_region].mean(axis=0)

    dist_region = point_line_dist(xyz_region, center, axis_vec)
    seed_dist = dist_region[seed_region]
    if seed_dist.size == 0:
        seed_radius = args.stem_radius_max
    else:
        seed_radius = float(np.percentile(seed_dist, 80) * 2.5)
    seed_radius = float(np.clip(seed_radius, args.stem_radius_min, args.stem_radius_max))

    axis_vals = xyz_region[:, axis_idx]
    stem_tube_region = (
        stem_candidate_region &
        (dist_region <= seed_radius) &
        (axis_vals >= (root_axis_val - 0.03))
    )

    xyz_tube = xyz_region[stem_tube_region]
    tube_ids_region = np.where(stem_tube_region)[0]
    stem_region = np.zeros(xyz_region.shape[0], dtype=bool)
    if xyz_tube.shape[0] >= 10:
        keep_local = run_dbscan_largest(xyz_tube, args.stem_connect_eps, args.stem_connect_min_points)
        if np.count_nonzero(keep_local) > 0:
            stem_region[tube_ids_region[keep_local]] = True
        else:
            stem_region[stem_tube_region] = True
    else:
        stem_region[seed_region] = True

    stem_mask = np.zeros(total_n, dtype=bool)
    stem_mask[region_idx[stem_region]] = True

    mismatch = region_mask & stem_mask & (labels != args.stem_label) & ~np.isin(labels, list(special_labels))
    relabel_candidates = np.zeros(total_n, dtype=bool)
    mm_idx = np.where(mismatch)[0]
    if mm_idx.size > 0:
        keep_small = np.zeros(mm_idx.size, dtype=bool)
        mm_local_keep = run_dbscan_largest(xyz[mm_idx], args.vote_cluster_eps, 2)
        if mm_local_keep.size == keep_small.size:
            # run_dbscan_largest返回“最大簇成员”布尔掩码，不含完整簇ID信息；
            # 这里重新计算完整簇标签，用于按“小簇”规则筛选候选噪声点。
            mm_lbl = run_dbscan_labels(xyz[mm_idx], args.vote_cluster_eps, 2)
            if mm_lbl.max() >= 0:
                uniq, cnt = np.unique(mm_lbl[mm_lbl >= 0], return_counts=True)
                small = {u for u, c in zip(uniq, cnt) if c <= args.vote_small_cluster_max}
                keep_small = np.array([(v in small) if v >= 0 else True for v in mm_lbl], dtype=bool)
            else:
                keep_small[:] = True
        else:
            keep_small[:] = True
        relabel_candidates[mm_idx[keep_small]] = True

    relabeled = 0
    vote_consistency = []
    for idx in np.where(relabel_candidates)[0]:
        nbr = knn_indices(xyz, idx, args.vote_k)
        if nbr.size == 0:
            continue
        lbl = labels[nbr]
        uniq, cnt = np.unique(lbl, return_counts=True)
        m = int(uniq[np.argmax(cnt)])
        c = int(np.max(cnt))
        maj_ratio = c / float(nbr.size)
        non_major = int(nbr.size - c)
        if m == args.stem_label and maj_ratio >= args.vote_majority and non_major <= args.vote_boundary_nonmajor_max:
            labels[idx] = args.stem_label
            relabeled += 1
            vote_consistency.append(float(maj_ratio))

    upper_thr = np.median(xyz_region[:, axis_idx])
    upper_region = region_mask & (xyz[:, axis_idx] > upper_thr)
    floating_candidate = upper_region & (~stem_mask) & (~np.isin(labels, list(special_labels)))
    floating_drop = np.zeros(total_n, dtype=bool)
    fc_idx = np.where(floating_candidate)[0]
    if fc_idx.size > 0:
        pcd_fc = o3d.geometry.PointCloud()
        pcd_fc.points = o3d.utility.Vector3dVector(xyz[fc_idx])
        fc_lbl = np.asarray(pcd_fc.cluster_dbscan(eps=args.upper_eps, min_points=args.upper_min_points, print_progress=False))
        if fc_lbl.size > 0 and fc_lbl.max() >= 0:
            uniq, cnt = np.unique(fc_lbl[fc_lbl >= 0], return_counts=True)
            small = {u for u, c in zip(uniq, cnt) if c <= args.upper_small_cluster_max}
            float_local = np.array([(v in small) if v >= 0 else True for v in fc_lbl], dtype=bool)
            floating_idx = fc_idx[float_local]
            stable_idx = fc_idx[~float_local]
            if stable_idx.size > 0 and floating_idx.size > 0:
                stable_pts = xyz[stable_idx]
                stable_lbl = labels[stable_idx]
                for i in floating_idx:
                    d = np.linalg.norm(stable_pts - xyz[i][None, :], axis=1)
                    j = int(np.argmin(d))
                    if float(d[j]) <= args.upper_reassign_dist:
                        labels[i] = int(stable_lbl[j])
                    else:
                        floating_drop[i] = True
            else:
                floating_drop[floating_idx] = True
        else:
            floating_drop[fc_idx] = True

    # 质量门控指标
    stem_idx = np.where(stem_mask & region_mask & (~floating_drop))[0]
    stem_pts = xyz[stem_idx]
    continuity = 0.0
    breaks = 0
    radius_cv = 0.0
    if stem_pts.shape[0] >= 30:
        v = stem_pts[:, axis_idx]
        vmin, vmax = float(np.min(v)), float(np.max(v))
        bins = np.linspace(vmin, vmax, 31)
        occ = []
        radii = []
        for i in range(len(bins) - 1):
            low = bins[i]
            high = bins[i + 1]
            high_inclusive = (i == len(bins) - 2)
            m = (v >= low) & ((v <= high) if high_inclusive else (v < high))
            occ.append(np.count_nonzero(m) >= args.gate_min_points_per_bin)
            if np.count_nonzero(m) >= args.gate_min_points_per_bin:
                r = point_line_dist(stem_pts[m], center, axis_vec)
                radii.append(float(np.median(r)))
        occ_arr = np.asarray(occ, dtype=np.int32)
        continuity = safe_div(float(np.sum(occ_arr)), float(len(occ_arr)))
        if len(occ_arr) > 1:
            breaks = int(np.sum((occ_arr[:-1] == 0) & (occ_arr[1:] == 1)))
        if len(radii) >= 2:
            radius_cv = safe_div(float(np.std(radii)), float(np.mean(radii)))

    relabel_ratio = safe_div(float(relabeled), float(np.count_nonzero(region_mask)))
    relabel_consistency = float(np.mean(vote_consistency)) if vote_consistency else 1.0

    nonstem_before_mask = region_mask & (~stem_mask)
    before_pts = xyz[nonstem_before_mask]
    before_lbl = labels_orig[nonstem_before_mask]
    boundary_before, sampled = boundary_score(before_pts, before_lbl, k=min(args.vote_k, 18))

    remain_mask = region_mask & (~floating_drop)
    nonstem_after_mask = remain_mask & (~stem_mask)
    after_pts = xyz[nonstem_after_mask]
    after_lbl = labels[nonstem_after_mask]
    boundary_after, _ = boundary_score(after_pts, after_lbl, k=min(args.vote_k, 18))
    boundary_retention = safe_div(float(boundary_after), float(max(boundary_before, 1)))

    gate = {
        "continuity": continuity >= args.gate_min_continuity,
        "breaks": breaks <= args.gate_max_breaks,
        "radius_cv": radius_cv <= args.gate_max_radius_cv,
        "relabel_ratio": relabel_ratio <= args.gate_max_relabel_ratio,
        "relabel_consistency": relabel_consistency >= args.gate_min_relabel_consistency,
        "boundary_retention": boundary_retention >= args.gate_min_boundary_retention,
    }
    gate_pass = all(gate.values())

    # 输出文件
    if args.main_mode == "stem-only":
        main_mask = remain_mask & (stem_mask | np.isin(labels, list(root_labels)) | np.isin(labels, list(special_labels)))
    else:
        main_mask = remain_mask

    main_ply = os.path.join(train_dir, "point_cloud_main_clusters_merged.ply")
    hybrid_ply = os.path.join(out_dir, "hybrid_filtered_final.ply")
    stem_only_ply = os.path.join(out_dir, "stem_only_cleaned.ply")
    report_json = os.path.join(out_dir, "filtering_report.json")
    labels_json = os.path.join(out_dir, "resolved_label_roles.json")

    write_ascii_ply(main_ply, ply, main_mask, labels)
    write_binary_hybrid(hybrid_ply, ply, remain_mask)
    write_ascii_ply(stem_only_ply, ply, remain_mask & stem_mask, labels)

    role_info = {
        "stem_label": args.stem_label,
        "root_labels": sorted(root_labels),
        "special_labels": sorted(special_labels),
        "note_zh": "如你的根部语义确认为0，请将--root_labels包含0；如为221，请包含221。",
        "note_en": "If your root semantic ID is 0, include 0 in --root_labels; if it is 221, include 221.",
    }
    with open(labels_json, "w", encoding="utf-8") as f:
        json.dump(role_info, f, ensure_ascii=False, indent=2)

    ref_label_text = "220" if 220 in special_labels else "[none]"
    ear_labels = [v for v in sorted(special_labels) if 222 <= v <= 226]
    ear_label_text = ", ".join(map(str, ear_labels)) if ear_labels else "[none]"

    report = {
        "input_ply": input_path,
        "output_scene_dir": scene_dir,
        "scene_name": args.scene_name,
        "output_files": {
            "point_cloud_main_clusters_merged": main_ply,
            "hybrid_filtered_final": hybrid_ply,
            "stem_only_cleaned": stem_only_ply,
            "label_roles": labels_json,
        },
        "counts": {
            "total_input_points": int(total_n),
            "roi_points": int(np.count_nonzero(roi_mask)),
            "region_points_after_lcc": int(np.count_nonzero(region_mask)),
            "stem_points": int(np.count_nonzero(stem_mask & remain_mask)),
            "relabeled_points": int(relabeled),
            "floating_removed_points": int(np.count_nonzero(floating_drop)),
            "main_output_points": int(np.count_nonzero(main_mask)),
            "hybrid_output_points": int(np.count_nonzero(remain_mask)),
        },
        "quality_metrics": {
            "stem_continuity_ratio": continuity,
            "stem_break_count": breaks,
            "stem_radius_cv": radius_cv,
            "relabel_ratio": relabel_ratio,
            "relabel_neighborhood_consistency": relabel_consistency,
            "leaf_boundary_retention_rate": boundary_retention,
            "boundary_sampled_points": int(sampled),
        },
        "quality_gate": {
            "pass": gate_pass,
            "checks": gate,
            "thresholds": {
                "min_continuity": args.gate_min_continuity,
                "max_breaks": args.gate_max_breaks,
                "max_radius_cv": args.gate_max_radius_cv,
                "max_relabel_ratio": args.gate_max_relabel_ratio,
                "min_relabel_consistency": args.gate_min_relabel_consistency,
                "min_boundary_retention": args.gate_min_boundary_retention,
            },
        },
        "risk_hints": [
            (
                f"If reference labels {ref_label_text} "
                f"are removed, scale calibration may fail."
            ),
            (
                f"If root labels {sorted(root_labels)} are removed, plant-base height may drift."
            ),
            (
                f"If ear labels {ear_label_text} are over-pruned, "
                f"ear-height estimation may fail."
            ),
            "If relabel_ratio is too high, stem thickening / leaf swallowing may occur.",
            "若关键标签被误删，表型指标会失真；请结合quality_gate结果复核参数。",
        ],
        "parameters": vars(args),
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    phenotype_info = {"executed": False}
    if args.run_phenotype:
        cmd = [sys.executable, os.path.abspath(args.phenotype_script), scene_dir]
        if args.phenotype_skip_skeleton:
            cmd.append("--skip_skeleton")
        log_path = os.path.join(out_dir, "phenotype_run.log")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(result.stdout or "")
                if result.stderr:
                    f.write("\n[STDERR]\n")
                    f.write(result.stderr)
            phenotype_info = {
                "executed": True,
                "return_code": int(result.returncode),
                "command": cmd,
                "log_path": log_path,
            }
        except Exception as e:
            phenotype_info = {"executed": True, "error": str(e), "command": cmd}

    report["phenotype"] = phenotype_info
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("处理完成:")
    print(f"  主分析点云: {main_ply}")
    print(f"  辅助高度点云: {hybrid_ply}")
    print(f"  主干点云: {stem_only_ply}")
    print(f"  报告: {report_json}")
    print(f"  质量门控: {'PASS' if gate_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
