import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN


def read_ply(path: Path) -> np.ndarray:
    points = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        in_header = True
        for line in f:
            line = line.strip()
            if in_header:
                if line == "end_header":
                    in_header = False
                continue
            values = line.split()
            if len(values) < 7:
                continue
            points.append(
                [
                    float(values[0]),
                    float(values[1]),
                    float(values[2]),
                    int(values[3]),
                    int(values[4]),
                    int(values[5]),
                    int(values[6]),
                ]
            )
    return np.asarray(points, dtype=np.float64)


def write_ply(path: Path, pts: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property int object_value\n")
        f.write("end_header\n")
        for p in pts:
            f.write(
                f"{p[0]} {p[1]} {p[2]} "
                f"{int(p[3])} {int(p[4])} {int(p[5])} {int(p[6])}\n"
            )


def label_stats(pts: np.ndarray) -> Dict[int, int]:
    labels, counts = np.unique(pts[:, 6].astype(int), return_counts=True)
    return dict(zip(labels.tolist(), counts.tolist()))


def step1_remove_isolated_clusters(
    pts: np.ndarray,
    eps: float = 0.3,
    min_samples: int = 10,
    keep_top_n: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(pts[:, :3])
    cluster_ids = db.labels_

    valid_clusters = cluster_ids[cluster_ids >= 0]
    if len(valid_clusters) == 0:
        return pts.copy(), np.empty((0, pts.shape[1]), dtype=pts.dtype), cluster_ids

    unique, counts = np.unique(valid_clusters, return_counts=True)
    top_clusters = set(unique[np.argsort(-counts)[:keep_top_n]].tolist())
    keep_mask = np.isin(cluster_ids, list(top_clusters))

    return pts[keep_mask], pts[~keep_mask], cluster_ids


def _majority_label(row: np.ndarray) -> Tuple[int, float]:
    binc = np.bincount(row.astype(int))
    best = int(binc.argmax())
    ratio = float(binc[best] / len(row))
    return best, ratio


def step2_knn_label_fix(
    pts: np.ndarray,
    noise_labels: Iterable[int] = (220, 221, 222),
    core_labels: Iterable[int] = tuple(range(1, 10)),
    k: int = 15,
    noise_majority_thresh: float = 0.6,
    reverse_majority_thresh: float = 0.7,
    reverse_self_min_count: int = 2,
) -> Tuple[np.ndarray, Dict[str, int]]:
    labels = pts[:, 6].astype(int)
    xyz = pts[:, :3]
    new_labels = labels.copy()

    core_labels = tuple(core_labels)
    noise_labels = tuple(noise_labels)

    core_mask = np.isin(labels, core_labels)
    core_indices = np.flatnonzero(core_mask)
    core_points = xyz[core_mask]
    core_labels_arr = labels[core_mask]

    if len(core_points) == 0:
        out = pts.copy()
        out[:, 6] = new_labels
        return out, {"noise_to_core": 0, "core_to_core": 0}

    core_tree = KDTree(core_points)
    k_eff = min(k, len(core_points))

    changed_noise_to_core = 0
    for noise_label in noise_labels:
        noise_mask = labels == noise_label
        noise_indices = np.flatnonzero(noise_mask)
        if len(noise_indices) == 0:
            continue
        query_pts = xyz[noise_indices]
        _, idxs = core_tree.query(query_pts, k=k_eff)
        if idxs.ndim == 1:
            idxs = idxs[:, None]
        neighbor_labels = core_labels_arr[idxs]

        for local_i, neighbor_row in enumerate(neighbor_labels):
            best_label, ratio = _majority_label(neighbor_row)
            if ratio >= noise_majority_thresh:
                new_labels[noise_indices[local_i]] = best_label
                changed_noise_to_core += 1

    # Reverse correction for core labels: local minority points are reassigned.
    changed_core_to_core = 0
    for core_label in core_labels:
        label_mask = labels == core_label
        label_indices = np.flatnonzero(label_mask)
        if len(label_indices) == 0:
            continue
        query_pts = xyz[label_indices]
        _, idxs = core_tree.query(query_pts, k=k_eff)
        if idxs.ndim == 1:
            idxs = idxs[:, None]
        neighbor_labels = core_labels_arr[idxs]

        for local_i, neighbor_row in enumerate(neighbor_labels):
            self_count = int(np.sum(neighbor_row == core_label))
            best_label, ratio = _majority_label(neighbor_row)
            if (
                self_count < reverse_self_min_count
                and ratio >= reverse_majority_thresh
                and best_label != core_label
            ):
                new_labels[label_indices[local_i]] = best_label
                changed_core_to_core += 1

    out = pts.copy()
    out[:, 6] = new_labels
    return out, {"noise_to_core": changed_noise_to_core, "core_to_core": changed_core_to_core}


def step3_intra_label_outlier_removal(
    pts: np.ndarray,
    core_labels: Iterable[int] = tuple(range(1, 10)),
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    z_threshold: float = 4.7,
    high_z_dist_thresh: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    labels = pts[:, 6].astype(int)
    keep_parts = []
    noise_parts = []

    stat_removed_count = 0
    for label in np.unique(labels):
        mask = labels == label
        sub = pts[mask]
        if len(sub) < 30:
            keep_parts.append(sub)
            continue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sub[:, :3])
        _, inlier_idx = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )
        inlier_idx = np.asarray(inlier_idx, dtype=int)
        outlier_idx = np.setdiff1d(np.arange(len(sub)), inlier_idx)

        keep_parts.append(sub[inlier_idx])
        if len(outlier_idx) > 0:
            noise_parts.append(sub[outlier_idx])
            stat_removed_count += len(outlier_idx)

    kept = np.vstack(keep_parts) if keep_parts else np.empty((0, pts.shape[1]), dtype=pts.dtype)
    stat_noise = (
        np.vstack(noise_parts) if noise_parts else np.empty((0, pts.shape[1]), dtype=pts.dtype)
    )

    # Extra high-Z rule: remove points with Z > z_threshold and far from any core label.
    kept_labels = kept[:, 6].astype(int) if len(kept) else np.array([], dtype=int)
    core_mask = np.isin(kept_labels, list(core_labels))
    core_points = kept[core_mask, :3]

    high_z_removed_count = 0
    if len(kept) > 0 and len(core_points) > 0:
        high_z_mask = kept[:, 2] > z_threshold
        high_z_idx = np.flatnonzero(high_z_mask)
        if len(high_z_idx) > 0:
            tree = KDTree(core_points)
            dists, _ = tree.query(kept[high_z_idx, :3], k=1)
            remove_high_z = dists > high_z_dist_thresh
            remove_idx = high_z_idx[remove_high_z]
            if len(remove_idx) > 0:
                keep_mask = np.ones(len(kept), dtype=bool)
                keep_mask[remove_idx] = False
                high_z_noise = kept[remove_idx]
                kept = kept[keep_mask]
                stat_noise = (
                    np.vstack([stat_noise, high_z_noise])
                    if len(stat_noise)
                    else high_z_noise.copy()
                )
                high_z_removed_count = len(remove_idx)

    return kept, stat_noise, {"stat_removed": stat_removed_count, "high_z_removed": high_z_removed_count}


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="玉米点云清洗三步流水线")
    parser.add_argument(
        "--input",
        type=Path,
        default=base_dir / "point_cloud_non_bg_with_values.ply",
        help="输入 PLY 文件路径（ASCII，含 object_value）",
    )
    parser.add_argument(
        "--output-clean",
        type=Path,
        default=base_dir / "point_cloud_cleaned.ply",
        help="清洗后输出 PLY 路径",
    )
    parser.add_argument(
        "--output-noise",
        type=Path,
        default=base_dir / "point_cloud_noise.ply",
        help="噪声点输出 PLY 路径",
    )
    parser.add_argument(
        "--stats-before",
        type=Path,
        default=base_dir / "label_stats_before.json",
        help="处理前标签统计 JSON 路径",
    )
    parser.add_argument(
        "--stats-after",
        type=Path,
        default=base_dir / "label_stats_after.json",
        help="处理后标签统计 JSON 路径",
    )
    args = parser.parse_args()

    print("=== 读取点云 ===")
    pts = read_ply(args.input)
    print(f"输入文件: {args.input}")
    print(f"原始点数: {len(pts)}")

    stats_before = label_stats(pts)
    print(f"处理前标签分布: {stats_before}")

    print("\n=== Step 1: 去除外围孤立噪声簇 ===")
    pts_s1, noise_s1, cluster_ids = step1_remove_isolated_clusters(pts)
    print(f"保留点数: {len(pts_s1)}")
    print(f"剔除点数: {len(noise_s1)}")
    print(f"DBSCAN 噪声点数: {int(np.sum(cluster_ids == -1))}")

    print("\n=== Step 2: KNN 多数票标签修正 ===")
    pts_s2, step2_stats = step2_knn_label_fix(pts_s1)
    print(f"noise->core 修正数量: {step2_stats['noise_to_core']}")
    print(f"core->core 修正数量: {step2_stats['core_to_core']}")

    print("\n=== Step 3: 标签内统计去噪 + 高度层规则 ===")
    pts_final, noise_s3, step3_stats = step3_intra_label_outlier_removal(pts_s2)
    print(f"统计离群点删除数量: {step3_stats['stat_removed']}")
    print(f"高 Z 规则删除数量: {step3_stats['high_z_removed']}")
    print(f"Step 3 后保留点数: {len(pts_final)}")

    stats_after = label_stats(pts_final)
    print(f"\n处理后标签分布: {stats_after}")

    all_noise = (
        np.vstack([noise_s1, noise_s3])
        if len(noise_s1) and len(noise_s3)
        else (noise_s1 if len(noise_s1) else noise_s3)
    )

    args.output_clean.parent.mkdir(parents=True, exist_ok=True)
    args.output_noise.parent.mkdir(parents=True, exist_ok=True)
    args.stats_before.parent.mkdir(parents=True, exist_ok=True)
    args.stats_after.parent.mkdir(parents=True, exist_ok=True)

    write_ply(args.output_clean, pts_final)
    write_ply(args.output_noise, all_noise if len(all_noise) else np.empty((0, 7)))
    args.stats_before.write_text(
        json.dumps(stats_before, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.stats_after.write_text(
        json.dumps(stats_after, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== 输出 ===")
    print(f"清洗点云: {args.output_clean} ({len(pts_final)} 点)")
    print(f"噪声点云: {args.output_noise} ({len(all_noise)} 点)")
    print(f"标签统计(前): {args.stats_before}")
    print(f"标签统计(后): {args.stats_after}")

    for key, min_required in {220: 50, 221: 30}.items():
        remaining = stats_after.get(key, 0)
        if remaining < min_required:
            print(
                f"⚠️ 警告: label {key} 仅剩 {remaining} 点 (< {min_required})，"
                "可能影响后续表型提取稳定性。"
            )


if __name__ == "__main__":
    main()
