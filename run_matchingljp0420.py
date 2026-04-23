#!/usr/bin/env python3
"""
run_matchingljp0326.py  (全帧匹配 + Tracking 证据)
====================================================
核心功能：
  1. auto_detect_max_depth()   —— 扫描实际深度图，自动设定合理的 max_depth
  2. 全帧匹配                    —— 不再使用关键帧，所有帧都参与匹配
  3. Tracking 证据               —— 读取 YOLO Tracking 的 track_id 作为匹配强信号
  4. 三轮渐进式匹配             —— 继承自 matching_fix_patch 的修复逻辑
  5. 自动补齐 unified_masks 和 id_mapping.json 以供 3DGS 渲染
"""

import sys

ORIGINAL_SCRIPT_DIR = "/datashare/dir_liusha/xibeinonglin/1_15_提取表型"
sys.path.insert(0, ORIGINAL_SCRIPT_DIR)

import os
import re
import glob
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 导入原始 CenterPoint3DMatcher ─────────────────────────────
try:
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location(
        "matcher_module",
        os.path.join(ORIGINAL_SCRIPT_DIR,
                     "0302直接3D匹配_xin2_0305_使用位姿信息进行运动预测"
                     "+双向匹配_0311_不跑全部帧.py")
    )
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    _OrigMatcher = mod.CenterPoint3DMatcher
    print("✅ 成功从原始脚本导入 CenterPoint3DMatcher")
except Exception as e:
    print(f"❌ 无法导入原始脚本: {e}")
    sys.exit(1)

from scipy.spatial.distance import cdist


def auto_detect_max_depth(depth_dir: str, depth_scale: float = 1000.0,
                          sample_n: int = 10,
                          percentile: float = 95.0,
                          scene_hard_cap: float = 30.0) -> float:
    depth_dir = Path(depth_dir)
    files = sorted(list(depth_dir.glob("*.png")) + list(depth_dir.glob("*.tif")))

    if not files:
        print(f"  ⚠️  未找到深度图，使用默认 max_depth=10.0m")
        return 10.0

    step   = max(1, len(files) // sample_n)
    sample = files[::step][:sample_n]

    all_depths = []
    for f in sample:
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if len(img.shape) == 3:
            img = img[:, :, 0]
        d = img.astype(np.float32) / depth_scale
        valid = d[(d > 0.1) & (d < scene_hard_cap)]
        if len(valid) > 0:
            all_depths.append(valid)

    if not all_depths:
        return 10.0

    combined = np.concatenate(all_depths)
    q25, q75 = np.percentile(combined, [25, 75])
    iqr = q75 - q25
    iqr_upper = q75 + 3.0 * iqr
    combined_clean = combined[combined <= iqr_upper]

    if len(combined_clean) < 100:
        combined_clean = combined

    p_val     = float(np.percentile(combined_clean, percentile))
    max_depth = float(np.clip(p_val * 1.2, 3.0, scene_hard_cap))
    return max_depth


class OptimizedCenterPoint3DMatcher(_OrigMatcher):
    def __init__(self, config):
        self._loose_multiplier     = config.get('loose_multiplier',     3.0)
        self._loosest_multiplier   = config.get('loosest_multiplier',   8.0)
        self._allow_new_global_ids = config.get('allow_new_global_ids', True)
        self._adaptive_threshold   = config.get('adaptive_threshold',   True)

        self.frame_track_ids = {}
        self.global_track_ids = {}

        super().__init__(config)
        self._load_track_ids()
        self._adapt_threshold()

    def _load_track_ids(self):
        mask_dir = Path(self.config['mask_dir'])
        info_files = sorted(mask_dir.glob("class_info_*.json"))
        n_with_track = 0

        for info_file in info_files:
            base_name = info_file.stem.replace('class_info_', '')
            m = re.search(r'(\d+)', base_name)
            if not m: continue
            frame_num = int(m.group(1))

            with open(info_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            tid_map = {}
            for inst in data.get('instances', []):
                if 'track_id' in inst:
                    tid_map[inst['instance_id']] = inst['track_id']

            if tid_map:
                self.frame_track_ids[frame_num] = tid_map
                n_with_track += 1

    def _get_remapped_track_id(self, frame_num, local_id):
        if frame_num not in self.frame_track_ids:
            return None
        tid_map = self.frame_track_ids[frame_num]
        if local_id in tid_map:
            return tid_map[local_id]
        target_ids_sorted = sorted(
            iid for iid in tid_map.keys()
            if self.frame_class_info.get(frame_num, {}).get(local_id) == self.target_class
        )
        if 0 < local_id <= len(target_ids_sorted):
            orig_id = target_ids_sorted[local_id - 1]
            return tid_map.get(orig_id)

    def _adapt_threshold(self):
        if not self._adaptive_threshold: return
        all_pts = []
        for frame_num, frame_pts in self.instance_centers_3d.items():
            max_id = self.frame_mask_counts.get(frame_num, 0)
            for inst_id, pt in frame_pts.items():
                if inst_id <= max_id:
                    all_pts.append(pt)
        if len(all_pts) < 2: return
        pts_arr = np.array(all_pts)

        mask = np.ones(len(pts_arr), dtype=bool)
        for axis in range(pts_arr.shape[1]):
            vals = pts_arr[:, axis]
            q25, q75 = np.percentile(vals, [25, 75])
            iqr = q75 - q25
            mask &= (vals >= q25 - 3.0 * iqr) & (vals <= q75 + 3.0 * iqr)
        pts_clean = pts_arr[mask]

        if len(pts_clean) < 2: pts_clean = pts_arr

        bbox_diag_clean = np.linalg.norm(pts_clean.max(axis=0) - pts_clean.min(axis=0))
        bbox_diag = min(bbox_diag_clean, 10.0)

        adaptive_base = max(self.spatial_threshold, bbox_diag * 0.03)
        adaptive_base = min(adaptive_base, 1.0)

        if adaptive_base > self.spatial_threshold * 1.1:
            self.spatial_threshold = adaptive_base
            self.motion_threshold  = max(self.motion_threshold, adaptive_base * 1.5)

    def _assign_new_global_ids(self, frame_num, unmatched_local_ids):
        new_matches = {}
        for lid in unmatched_local_ids:
            if lid not in self.instance_centers_3d.get(frame_num, {}): continue
            stats = self.instance_stats.get(frame_num, {}).get(lid)
            if stats is None: continue
            point     = self.instance_centers_3d[frame_num][lid]
            global_id = self.global_instance_counter
            self.global_instance_counter += 1
            key = (frame_num, lid)
            self.global_instance_map[key] = global_id
            self.global_centers_3d[global_id] = {
                'point':     list(point) if not isinstance(point, list) else point,
                'frames':    [frame_num],
                'instances': [(frame_num, lid)],
                'distances': [],
                'num_views': 1,
                'confidence': 1.0,
                'is_fixed':  False,
                'class_id':  self.target_class
            }
            if frame_num in self.masks and lid in np.unique(self.masks[frame_num]):
                self.global_masks[global_id] = {frame_num: (self.masks[frame_num] == lid)}
            self.global_stats[global_id] = {
                'avg_depth':  stats.get('depth_median', 0),
                'total_area': stats.get('area', 0),
                'num_views':  1,
                'class_id':   self.target_class
            }
            self.motion_trajectories[global_id].append((frame_num, list(point) if not isinstance(point, list) else point))
            max_id = max(self.global_id_colors.keys()) if self.global_id_colors else 0
            if global_id > max_id:
                self.global_id_colors.update(self.generate_colormap(global_id))
            new_matches[lid] = {
                'global_id':    global_id,
                'confidence':   1.0,
                'match_type':   '新建',
                'evidence_count': 0
            }
        return new_matches

    def match_with_coherence(self, frame_num):
        if frame_num not in self.instance_centers_3d: return {}
        current_instances = {
            iid: pt for iid, pt in self.instance_centers_3d[frame_num].items()
            if iid <= self.frame_mask_counts.get(frame_num, 0)
        }
        if not current_instances: return {}

        global_ids, global_points = [], []
        for gid, info in self.global_centers_3d.items():
            if gid in self.fixed_class_ids.values(): continue
            global_ids.append(gid)
            global_points.append(info['point'])

        if not global_points:
            if self._allow_new_global_ids:
                nm = self._assign_new_global_ids(frame_num, list(current_instances.keys()))
                self.frame_matched_counts[frame_num] = len(nm)
                return nm
            return {}

        gp_arr   = np.array(global_points)
        frame_ids = list(current_instances.keys())
        fp_arr   = np.array(list(current_instances.values()))
        distances = cdist(fp_arr, gp_arr)

        def _one_round(pending, used_g, sp_thr, mo_thr, label):
            if not pending: return {}, []
            lidx = {lid: frame_ids.index(lid) for lid in pending}
            ev   = defaultdict(list)

            if self.frame_track_ids:
                for lid in pending:
                    tid = self._get_remapped_track_id(frame_num, lid)
                    if tid is None: continue
                    for gid in global_ids:
                        if gid in used_g: continue
                        g_tid = self.global_track_ids.get(gid)
                        if g_tid is not None and g_tid == tid:
                            ev[(lid, gid)].append({'type': 'tracking', 'score': 2.0})

            for pred in self.get_motion_predictions(frame_num, global_ids):
                gid, ppos, conf = (pred['global_id'], pred['predicted_pos'], pred['confidence'])
                for lid, fi in lidx.items():
                    d = np.linalg.norm(np.array(ppos) - fp_arr[fi])
                    if d < mo_thr * 2:
                        ev[(lid, gid)].append({'type': 'motion', 'score': conf / (1.0 + d * 5), 'distance': d})

            for lid, fi in lidx.items():
                for ji, gid in enumerate(global_ids):
                    if gid in used_g: continue
                    d = distances[fi, ji]
                    if d >= sp_thr * 1.5: continue
                    fwd = (np.argmin(distances[fi]) == ji)
                    bwd = (np.argmin(distances[:, ji]) == fi)
                    mult = 1.5 if (fwd and bwd) else (1.0 if (fwd or bwd) else 0.6)
                    ev[(lid, gid)].append({'type': 'bidirectional', 'score': mult / (1.0 + d * 5), 'distance': d})

            cands = []
            for (lid, gid), evs in ev.items():
                total = nm = nb = ns = nt = 0
                for e in evs:
                    t = e['type']
                    if   t == 'tracking':     total += e['score'] * 1.0; nt += 1
                    elif t == 'motion':       total += e['score']*self.motion_weight; nm += 1
                    elif t == 'bidirectional':total += e['score']*self.bidirectional_weight; nb += 1
                n_t = (nt > 0) + (nm > 0) + (nb > 0)
                if n_t >= 2: total *= 1.2; mt = '协同'
                elif nt: mt = 'tracking'
                elif nm: mt = '运动'
                elif nb: mt = '双向'
                else:    mt = '其他'
                cands.append({'lid': lid, 'gid': gid, 'score': total, 'mt': mt})

            cands.sort(key=lambda x: -x['score'])
            rm = {}
            ul = set()
            for c in cands:
                lid, gid = c['lid'], c['gid']
                if lid in ul or gid in used_g: continue
                conflict = any(
                    np.linalg.norm(fp_arr[frame_ids.index(sl)] - fp_arr[lidx[lid]]) < self.coherence_threshold
                    for sl in rm
                )
                if conflict: continue
                rm[lid] = {'global_id': gid, 'confidence': c['score'], 'match_type': c['mt'], 'evidence_count': 1}
                ul.add(lid); used_g.add(gid)
            still = [lid for lid in pending if lid not in ul]
            return rm, still

        final, used_g, details = {}, set(), []
        thr1 = self.spatial_threshold
        thr2 = thr1 * self._loose_multiplier
        thr3 = thr1 * self._loosest_multiplier

        r1, pending = _one_round(frame_ids, used_g, thr1, self.motion_threshold, "R1")
        final.update(r1)
        for m in r1.values(): used_g.add(m['global_id']); self._stat(m['match_type'])

        if pending:
            r2, pending = _one_round(pending, used_g, thr2, self.motion_threshold * self._loose_multiplier, "R2")
            final.update(r2)
            for m in r2.values(): used_g.add(m['global_id']); self._stat(m['match_type'])

        if pending:
            still3 = []
            for lid in pending:
                fi  = frame_ids.index(lid)
                row = distances[fi].copy()
                for ji, gid in enumerate(global_ids):
                    if gid in used_g: row[ji] = np.inf
                bj, bd = int(np.argmin(row)), float(np.min(row))
                bgid = global_ids[bj]
                if bd < thr3:
                    final[lid] = {'global_id': bgid, 'confidence': 1.0/(1.0+bd), 'match_type': '最近邻', 'evidence_count': 1}
                    used_g.add(bgid)
                    self.matching_stats['bidirectional_matches'] += 1
                else:
                    still3.append(lid)
            pending = still3

        if pending and self._allow_new_global_ids:
            nm = self._assign_new_global_ids(frame_num, pending)
            final.update(nm)
            pending = [lid for lid in pending if lid not in nm]

        self.frame_matched_counts[frame_num] = len(final)
        return final

    def _stat(self, mt):
        if 'tracking' in mt: self.matching_stats.setdefault('tracking_matches', 0); self.matching_stats['tracking_matches'] += 1
        elif '协同' in mt:  self.matching_stats['cooperative_matches']   += 1
        elif '运动' in mt: self.matching_stats['motion_matches']        += 1
        elif '双向' in mt: self.matching_stats['bidirectional_matches'] += 1
        elif '形状' in mt: self.matching_stats['shape_matches']         += 1

    # 🚀🚀🚀 [重要修改] 重写保存函数，强制生成渲染所需文件 🚀🚀🚀
    def save_results(self):
        # 1. 尝试调用基类原本的保存逻辑（保留 color_masks 等，哪怕报错也继续往下走）
        try:
            super().save_results()
        except Exception as e:
            print(f"  ⚠️ 基类 save_results 发生异常，但无大碍: {e}")

        print("\n💾 正在补齐 STEP 6 渲染所需的 unified_masks 和 id_mapping.json ...")
        out_dir = Path(self.config['output_dir'])
        out_dir.mkdir(parents=True, exist_ok=True)

        # 2. 生成 id_mapping.json
        mapping_file = out_dir / "id_mapping.json"
        mapping_data = {}
        for global_id, info in self.global_centers_3d.items():
            instances = info.get('instances', [])
            mapping_data[str(global_id)] = [
                {"frame": fn, "local_id": lid} for (fn, lid) in instances
            ]
        try:
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(mapping_data, f, indent=4, ensure_ascii=False)
            print(f"  ✅ 成功生成 id_mapping.json (全局实例数: {len(mapping_data)})")
        except Exception as e:
            print(f"  ❌ 生成 id_mapping.json 失败: {e}")

        # 3. 生成 unified_masks 目录
        unified_dir = out_dir / "unified_masks"
        unified_dir.mkdir(exist_ok=True)
        
        frames_saved = 0
        for frame_num, local_mask in self.masks.items():
            # 创建全0的 uint16 矩阵（一定要是 uint16 才能保存大ID）
            unified_mask = np.zeros_like(local_mask, dtype=np.uint16)
            for lid in np.unique(local_mask):
                if lid == 0: continue
                # 将本地 ID 映射为全局 ID
                key = (frame_num, lid)
                gid = self.global_instance_map.get(key)
                if gid is not None:
                    unified_mask[local_mask == lid] = gid
            
            # 保存为单通道 16位 PNG
            save_path = unified_dir / f"unified_mask_{frame_num:04d}.png"
            cv2.imwrite(str(save_path), unified_mask)
            frames_saved += 1
            
        print(f"  ✅ 成功生成 unified_masks 文件夹 (共转换 {frames_saved} 帧单通道整型掩码)")

    def run_matching(self):
        first_frame = self.initialize_with_first_frame()
        if first_frame is None: return False

        for (fn, lid), gid in self.global_instance_map.items():
            if fn == first_frame:
                tid = self._get_remapped_track_id(fn, lid)
                if tid is not None: self.global_track_ids[gid] = tid

        all_frames      = sorted(self.instance_centers_3d.keys())
        forward_frames  = [f for f in all_frames if f > first_frame]
        backward_frames = [f for f in all_frames if f < first_frame]
        ordered_frames  = [first_frame] + forward_frames + backward_frames

        self.total_matches = self.unmatched_total = 0

        for cf in tqdm(ordered_frames[1:], desc="处理帧"):
            matches      = self.match_with_coherence(cf)
            target_count = len([k for k in self.instance_centers_3d[cf]
                                 if k <= self.frame_mask_counts.get(cf, 0)])

            for lid, mi in matches.items():
                gid   = mi['global_id']
                point = np.array(self.instance_centers_3d[cf][lid])
                key   = (cf, lid)
                if key not in self.global_instance_map:
                    self.global_instance_map[key] = gid
                if mi.get('match_type') != '新建':
                    self.update_global_point(gid, point, cf, lid, mi.get('distance', 0.3))
                self.total_matches += 1
                
                tid = self._get_remapped_track_id(cf, lid)
                if tid is not None and gid not in self.global_track_ids:
                    self.global_track_ids[gid] = tid

            self.unmatched_total += max(0, target_count - len(matches))

        self.matching_stats['unmatched_instances'] = self.unmatched_total
        self.matching_stats['total_matches']       = self.total_matches

        # 这一步会执行我们刚刚补全的 save_results 函数
        self.save_results()
        self.save_all_centers_3d_info()
        self.verify_matching_results()
        return True


def build_config(folder_path, output_base_dir, max_depth, scene_hard_cap=30.0):
    folder_path = Path(folder_path)
    folder_name = folder_path.name
    output_dirs = list(folder_path.glob("output_*"))
    depth_dir = camera_json = None

    for od in output_dirs:
        pd = od / "train" / "ours_30000" / "depth"
        if pd.exists(): depth_dir = str(pd)
        pc = od / "cameras.json"
        if pc.exists(): camera_json = str(pc)

    if not depth_dir:
        for od in output_dirs:
            pd = od / "depth"
            if pd.exists(): depth_dir = str(pd); break
    if not camera_json:
        pc = folder_path / "cameras.json"
        if pc.exists(): camera_json = str(pc)

    mask_dir = folder_path / "masks_results" / "integer_masks"
    if mask_dir.exists():
        mask_dir = str(mask_dir)
    else:
        cands = list(folder_path.glob("**/integer_masks"))
        mask_dir = str(cands[0]) if cands else None

    output_dir = str(folder_path / "数据驱动匹配") \
        if output_base_dir is None \
        else str(Path(output_base_dir) / folder_name)

    if max_depth is None:
        max_depth = auto_detect_max_depth(depth_dir, scene_hard_cap=scene_hard_cap)
    else:
        if max_depth > scene_hard_cap: max_depth = scene_hard_cap

    all_mask_files = sorted(Path(mask_dir).glob("mask_*.png"))
    all_frame_nums = [int(re.search(r'(\d+)', mf.stem.replace('mask_', '')).group(1)) for mf in all_mask_files if re.search(r'(\d+)', mf.stem.replace('mask_', ''))]
    max_frames_val = max(all_frame_nums) if all_frame_nums else 20
    first_frame_candidates = sorted(all_frame_nums[:5]) if len(all_frame_nums) >= 5 else (all_frame_nums if all_frame_nums else [1, 2, 3, 4, 5])

    config = {
        'depth_dir':    depth_dir,
        'camera_json':  camera_json,
        'mask_dir':     mask_dir,
        'output_dir':   output_dir,
        'target_class':    3,
        'fixed_class_ids': {0: 220, 1: 221, 2: 222},
        'first_frame_candidates': first_frame_candidates,
        'depth_scale':         1000.0,
        'max_depth':           max_depth,
        'spatial_threshold':   0.15,
        'motion_threshold':    0.30,
        'loose_multiplier':    4.0,
        'loosest_multiplier':  12.0,
        'allow_new_global_ids': True,
        'adaptive_threshold':  True,
        'motion_weight':        0.4,
        'bidirectional_weight': 0.4,
        'shape_weight':         0.2,
        'coherence_threshold':  0.05,
        'confidence_threshold': 0.3,
        'min_instance_area':    10,
        'max_frames':       max_frames_val,
        'use_motion_prior': True,
        'use_top_k_depths':  True,
        'top_k_min_depths':  300,
        'colormap': 'tab20',
        'pixel_depth_save': { 'enabled': False },
        'first_frame':      None,
    }
    return config

def process_folders(folder_paths, output_base_dir=None, max_depth_override=None, scene_hard_cap=30.0):
    if isinstance(folder_paths, str): folder_paths = [folder_paths]
    results = []
    for fp in folder_paths:
        try:
            config = build_config(fp, output_base_dir, max_depth=max_depth_override, scene_hard_cap=scene_hard_cap)
            if config:
                matcher = OptimizedCenterPoint3DMatcher(config)
                success = matcher.run_matching()
                results.append({'folder': fp, 'success': success, 'output_dir': config['output_dir']})
        except Exception as e:
            results.append({'folder': fp, 'success': False, 'error': str(e)})
    return results

def main():
    parser = argparse.ArgumentParser(description='3D匹配')
    parser.add_argument('folders',       nargs='+')
    parser.add_argument('--max_depth',   type=float, default=None)
    parser.add_argument('--scene_hard_cap', type=float, default=30.0)
    parser.add_argument('--output', '-o', default=None)
    args = parser.parse_args()

    results = process_folders(args.folders, output_base_dir=args.output, max_depth_override=args.max_depth, scene_hard_cap=args.scene_hard_cap)

if __name__ == "__main__":
    main()