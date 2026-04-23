#!/usr/bin/env bash
# =============================================================================
# pipeline.sh — 3D 语义点云全自动处理流水线 (v4 掩码训练版)
#
# 核心新增（v4）：
#   STEP 2.5  create_masked_images.py  ← 新增
#       将 YOLO 分割结果叠加到原始帧，生成 masked_images/
#       2DGS 训练和后续点云处理完全基于这批掩码图像
#       训练时通过 --images masked_images 参数指定图像子目录
#       相机位姿仍来自 VGGSfM sparse/0/（文件名完全对齐）
#
#   v3 保留功能：
#     外观相似性证据 / YOLO 置信度过滤 / 噪声实例后处理 / 0302 参数回归
#
# 用法:
#   bash pipeline.sh --video /path/to/video.mp4 [选项]
#   bash pipeline.sh --scene_dir /path/to/scene [选项]
# =============================================================================
set -euo pipefail

# ─────────────────────────── 颜色输出与日志系统 ────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log_step() {
    echo -e "\n${BOLD}${BLUE}══════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  STEP $1: $2${NC}"
    echo -e "${BOLD}${BLUE}══════════════════════════════════════════════════════${NC}"
}
log_ok()   { echo -e "${GREEN}✅ $1${NC}"; }
log_warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_err()  { echo -e "${RED}❌ $1${NC}"; exit 1; }
log_info() { echo -e "${CYAN}ℹ  $1${NC}"; }
log_skip() { echo -e "${YELLOW}⏭  $1 (已满足条件，跳过)${NC}"; }
log_env()  { echo -e "${BOLD}${YELLOW}🐍 conda env: $1${NC}"; }

# =============================================================================
# ██  1. 初始化 Conda  ██
# =============================================================================
CONDA_BASE="/datashare/dir_liuzy0/anaconda3"
[[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]] \
    || log_err "找不到 conda 初始化脚本: ${CONDA_BASE}/etc/profile.d/conda.sh"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
log_ok "Conda 初始化完成 (Base: ${CONDA_BASE})"

# =============================================================================
# ██  2. 核心路径与环境配置  ██
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_VGGSFM="vggsfm"
ENV_GS="gaussian_splatting"

VGGSFM_DIR="/extdatashare/liuzy0/code/vggsfm"
YOLO_MODEL="/extdatashare/liuzy0/ma_dataljp0330/runs/segment/corn-seg-ljp04162/weights/best.pt"

GS_ITERATIONS=30000
VIDEO_FPS=2
YOLO_CONF=0.30

# =============================================================================
# ██  3. Conda Run 执行器封装  ██
# =============================================================================
_find_libittnotify() {
    local env_lib="${CONDA_BASE}/envs/${ENV_VGGSFM}/lib"
    local found
    found=$(find "$env_lib" -maxdepth 3 -name "libittnotify*.so*" 2>/dev/null | head -1)
    [[ -z "$found" ]] && found=$(ldconfig -p 2>/dev/null | awk '/libittnotify/{print $NF}' | head -1)
    echo "$found"
}

_LIBITT=$(_find_libittnotify)
if [[ -n "$_LIBITT" ]]; then
    log_info "找到 libittnotify: ${_LIBITT}，已设置 LD_PRELOAD"
    RUN_VGGSFM() {
        LD_PRELOAD="${_LIBITT}${LD_PRELOAD:+:$LD_PRELOAD}" \
        conda run --no-capture-output -n "$ENV_VGGSFM" python "$@"
    }
else
    log_warn "未找到 libittnotify.so，清空 LD_PRELOAD 规避冲突..."
    RUN_VGGSFM() { LD_PRELOAD="" conda run --no-capture-output -n "$ENV_VGGSFM" python "$@"; }
fi

RUN_GS() { conda run --no-capture-output -n "$ENV_GS" python "$@"; }

# =============================================================================
# ██  4. 命令行参数解析  ██
# =============================================================================
VIDEO=""
SCENE_DIR=""

# ── STEP 2.5 掩码图像参数 ──────────────────────────────────────────────────
MASK_TARGET_CLASSES="3"          # YOLO 类别 ID，空格分隔（3=叶片）
MASK_KEEP_ALL=false              # true 则保留所有检测类别
MASK_BG_COLOR="255 255 255"      # 背景色 BGR（白色）
MASK_DILATE=3                    # 掩码边缘膨胀像素数

# ── STEP 5 匹配参数 ────────────────────────────────────────────────────────
MAX_DEPTH=""
SCENE_HARD_CAP="15.0"
MIN_YOLO_CONF="0.3"
MIN_VIEWS="2"
NO_APPEARANCE=false

# ── 断点续跑控制 ────────────────────────────────────────────────────────────
SKIP_VGGSFM=false
SKIP_YOLO=false
SKIP_MASK_IMAGES=false           # ← 新增
SKIP_TRAIN=false
SKIP_DEPTH=false
SKIP_MATCH=false
SKIP_RENDER=false

usage() {
    cat <<EOF
用法: bash pipeline.sh [选项]

必选（二选一）:
  --video      <path>   输入视频文件路径 (.mp4)
  --scene_dir  <path>   已有场景目录（用于跳过初期步骤）

通用参数:
  --fps         <int>   抽帧帧率 (默认: $VIDEO_FPS)
  --yolo_conf   <float> YOLO 推理置信度 (默认: $YOLO_CONF)
  --iterations  <int>   2D-GS 训练次数 (默认: $GS_ITERATIONS)

★ STEP 2.5 掩码图像参数（新增）:
  --target_classes <ids>  要保留的 YOLO class_id，空格分隔 (默认: "$MASK_TARGET_CLASSES")
                          0=茎 1=根 2=叶鞘 3=叶片
  --keep_all_classes      保留所有 YOLO 检测到的类别（忽略 --target_classes）
  --bg_color <B G R>      背景填充色 BGR 格式 (默认: $MASK_BG_COLOR 白色)
  --dilate_pixels <int>   掩码边缘膨胀像素数 (默认: $MASK_DILATE)
  --skip_mask_images      跳过 STEP 2.5（masked_images/ 已存在时自动跳过）

STEP 5 匹配参数:
  --max_depth       <float> 深度上限（米），不填则自动检测
  --scene_hard_cap  <float> 场景深度硬性上限 (默认: $SCENE_HARD_CAP m)
  --min_yolo_conf   <float> YOLO 置信度过滤门槛 (默认: $MIN_YOLO_CONF)
  --min_views       <int>   全局 ID 最少出现帧数 (默认: $MIN_VIEWS)
  --no_appearance         关闭外观相似性证据（调试用）

断点续跑控制:
  --skip_vggsfm       跳过 STEP 1
  --skip_yolo         跳过 STEP 2
  --skip_mask_images  跳过 STEP 2.5
  --skip_train        跳过 STEP 3
  --skip_depth        跳过 STEP 4
  --skip_match        跳过 STEP 5
  --skip_render       跳过 STEP 6

示例:
  # 全新处理（叶片类别，白色背景）
  bash pipeline.sh --video /data/corn.mp4

  # 保留所有类别 + 黑色背景
  bash pipeline.sh --video /data/corn.mp4 \\
      --keep_all_classes --bg_color 0 0 0

  # 保留茎和叶片（class 0 和 3）
  bash pipeline.sh --video /data/corn.mp4 \\
      --target_classes "0 3"

  # 只重跑掩码生成 + 训练（YOLO 已完成）
  bash pipeline.sh --scene_dir /data/corn_frames \\
      --skip_vggsfm --skip_yolo

  # 只重跑匹配（训练已完成）
  bash pipeline.sh --scene_dir /data/corn_frames \\
      --skip_vggsfm --skip_yolo --skip_mask_images --skip_train --skip_depth

EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --video)              VIDEO="$2";              shift 2 ;;
        --scene_dir)          SCENE_DIR="$2";          shift 2 ;;
        # STEP 2.5 参数
        --target_classes)     MASK_TARGET_CLASSES="$2"; shift 2 ;;
        --keep_all_classes)   MASK_KEEP_ALL=true;      shift ;;
        --bg_color)           MASK_BG_COLOR="$2 $3 $4"; shift 4 ;;
        --dilate_pixels)      MASK_DILATE="$2";        shift 2 ;;
        --skip_mask_images)   SKIP_MASK_IMAGES=true;   shift ;;
        # STEP 5 参数
        --max_depth)          MAX_DEPTH="$2";          shift 2 ;;
        --scene_hard_cap)     SCENE_HARD_CAP="$2";     shift 2 ;;
        --min_yolo_conf)      MIN_YOLO_CONF="$2";      shift 2 ;;
        --min_views)          MIN_VIEWS="$2";          shift 2 ;;
        --no_appearance)      NO_APPEARANCE=true;      shift ;;
        # 通用参数
        --fps)                VIDEO_FPS="$2";          shift 2 ;;
        --yolo_conf)          YOLO_CONF="$2";          shift 2 ;;
        --iterations)         GS_ITERATIONS="$2";      shift 2 ;;
        # 断点续跑
        --skip_vggsfm)        SKIP_VGGSFM=true;        shift ;;
        --skip_yolo)          SKIP_YOLO=true;          shift ;;
        --skip_train)         SKIP_TRAIN=true;         shift ;;
        --skip_depth)         SKIP_DEPTH=true;         shift ;;
        --skip_match)         SKIP_MATCH=true;         shift ;;
        --skip_render)        SKIP_RENDER=true;        shift ;;
        -h|--help)            usage ;;
        *) log_err "未知参数: $1（使用 -h 查看帮助）" ;;
    esac
done

# ── 路径推断与创建 ────────────────────────────────────────────
[[ -z "$VIDEO" && -z "$SCENE_DIR" ]] && log_err "请提供 --video 或 --scene_dir"

if [[ -n "$VIDEO" && -z "$SCENE_DIR" ]]; then
    VIDEO_ABS="$(realpath "$VIDEO")"
    SCENE_NAME="$(basename "$VIDEO_ABS" .mp4)"
    DIR_NAME="$(basename "$(dirname "$VIDEO_ABS")")"
    MAIN_DIR="$([ "$DIR_NAME" = "video" ] && dirname "$(dirname "$VIDEO_ABS")" || dirname "$VIDEO_ABS")"
    SCENE_DIR="${MAIN_DIR}/${SCENE_NAME}"
fi

SCENE_DIR="$(realpath -m "$SCENE_DIR")"
mkdir -p "$SCENE_DIR"

# ── 校验脚本 ──────────────────────────────────────────────────
for f in "vggsfm_cli.py" "reason_cli.py" "create_masked_images.py" \
         "train_0211之前.py" "pirender深度图.py" \
         "run_matchingljp0326.py" "pirenderljp0326.py"; do
    [[ -f "${SCRIPT_DIR}/$f" ]] || log_err "在 ${SCRIPT_DIR} 找不到脚本: $f"
done
[[ -f "$YOLO_MODEL" ]] || log_err "YOLO 模型不存在: $YOLO_MODEL"

# =============================================================================
# ██  5. 全局日志  ██
# =============================================================================
LOG_FILE="${SCENE_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║   corn-seg 3D Semantic Pipeline (v4 掩码训练版)      ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${NC}"
printf "  %-22s: %s\n" "场景目录"           "$SCENE_DIR"
printf "  %-22s: %s\n" "输入视频"           "${VIDEO:-[使用已有场景目录]}"
printf "  %-22s: %s\n" "掩码类别"           "$( $MASK_KEEP_ALL && echo '全部类别' || echo "class $MASK_TARGET_CLASSES")"
printf "  %-22s: %s\n" "背景色(BGR)"        "$MASK_BG_COLOR"
printf "  %-22s: %s\n" "掩码边缘膨胀"       "${MASK_DILATE}px"
printf "  %-22s: %s\n" "max_depth"          "${MAX_DEPTH:-自动检测}"
printf "  %-22s: %s\n" "scene_hard_cap"     "${SCENE_HARD_CAP}m"
printf "  %-22s: %s\n" "min_yolo_conf"      "$MIN_YOLO_CONF"
printf "  %-22s: %s\n" "min_views"          "$MIN_VIEWS"
printf "  %-22s: %s\n" "外观相似性"         "$( $NO_APPEARANCE && echo '关闭(调试)' || echo '开启')"
printf "  %-22s: %s\n" "日志文件"           "$LOG_FILE"
echo ""

PIPELINE_START=$(date +%s)
elapsed() { local s=$(( $(date +%s) - $1 )); printf "%dm%02ds" $(( s/60 )) $(( s%60 )); }

# =============================================================================
# STEP 1: VGGSfM 视频抽帧 + 稀疏重建
# =============================================================================
log_step 1 "VGGSfM 视频抽帧 + 稀疏重建"
log_env "$ENV_VGGSFM"
T1=$(date +%s)

if $SKIP_VGGSFM; then
    log_skip "VGGSfM (--skip_vggsfm)"
else
    if [[ -d "${SCENE_DIR}/sparse" ]]; then
        log_skip "sparse/ 已存在"
    else
        [[ -n "$VIDEO" ]] || log_err "未提供视频路径（--video）"
        cd "$VGGSFM_DIR"
        RUN_VGGSFM "${SCRIPT_DIR}/vggsfm_cli.py" \
            --video "$VIDEO_ABS" --scene_dir "$SCENE_DIR" \
            --fps "$VIDEO_FPS" --vggsfm_dir "$VGGSFM_DIR"
        cd - > /dev/null
    fi
fi

N_FRAMES=$(find "${SCENE_DIR}/images" -maxdepth 1 \
    \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l || echo 0)
log_info "图像总帧数: ${N_FRAMES}"

# sparse/0 目录对齐
SPARSE_DIR="${SCENE_DIR}/sparse"
SPARSE_0_DIR="${SPARSE_DIR}/0"
if [[ -d "$SPARSE_DIR" && ! -d "$SPARSE_0_DIR" ]]; then
    log_info "对齐 sparse/ → sparse/0/ ..."
    HAS_COLMAP_FILES=false
    for ext in bin txt; do
        for f in cameras images points3D; do
            [[ -f "${SPARSE_DIR}/${f}.${ext}" ]] && HAS_COLMAP_FILES=true && break 2
        done
    done
    if $HAS_COLMAP_FILES; then
        mkdir -p "$SPARSE_0_DIR"
        for ext in bin txt; do
            for f in cameras images points3D; do
                SRC="${SPARSE_DIR}/${f}.${ext}"
                [[ -f "$SRC" ]] && mv "$SRC" "${SPARSE_0_DIR}/"
            done
        done
        log_ok "sparse/0/ 对齐完成"
    fi
fi
log_ok "STEP 1 完成 (用时 $(elapsed $T1))"

# =============================================================================
# STEP 2: YOLO 2D 实例分割
# =============================================================================
log_step 2 "YOLO 2D 实例分割"
log_env "$ENV_GS"
T2=$(date +%s)

MASK_DIR="${SCENE_DIR}/masks_results/integer_masks"

if $SKIP_YOLO; then
    log_skip "YOLO (--skip_yolo)"
else
    EXISTING_MASKS=$(find "$MASK_DIR" -name "mask_*.png" 2>/dev/null | wc -l || echo 0)
    if [[ "$EXISTING_MASKS" -ge "$N_FRAMES" && "$EXISTING_MASKS" -gt 0 ]]; then
        log_skip "掩码已完整 ($EXISTING_MASKS 帧)"
    else
        RUN_GS "${SCRIPT_DIR}/reason_cli.py" \
            --scene_dir "$SCENE_DIR" \
            --model     "$YOLO_MODEL" \
            --conf      "$YOLO_CONF" \
            --save_viz
    fi
fi
log_ok "STEP 2 完成 (用时 $(elapsed $T2))"

# =============================================================================
# STEP 2.5: 生成掩码训练图像（★ 新增）
# =============================================================================
log_step "2.5" "生成 YOLO 掩码图像用于 2DGS 训练"
log_env "$ENV_GS"
T25=$(date +%s)

MASKED_IMAGES_DIR="${SCENE_DIR}/masked_images"

if $SKIP_MASK_IMAGES; then
    log_skip "掩码图像生成 (--skip_mask_images)"
else
    # 检查是否已完整生成（与原始帧数相同）
    EXISTING_MASKED=$(find "$MASKED_IMAGES_DIR" \
        \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l || echo 0)

    if [[ "$EXISTING_MASKED" -ge "$N_FRAMES" && "$EXISTING_MASKED" -gt 0 ]]; then
        log_skip "masked_images/ 已完整 ($EXISTING_MASKED 张)"
    else
        # 构建命令
        MASK_CMD=(
            "${SCRIPT_DIR}/create_masked_images.py"
            "$SCENE_DIR"
            "--bg_color" $MASK_BG_COLOR
            "--dilate_pixels" "$MASK_DILATE"
            "--output_dir_name" "masked_images"
            "--verify"
        )

        if $MASK_KEEP_ALL; then
            MASK_CMD+=("--keep_all_classes")
            log_info "保留所有 YOLO 检测类别"
        else
            # 将空格分隔的类别 ID 拆分传入
            # shellcheck disable=SC2086
            MASK_CMD+=("--target_classes" $MASK_TARGET_CLASSES)
            log_info "保留类别: $MASK_TARGET_CLASSES  (0=茎 1=根 2=叶鞘 3=叶片)"
        fi

        log_info "背景色 BGR: $MASK_BG_COLOR  |  边缘膨胀: ${MASK_DILATE}px"
        RUN_GS "${MASK_CMD[@]}"

        # 验证输出数量
        GENERATED=$(find "$MASKED_IMAGES_DIR" \
            \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l || echo 0)
        log_info "生成掩码图像: ${GENERATED} 张"

        if [[ "$GENERATED" -lt "$N_FRAMES" ]]; then
            log_warn "掩码图像数 ($GENERATED) 少于原始帧数 ($N_FRAMES)，请检查 YOLO 结果"
        fi
    fi
fi
log_ok "STEP 2.5 完成 (用时 $(elapsed $T25))"

# =============================================================================
# STEP 3: 2D-GS 训练（使用掩码图像）
# =============================================================================
log_step 3 "2D-GS 高斯重建 (iter=$GS_ITERATIONS，图像=masked_images/)"
log_env "$ENV_GS"
T3=$(date +%s)

SCENE_BASENAME="$(basename "$SCENE_DIR")"
BASE_NAME="${SCENE_BASENAME%_frames}"
OUTPUT_DIR="${SCENE_DIR}/output_${BASE_NAME}"

if $SKIP_TRAIN; then
    log_skip "2D-GS 训练 (--skip_train)"
else
    if [[ -d "$OUTPUT_DIR" && -f "${OUTPUT_DIR}/cameras.json" ]]; then
        log_skip "output_*/cameras.json 已存在"
    else
        # ★ 关键变更：传入 --images masked_images
        # 相机位姿仍来自 sparse/0/（VGGSfM 输出，文件名与 masked_images/ 完全对齐）
        # 2DGS 训练的光度监督使用 masked_images/ 中的掩码图像
        log_info "训练图像子目录: masked_images/  (相机位姿来自 sparse/0/)"

        RUN_GS "${SCRIPT_DIR}/train_0211之前.py" \
            --source_path "$SCENE_DIR" \
            -m            "$OUTPUT_DIR" \
            --iterations  "$GS_ITERATIONS" \
            --images      "masked_images"
    fi
fi
log_ok "STEP 3 完成 (用时 $(elapsed $T3))"

# =============================================================================
# STEP 4: 渲染深度图（基于掩码训练的 2DGS 模型）
# =============================================================================
log_step 4 "渲染全部深度图"
log_env "$ENV_GS"
T4=$(date +%s)

DEPTH_DIR="${OUTPUT_DIR}/train/ours_${GS_ITERATIONS}/depth"

if $SKIP_DEPTH; then
    log_skip "深度图渲染 (--skip_depth)"
else
    N_DEPTH=$(find "$DEPTH_DIR" -name "*.png" 2>/dev/null | wc -l || echo 0)
    if [[ "$N_DEPTH" -ge "$N_FRAMES" && "$N_DEPTH" -gt 0 ]]; then
        log_skip "深度图已全部生成 ($N_DEPTH 帧)"
    else
        RUN_GS "${SCRIPT_DIR}/pirender深度图.py" \
            "$OUTPUT_DIR" \
            --iteration "$GS_ITERATIONS" \
            --skip_mesh \
            --resolution 1
        N_DEPTH=$(find "$DEPTH_DIR" -name "*.png" 2>/dev/null | wc -l || echo 0)
        log_info "渲染完成，共 ${N_DEPTH} 帧深度图"
    fi
fi
log_ok "STEP 4 完成 (用时 $(elapsed $T4))"

# =============================================================================
# STEP 5: 3D 语义匹配 (v3 外观增强 + 噪声过滤 + 0302 参数)
# =============================================================================
log_step 5 "3D 语义匹配 v3（外观增强 + 噪声过滤）"
log_env "$ENV_GS"
T5=$(date +%s)

MATCH_DIR="${SCENE_DIR}/数据驱动匹配"

if $SKIP_MATCH; then
    log_skip "3D 匹配 (--skip_match)"
else
    if [[ -f "${MATCH_DIR}/id_mapping.json" ]]; then
        log_skip "id_mapping.json 已存在"
    else
        MATCH_CMD=(
            "${SCRIPT_DIR}/run_matchingljp0326.py"
            "$SCENE_DIR"
            "--scene_hard_cap" "$SCENE_HARD_CAP"
            "--min_yolo_conf"  "$MIN_YOLO_CONF"
            "--min_views"      "$MIN_VIEWS"
        )

        if [[ -n "$MAX_DEPTH" ]]; then
            MATCH_CMD+=("--max_depth" "$MAX_DEPTH")
            log_info "手动 max_depth=${MAX_DEPTH}m"
        else
            log_info "max_depth 自动检测 (hard_cap=${SCENE_HARD_CAP}m)"
        fi

        $NO_APPEARANCE && MATCH_CMD+=("--no_appearance") && log_warn "外观相似性已关闭（调试模式）"

        log_info "参数版本: 0302 回归 (spatial=0.3, top_k=2000)"
        export ORIGINAL_SCRIPT_DIR="${SCRIPT_DIR}"
        RUN_GS "${MATCH_CMD[@]}"
    fi
fi
log_ok "STEP 5 完成 (用时 $(elapsed $T5))"

# =============================================================================
# STEP 6: 最终渲染（带全局语义 ID）
# =============================================================================
log_step 6 "带全局语义 ID 的最终渲染"
log_env "$ENV_GS"
T6=$(date +%s)

if $SKIP_RENDER; then
    log_skip "最终渲染 (--skip_render)"
else
    RUN_GS "${SCRIPT_DIR}/pirenderljp0326.py" \
        "$SCENE_DIR" \
        --gs_id \
        --skip_train \
        --skip_test \
        --skip_mesh \
        --iteration "$GS_ITERATIONS"
fi
log_ok "STEP 6 完成 (用时 $(elapsed $T6))"

# =============================================================================
# 完成汇总
# =============================================================================
TOTAL_SEC=$(( $(date +%s) - PIPELINE_START ))

echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║       全流程完美结束 🎉  (v4 掩码训练版)            ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
printf "  %-22s: %s\n" "总耗时"          "$(printf "%dm%02ds" $(( TOTAL_SEC/60 )) $(( TOTAL_SEC%60 )))"
printf "  %-22s: %s\n" "场景目录"        "$SCENE_DIR"
printf "  %-22s: %s\n" "掩码图像目录"    "${SCENE_DIR}/masked_images"
printf "  %-22s: %s\n" "2DGS 输出"       "$OUTPUT_DIR"
printf "  %-22s: %s\n" "深度图目录"      "$DEPTH_DIR"
printf "  %-22s: %s\n" "3D 匹配输出"     "$MATCH_DIR"
printf "  %-22s: %s\n" "噪声ID清单"      "${MATCH_DIR}/noise_ids_removed.txt"
echo -e "\n${CYAN}日志: ${LOG_FILE}${NC}"
