#!/usr/bin/env bash
# =============================================================================
# pipeline_enhanced.sh — 含 RealBasicVSR 增强的 3D 语义点云流水线 (v5)
#
# ★ 相比 pipeline.sh 的核心新增步骤:
#   STEP 1.5  realbasicvsr_cli.py  ← 图像增强
#       对 VGGSfM 抽出的原始帧进行超分辨率增强
#       输出到 images_enhanced/，文件名与 images/ 完全对齐
#       VGGSfM 相机位姿已从 images/ 计算，本步不重跑
#
# 完整流程（8步）:
#   STEP 1   VGGSfM              原始帧 → sparse/0/ 相机位姿
#   STEP 1.5 RealBasicVSR        images/ → images_enhanced/ 超分增强
#   STEP 2   YOLO                images_enhanced/ → masks_results/ 实例分割
#   STEP 2.5 create_masked_images images_enhanced/ + masks → masked_images/
#   STEP 3   2D-GS 训练          masked_images/ + sparse/0/ → output_*/
#   STEP 4   深度图渲染           output_* → depth/
#   STEP 5   3D 语义匹配         depth + masks → 数据驱动匹配/
#   STEP 6   最终渲染            → 语义点云
#
# 设计要点:
#   · VGGSfM 使用原始 images/（保证特征匹配质量）
#   · YOLO 使用增强 images_enhanced/（更高清晰度 → 更好检测）
#   · 3DGS 使用 masked_images/（增强图像 + YOLO 掩码 → 更好重建质量）
#   · 深度图和相机参数来自 3DGS 点云，不依赖原始稀疏点云
#   · 所有步骤均支持断点续跑
#
# 用法:
#   bash pipeline_enhanced.sh --video /path/to/video.mp4 [选项]
#   bash pipeline_enhanced.sh --scene_dir /path/to/scene [选项]
# =============================================================================
set -euo pipefail

# ─────────────────────────── 颜色输出与日志系统 ────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
MAGENTA='\033[0;35m'

log_step()  {
    echo -e "\n${BOLD}${BLUE}══════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  STEP $1: $2${NC}"
    echo -e "${BOLD}${BLUE}══════════════════════════════════════════════════════${NC}"
}
log_new()   { echo -e "\n${BOLD}${MAGENTA}══════════════════════════════════════════════════════${NC}";
              echo -e "${BOLD}${MAGENTA}  ★ STEP $1: $2${NC}";
              echo -e "${BOLD}${MAGENTA}══════════════════════════════════════════════════════${NC}"; }
log_ok()    { echo -e "${GREEN}✅ $1${NC}"; }
log_warn()  { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_err()   { echo -e "${RED}❌ $1${NC}"; exit 1; }
log_info()  { echo -e "${CYAN}ℹ  $1${NC}"; }
log_skip()  { echo -e "${YELLOW}⏭  $1 (已满足条件，跳过)${NC}"; }
log_env()   { echo -e "${BOLD}${YELLOW}🐍 conda env: $1${NC}"; }

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
ENV_RBVSR="gaussian_splatting"        # ★ RealBasicVSR conda 环境
ENV_GS="gaussian_splatting"

VGGSFM_DIR="/extdatashare/liuzy0/code/vggsfm"
RBVSR_DIR="/extdatashare/liuzy0/code/RealBasicVSR"   # ★ RealBasicVSR 代码目录
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
    [[ -z "$found" ]] && \
        found=$(ldconfig -p 2>/dev/null | awk '/libittnotify/{print $NF}' | head -1)
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
    RUN_VGGSFM() {
        LD_PRELOAD="" conda run --no-capture-output -n "$ENV_VGGSFM" python "$@"
    }
fi

RUN_RBVSR() { conda run --no-capture-output -n "$ENV_RBVSR" python "$@"; }
RUN_GS()    { conda run --no-capture-output -n "$ENV_GS"    python "$@"; }

# =============================================================================
# ██  4. 命令行参数解析  ██
# =============================================================================
VIDEO=""
SCENE_DIR=""

# ── ★ STEP 1.5 RealBasicVSR 参数 ──────────────────────────────────────────
RBVSR_CONFIG=""          # 配置文件路径（不填则自动查找）
RBVSR_CHECKPOINT=""      # 模型权重路径（不填则自动查找）
RBVSR_MAX_SEQ_LEN=""     # 可选: --max_seq_len
RBVSR_SAVE_PNG=""        # 可选: --is_save_as_png 0/1

# ── STEP 2.5 掩码图像参数 ──────────────────────────────────────────────────
MASK_TARGET_CLASSES="3"
MASK_KEEP_ALL=false
MASK_BG_COLOR="255 255 255"
MASK_DILATE=3

# ── STEP 5 匹配参数 ────────────────────────────────────────────────────────
MAX_DEPTH=""
SCENE_HARD_CAP="15.0"
MIN_YOLO_CONF="0.3"
MIN_VIEWS="2"
NO_APPEARANCE=false

# ── 通用参数 ────────────────────────────────────────────────────────────────
VIDEO_FPS=2
YOLO_CONF=0.30
GS_ITERATIONS=30000

# ── 断点续跑控制 ────────────────────────────────────────────────────────────
SKIP_VGGSFM=false
SKIP_ENHANCE=false          # ★ 新增
SKIP_YOLO=false
SKIP_MASK_IMAGES=false
SKIP_TRAIN=false
SKIP_DEPTH=false
SKIP_MATCH=false
SKIP_RENDER=false

usage() {
    cat <<'EOF'
用法: bash pipeline_enhanced.sh [选项]

必选（二选一）:
  --video      <path>   输入视频文件路径 (.mp4)
  --scene_dir  <path>   已有场景目录（用于跳过初期步骤）

通用参数:
  --fps         <int>   抽帧帧率 (默认: 2)
  --yolo_conf   <float> YOLO 推理置信度 (默认: 0.30)
  --iterations  <int>   2D-GS 训练次数 (默认: 30000)

★ STEP 1.5 RealBasicVSR 参数（新增）:
  --rbvsr_dir        <path> RealBasicVSR 代码目录
                            (默认: /extdatashare/liuzy0/code/RealBasicVSR)
  --rbvsr_config     <path> 配置文件路径 options/test/RealBasicVSR/test_*.yml
                            （不填则自动查找）
  --rbvsr_checkpoint <path> 模型权重路径 checkpoints/RealBasicVSR_x4.pth
                            （不填则自动查找）
  --rbvsr_max_seq_len <int> 最大序列长度（可选）
  --rbvsr_save_png    <0/1> 是否保存为PNG: 0=否 1=是（可选）
  --skip_enhance            跳过 STEP 1.5（images_enhanced/ 已存在时自动跳过）

STEP 2.5 掩码图像参数:
  --target_classes <ids>  保留的 YOLO class_id，空格分隔 (默认: "3")
  --keep_all_classes      保留所有检测到的类别
  --bg_color <B G R>      背景填充色 BGR 格式 (默认: 255 255 255 白色)
  --dilate_pixels <int>   掩码边缘膨胀像素数 (默认: 3)
  --skip_mask_images      跳过 STEP 2.5

STEP 5 匹配参数:
  --max_depth       <float> 深度上限（米），不填则自动检测
  --scene_hard_cap  <float> 场景深度硬性上限 (默认: 15.0 m)
  --min_yolo_conf   <float> YOLO 置信度过滤门槛 (默认: 0.3)
  --min_views       <int>   全局 ID 最少出现帧数 (默认: 2)
  --no_appearance         关闭外观相似性证据（调试用）

断点续跑控制:
  --skip_vggsfm       跳过 STEP 1   (VGGSfM)
  --skip_enhance      跳过 STEP 1.5 (RealBasicVSR)
  --skip_yolo         跳过 STEP 2   (YOLO)
  --skip_mask_images  跳过 STEP 2.5 (掩码图像生成)
  --skip_train        跳过 STEP 3   (2DGS训练)
  --skip_depth        跳过 STEP 4   (深度图渲染)
  --skip_match        跳过 STEP 5   (3D匹配)
  --skip_render       跳过 STEP 6   (最终渲染)

示例:
  # 全新完整处理（视频输入）
  bash pipeline_enhanced.sh --video /data/corn.mp4

  # 已有抽帧，只跑增强 + 后续步骤
  bash pipeline_enhanced.sh --scene_dir /data/corn_frames \\
      --skip_vggsfm

  # 增强已完成，从 YOLO 开始
  bash pipeline_enhanced.sh --scene_dir /data/corn_frames \\
      --skip_vggsfm --skip_enhance

  # 只重跑匹配（3DGS 已训练完成）
  bash pipeline_enhanced.sh --scene_dir /data/corn_frames \\
      --skip_vggsfm --skip_enhance --skip_yolo \\
      --skip_mask_images --skip_train --skip_depth

  # 传递额外 RealBasicVSR 参数
  bash pipeline_enhanced.sh --video /data/corn.mp4 \\
      --rbvsr_args "--scale 4"
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --video)             VIDEO="$2";              shift 2 ;;
        --scene_dir)         SCENE_DIR="$2";          shift 2 ;;
        # ★ RealBasicVSR
        --rbvsr_dir)         RBVSR_DIR="$2";          shift 2 ;;
        --rbvsr_config)      RBVSR_CONFIG="$2";       shift 2 ;;
        --rbvsr_checkpoint)  RBVSR_CHECKPOINT="$2";  shift 2 ;;
        --rbvsr_max_seq_len) RBVSR_MAX_SEQ_LEN="$2"; shift 2 ;;
        --rbvsr_save_png)    RBVSR_SAVE_PNG="$2";    shift 2 ;;
        --skip_enhance)      SKIP_ENHANCE=true;       shift ;;
        # STEP 2.5
        --target_classes)    MASK_TARGET_CLASSES="$2"; shift 2 ;;
        --keep_all_classes)  MASK_KEEP_ALL=true;      shift ;;
        --bg_color)          MASK_BG_COLOR="$2 $3 $4"; shift 4 ;;
        --dilate_pixels)     MASK_DILATE="$2";        shift 2 ;;
        --skip_mask_images)  SKIP_MASK_IMAGES=true;   shift ;;
        # STEP 5
        --max_depth)         MAX_DEPTH="$2";          shift 2 ;;
        --scene_hard_cap)    SCENE_HARD_CAP="$2";     shift 2 ;;
        --min_yolo_conf)     MIN_YOLO_CONF="$2";      shift 2 ;;
        --min_views)         MIN_VIEWS="$2";          shift 2 ;;
        --no_appearance)     NO_APPEARANCE=true;      shift ;;
        # 通用
        --fps)               VIDEO_FPS="$2";          shift 2 ;;
        --yolo_conf)         YOLO_CONF="$2";          shift 2 ;;
        --iterations)        GS_ITERATIONS="$2";      shift 2 ;;
        # 断点续跑
        --skip_vggsfm)       SKIP_VGGSFM=true;        shift ;;
        --skip_yolo)         SKIP_YOLO=true;          shift ;;
        --skip_train)        SKIP_TRAIN=true;         shift ;;
        --skip_depth)        SKIP_DEPTH=true;         shift ;;
        --skip_match)        SKIP_MATCH=true;         shift ;;
        --skip_render)       SKIP_RENDER=true;        shift ;;
        -h|--help)           usage ;;
        *) log_err "未知参数: $1（使用 -h 查看帮助）" ;;
    esac
done

# ── 路径推断与创建 ────────────────────────────────────────────
[[ -z "$VIDEO" && -z "$SCENE_DIR" ]] && log_err "请提供 --video 或 --scene_dir"

if [[ -n "$VIDEO" && -z "$SCENE_DIR" ]]; then
    VIDEO_ABS="$(realpath "$VIDEO")"
    SCENE_NAME="$(basename "$VIDEO_ABS" .mp4)"
    DIR_NAME="$(basename "$(dirname "$VIDEO_ABS")")"
    MAIN_DIR="$([ "$DIR_NAME" = "video" ] && \
        dirname "$(dirname "$VIDEO_ABS")" || dirname "$VIDEO_ABS")"
    SCENE_DIR="${MAIN_DIR}/${SCENE_NAME}"
fi

SCENE_DIR="$(realpath -m "$SCENE_DIR")"
mkdir -p "$SCENE_DIR"

# ── 校验脚本文件 ──────────────────────────────────────────────
for f in "vggsfm_cli.py" "realbasicvsr_cli.py" "reason_cli.py" \
         "create_masked_images.py" "train_0211之前.py" \
         "pirender深度图.py" "run_matchingljp0326.py" "pirenderljp0326.py"; do
    [[ -f "${SCRIPT_DIR}/$f" ]] || log_err "在 ${SCRIPT_DIR} 找不到脚本: $f"
done
[[ -f "$YOLO_MODEL" ]] || log_err "YOLO 模型不存在: $YOLO_MODEL"
[[ -d "$RBVSR_DIR"  ]] || log_err "RealBasicVSR 目录不存在: $RBVSR_DIR"

# =============================================================================
# ██  5. 全局日志  ██
# =============================================================================
LOG_FILE="${SCENE_DIR}/pipeline_enhanced_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║  3D Semantic Pipeline  v5 (RealBasicVSR 增强版)      ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${NC}"
printf "  %-24s: %s\n" "场景目录"           "$SCENE_DIR"
printf "  %-24s: %s\n" "输入视频"           "${VIDEO:-[使用已有场景目录]}"
printf "  %-24s: %s\n" "RealBasicVSR目录"   "$RBVSR_DIR"
printf "  %-24s: %s\n" "YOLO分割图像来源"   "images_enhanced/"
printf "  %-24s: %s\n" "3DGS训练图像来源"   "masked_images/ (增强+掩码)"
printf "  %-24s: %s\n" "深度匹配点云来源"   "3DGS output_*/ 点云"
printf "  %-24s: %s\n" "掩码类别"           "$( $MASK_KEEP_ALL && echo '全部' || echo "class $MASK_TARGET_CLASSES")"
printf "  %-24s: %s\n" "max_depth"          "${MAX_DEPTH:-自动检测}"
printf "  %-24s: %s\n" "scene_hard_cap"     "${SCENE_HARD_CAP}m"
printf "  %-24s: %s\n" "日志文件"           "$LOG_FILE"
echo ""

PIPELINE_START=$(date +%s)
elapsed() { local s=$(( $(date +%s) - $1 )); printf "%dm%02ds" $(( s/60 )) $(( s%60 )); }

# =============================================================================
# STEP 1: VGGSfM 视频抽帧 + 稀疏重建（使用原始 images/）
# =============================================================================
log_step 1 "VGGSfM 视频抽帧 + 稀疏重建（原始图像 → 相机位姿）"
log_env "$ENV_VGGSFM"
log_info "输入: 原始视频 | 输出: images/ + sparse/0/"
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
log_info "原始图像总帧数: ${N_FRAMES}"

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
# ★ STEP 1.5: RealBasicVSR 图像超分辨率增强
# =============================================================================
log_new "1.5" "RealBasicVSR 图像增强（images/ → images_enhanced/）"
log_env "$ENV_RBVSR"
log_info "输入: images/（原始帧）| 输出: images_enhanced/（增强帧）"
log_info "后续 YOLO 和 3DGS 均使用增强图像，相机位姿仍来自 images/ 的 sparse/0/"
T15=$(date +%s)

ENHANCED_DIR="${SCENE_DIR}/images_enhanced"

if $SKIP_ENHANCE; then
    log_skip "RealBasicVSR 增强 (--skip_enhance)"
else
    # 检查是否已完成（增强帧数 >= 原始帧数）
    N_ENHANCED=$(find "$ENHANCED_DIR" \
        \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l || echo 0)

    if [[ "$N_ENHANCED" -ge "$N_FRAMES" && "$N_ENHANCED" -gt 0 ]]; then
        log_skip "images_enhanced/ 已完整 ($N_ENHANCED 张)"
    else
        ENHANCE_CMD=(
            "${SCRIPT_DIR}/realbasicvsr_cli.py"
            "--scene_dir" "$SCENE_DIR"
            "--rbvsr_dir" "$RBVSR_DIR"
        )

        # 传入 config（不填则 realbasicvsr_cli.py 自动查找）
        if [[ -n "$RBVSR_CONFIG" ]]; then
            ENHANCE_CMD+=("--config" "$RBVSR_CONFIG")
            log_info "RealBasicVSR config: $RBVSR_CONFIG"
        fi

        # 传入 checkpoint（不填则 realbasicvsr_cli.py 自动查找）
        if [[ -n "$RBVSR_CHECKPOINT" ]]; then
            ENHANCE_CMD+=("--checkpoint" "$RBVSR_CHECKPOINT")
            log_info "RealBasicVSR checkpoint: $RBVSR_CHECKPOINT"
        fi

        # 可选参数
        [[ -n "$RBVSR_MAX_SEQ_LEN" ]] && ENHANCE_CMD+=("--max_seq_len" "$RBVSR_MAX_SEQ_LEN")
        [[ -n "$RBVSR_SAVE_PNG"    ]] && ENHANCE_CMD+=("--is_save_as_png" "$RBVSR_SAVE_PNG")

        RUN_RBVSR "${ENHANCE_CMD[@]}"

        # 验证输出
        N_ENHANCED=$(find "$ENHANCED_DIR" \
            \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l || echo 0)
        log_info "增强完成，共 ${N_ENHANCED} 张增强图像"

        if [[ "$N_ENHANCED" -lt "$N_FRAMES" ]]; then
            log_warn "增强帧数(${N_ENHANCED}) < 原始帧数(${N_FRAMES})，请检查"
        fi
    fi
fi

# 更新实际使用的帧数（以增强目录为准）
N_FRAMES_ENHANCED=$(find "$ENHANCED_DIR" \
    \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l || echo "$N_FRAMES")
log_info "增强图像帧数: ${N_FRAMES_ENHANCED}"
log_ok "STEP 1.5 完成 (用时 $(elapsed $T15))"

# =============================================================================
# STEP 2: YOLO 2D 实例分割（使用增强图像 images_enhanced/）
# =============================================================================
log_step 2 "YOLO 2D 实例分割（输入: images_enhanced/）"
log_env "$ENV_GS"
log_info "★ 使用增强图像进行分割，提升检测精度"
T2=$(date +%s)

MASK_DIR="${SCENE_DIR}/masks_results/integer_masks"

if $SKIP_YOLO; then
    log_skip "YOLO (--skip_yolo)"
else
    EXISTING_MASKS=$(find "$MASK_DIR" -name "mask_*.png" 2>/dev/null | wc -l || echo 0)
    if [[ "$EXISTING_MASKS" -ge "$N_FRAMES_ENHANCED" && "$EXISTING_MASKS" -gt 0 ]]; then
        log_skip "掩码已完整 ($EXISTING_MASKS 帧)"
    else
        RUN_GS "${SCRIPT_DIR}/reason_cli.py" \
            --scene_dir    "$SCENE_DIR" \
            --model        "$YOLO_MODEL" \
            --conf         "$YOLO_CONF" \
            --images_subdir "images_enhanced" \
            --save_viz
    fi
fi
log_ok "STEP 2 完成 (用时 $(elapsed $T2))"

# =============================================================================
# STEP 2.5: 生成掩码训练图像（增强图像 + YOLO掩码 → masked_images/）
# =============================================================================
log_step "2.5" "生成掩码训练图像（images_enhanced/ + masks → masked_images/）"
log_env "$ENV_GS"
log_info "★ 源图像使用增强帧，掩码来自 YOLO 分割结果"
T25=$(date +%s)

MASKED_IMAGES_DIR="${SCENE_DIR}/masked_images"

if $SKIP_MASK_IMAGES; then
    log_skip "掩码图像生成 (--skip_mask_images)"
else
    EXISTING_MASKED=$(find "$MASKED_IMAGES_DIR" \
        \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l || echo 0)

    if [[ "$EXISTING_MASKED" -ge "$N_FRAMES_ENHANCED" && "$EXISTING_MASKED" -gt 0 ]]; then
        log_skip "masked_images/ 已完整 ($EXISTING_MASKED 张)"
    else
        MASK_CMD=(
            "${SCRIPT_DIR}/create_masked_images.py"
            "$SCENE_DIR"
            "--images_subdir"  "images_enhanced"   # ★ 使用增强图像
            "--bg_color"       $MASK_BG_COLOR
            "--dilate_pixels"  "$MASK_DILATE"
            "--output_dir_name" "masked_images"
            "--verify"
        )

        if $MASK_KEEP_ALL; then
            MASK_CMD+=("--keep_all_classes")
            log_info "保留所有 YOLO 检测类别"
        else
            # shellcheck disable=SC2086
            MASK_CMD+=("--target_classes" $MASK_TARGET_CLASSES)
            log_info "保留类别: $MASK_TARGET_CLASSES  (0=茎 1=根 2=叶鞘 3=叶片)"
        fi

        RUN_GS "${MASK_CMD[@]}"

        GENERATED=$(find "$MASKED_IMAGES_DIR" \
            \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l || echo 0)
        log_info "生成掩码图像: ${GENERATED} 张"
        [[ "$GENERATED" -lt "$N_FRAMES_ENHANCED" ]] && \
            log_warn "掩码图像数(${GENERATED}) < 增强帧数(${N_FRAMES_ENHANCED})，请检查"
    fi
fi
log_ok "STEP 2.5 完成 (用时 $(elapsed $T25))"

# =============================================================================
# STEP 3: 2D-GS 训练（masked_images/ + sparse/0/ → output_*/）
# =============================================================================
log_step 3 "2D-GS 高斯重建 (iter=$GS_ITERATIONS)"
log_env "$ENV_GS"
log_info "训练图像: masked_images/（增强+掩码）| 相机位姿: sparse/0/"
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
        RUN_GS "${SCRIPT_DIR}/train_0211之前.py" \
            --source_path "$SCENE_DIR" \
            -m            "$OUTPUT_DIR" \
            --iterations  "$GS_ITERATIONS" \
            --images      "masked_images"
    fi
fi
log_ok "STEP 3 完成 (用时 $(elapsed $T3))"

# =============================================================================
# STEP 4: 深度图渲染（从 3DGS 点云渲染，不依赖稀疏点云）
# =============================================================================
log_step 4 "渲染全部深度图（来源: 3DGS 点云）"
log_env "$ENV_GS"
log_info "★ 深度图由 3DGS 渲染，质量优于稀疏点云投影"
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
# STEP 5: 3D 语义匹配（深度 + YOLO掩码 + 3DGS点云坐标）
# =============================================================================
log_step 5 "3D 语义匹配（3DGS点云 + YOLO掩码 → 全局实例ID）"
log_env "$ENV_GS"
log_info "★ 匹配使用 3DGS 渲染的稠密深度图，精度高于稀疏点云"
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

        $NO_APPEARANCE && MATCH_CMD+=("--no_appearance") && \
            log_warn "外观相似性已关闭（调试模式）"

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
echo -e "${BOLD}${GREEN}║   全流程完成 🎉  (v5 RealBasicVSR 增强版)           ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
printf "  %-26s: %s\n" "总耗时"             "$(printf "%dm%02ds" $(( TOTAL_SEC/60 )) $(( TOTAL_SEC%60 )))"
printf "  %-26s: %s\n" "场景目录"           "$SCENE_DIR"
printf "  %-26s: %s\n" "原始图像"           "${SCENE_DIR}/images/"
printf "  %-26s: %s\n" "★增强图像"          "${SCENE_DIR}/images_enhanced/"
printf "  %-26s: %s\n" "YOLO掩码"           "${SCENE_DIR}/masks_results/"
printf "  %-26s: %s\n" "训练图像(增强+掩码)" "${SCENE_DIR}/masked_images/"
printf "  %-26s: %s\n" "3DGS输出+深度图"    "$OUTPUT_DIR"
printf "  %-26s: %s\n" "3D匹配输出"         "$MATCH_DIR"
printf "  %-26s: %s\n" "噪声ID清单"         "${MATCH_DIR}/noise_ids_removed.txt"
echo -e "\n${CYAN}日志: ${LOG_FILE}${NC}"

echo ""
echo -e "${BOLD}${CYAN}数据流向总结:${NC}"
echo -e "  images/ ──────────────────────────────→ VGGSfM → sparse/0/ (相机位姿)"
echo -e "  images/ → ${BOLD}RealBasicVSR${NC} → images_enhanced/"
echo -e "  images_enhanced/ → YOLO → masks_results/"
echo -e "  images_enhanced/ + masks → masked_images/"
echo -e "  masked_images/ + sparse/0/ → 3DGS → output_*/ (点云+深度图)"
echo -e "  output_*/ + masks → 3D匹配 → 数据驱动匹配/ (全局语义ID)"
echo -e "  数据驱动匹配/ → 最终渲染 → ${BOLD}语义点云${NC}"
