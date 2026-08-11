#!/usr/bin/env bash
# run_aaa_cp_lr.sh
# 完整流程：gather C → gather P → 左右并列 AAA 图
# 运行方法：bash run_aaa_cp_lr.sh
set -e
cd "$(dirname "$0")"

# ── Step 1: gather Imagenet-C ─────────────────────────────────────────────
echo "[1/3] gathering Imagenet-C ..."
python result_analyse_tool/gather_from_given_dirs.py \
  --out summaries/ImagenetC.xlsx \
  --inc 10 \
  --dataset Imagenet-C \
  --rank 16 \
  --plot_aaa_meanvar  --aaa_out_dir summaries/AAA_imagenetC_curves \
  --plot_nme_meanvar  --nme_out_dir summaries/NME_imagenetC_curves \
  --base_dirs \
  logs_inc_lora/olora/gam/tiny_imagenetc/42/olora_gam_imagenetc_t20_r8_42/exp_ablation \
  logs_inc_lora/olora/sam/tiny_imagenetc/1993/olora_sam_imagenetc_t20_r8_1993/exp_nme \
  logs_inc_lora/olora/sgd/tiny_imagenetc/1993/olora_sgd_imagenetc_t20_r8_1993/exp_nme \
  logs_inc_lora/seqlora/sgd/tiny_imagenetc/1993/seqlora_sgd_imagenetc_t20_r8_1993/exp_nme \
  logs_inc_lora/seqlora/sam/tiny_imagenetc/1993/seqlora_sam_imagenetc_t20_r8_1993/exp_nme \
  logs_inc_lora/seqlora/gam/tiny_imagenetc/42/seqlora_gam_imagenetc_t20_r8_42/exp_ablation \
  logs_inc_lora/inclora/sam/tiny_imagenetc/1993/inclora_sam_imagenetc_t20_r8_1993/exp_nme \
  logs_inc_lora/inclora/sgd/tiny_imagenetc/1993/inclora_sgd_imagenetc_t20_r8_1993/exp_nme \
  logs_inc_lora/inclora/gam/tiny_imagenetc/42/inclora_gam_imagenetc_t20_r8_42/exp_ablation

# ── Step 2: gather Imagenet-P ─────────────────────────────────────────────
echo "[2/3] gathering Imagenet-P ..."
python result_analyse_tool/gather_from_given_dirs.py \
  --out summaries/ImagenetP.xlsx \
  --inc 10 \
  --dataset Imagenet-P \
  --rank 16 \
  --plot_aaa_meanvar  --aaa_out_dir summaries/AAA_imagenetP_curves \
  --plot_nme_meanvar  --nme_out_dir summaries/NME_imagenetP_curves \
  --base_dirs \
  logs_inc_lora/inclora/sam/tiny_imagenetp/1993/inclora_sam_imagenetp_t20_r8_1993/exp_nme \
  logs_inc_lora/inclora/sgd/tiny_imagenetp/1993/inclora_sgd_imagenetp_t20_r8_1993/exp_nme \
  logs_inc_lora/seqlora/sam/tiny_imagenetp/1993/seqlora_sam_imagenetp_t20_r8_1993/exp_nme \
  logs_inc_lora/seqlora/sgd/tiny_imagenetp/1993/seqlora_sgd_imagenetp_t20_r8_1993/exp_nme \
  logs_inc_lora/olora/sam/tiny_imagenetp/1993/olora_sam_imagenetp_t20_r8_1993/exp_nme \
  logs_inc_lora/olora/sgd/tiny_imagenetp/1993/olora_sgd_imagenetp_t20_r8_1993/exp_nme \
  logs_inc_lora/seqlora/gam/tiny_imagenetp/1993/seqlora_gam_imagenetp_t20_r8_1993/exp_ablation \
  logs_inc_lora/olora/gam/tiny_imagenetp/1993/olora_gam_imagenetp_t20_r8_1993/exp_ablation \
  logs_inc_lora/inclora/gam/tiny_imagenetp/1993/inclora_gam_imagenetp_t20_r8_1993/exp_ablation

# ── Step 3: 左右并列 AAA 图 ────────────────────────────────────────────────
echo "[3/3] plotting left-right AAA figure ..."
python - <<'PYEOF'
import sys
sys.path.insert(0, ".")
from result_analyse_tool.visual_ImagenetCP import plot_aaa_task_curves_lr

plot_aaa_task_curves_lr(
    xlsx_left   = "summaries/ImagenetC.xlsx",
    xlsx_right  = "summaries/ImagenetP.xlsx",
    label_left  = "Imagenet-C",
    label_right = "Imagenet-P",
    out_path    = "summaries/AAA_imagenetCP_lr.png",
    figsize     = (8.6, 2.5),
)
PYEOF

echo ""
echo "Done. Output: summaries/AAA_imagenetCP_lr.png"
echo "              summaries/AAA_imagenetCP_lr.pdf"
