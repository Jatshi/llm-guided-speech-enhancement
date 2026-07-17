#!/bin/bash
# =============================================================
# 全流程编排器：等待环境安装 + 模型下载完成 -> 搬移模型到数据盘(symlink)
# -> 生成数据 -> 构建训练数据 -> SFT -> DPO -> 评估。
# 每个阶段写入 pipeline_status.log，供 30 分钟定时检查读取进度。
# 用法：nohup bash run_pipeline.sh > outputs/logs/pipeline.log 2>&1 &
# =============================================================
set -u
P=/root/autodl-tmp/llm-speech-enhancement-v2
LOG=$P/outputs/logs
mkdir -p "$LOG"
STATUS=$LOG/pipeline_status.log
MODELDIR=/root/models/Qwen2.5-7B-Instruct
DEST=$P/models/Qwen2.5-7B-Instruct

source /root/miniconda3/etc/profile.d/conda.sh
conda activate llm-se-v2

st(){ echo "[$(date '+%F %T')] $1" | tee -a "$STATUS"; }

st "PIPELINE_START"

# ---------- 1. 等待依赖安装完成 ----------
st "STAGE=wait_install"
for i in $(seq 1 360); do          # 最多等 3 小时
  grep -q ALL_INSTALL_DONE "$LOG/install.log" 2>/dev/null && break
  sleep 30
done
if ! grep -q ALL_INSTALL_DONE "$LOG/install.log" 2>/dev/null; then
  st "ERROR=install_timeout"; exit 1
fi
st "install_done"

# ---------- 2. 等待模型下载完成 ----------
st "STAGE=wait_download"
for i in $(seq 1 720); do          # 最多等 6 小时
  inc=$(ls "$MODELDIR"/*.incomplete 2>/dev/null | wc -l)
  sh=$(ls "$MODELDIR"/model-0000*-of-00004.safetensors 2>/dev/null | wc -l)
  if [ "$inc" = "0" ] && [ "$sh" = "4" ] && [ -f "$MODELDIR/config.json" ] && [ -f "$MODELDIR/tokenizer.json" ]; then
    break
  fi
  sleep 30
done
sh=$(ls "$MODELDIR"/model-0000*-of-00004.safetensors 2>/dev/null | wc -l)
if [ "$sh" != "4" ] || [ ! -f "$MODELDIR/config.json" ]; then
  st "ERROR=download_incomplete shards=$sh"; exit 1
fi
st "download_done"

# ---------- 3. 搬移模型到数据盘 + 建立软链 ----------
st "STAGE=relocate_model"
if [ ! -L "$MODELDIR" ]; then
  mkdir -p "$P/models"
  mv "$MODELDIR" "$DEST"
  ln -s "$DEST" "$MODELDIR"
  st "model_relocated -> $DEST"
else
  st "model_already_symlinked"
fi
df -h / /root/autodl-tmp | tee -a "$STATUS"

# ---------- 4. 生成退化数据 ----------
st "STAGE=generate_data"
python "$P/src/generate_degraded_data.py" > "$LOG/gen_data.log" 2>&1
[ $? -ne 0 ] && { st "ERROR=gen_data"; tail -20 "$LOG/gen_data.log" | tee -a "$STATUS"; exit 1; }

st "STAGE=build_data"
python "$P/src/build_llm_data.py" > "$LOG/build_data.log" 2>&1
[ $? -ne 0 ] && { st "ERROR=build_data"; tail -20 "$LOG/build_data.log" | tee -a "$STATUS"; exit 1; }
st "data_ready"

# ---------- 5. SFT ----------
st "STAGE=sft_training"
python "$P/src/train_sft.py" > "$LOG/train_sft.log" 2>&1
[ $? -ne 0 ] && { st "ERROR=sft"; tail -30 "$LOG/train_sft.log" | tee -a "$STATUS"; exit 1; }
if [ ! -f "$P/outputs/sft/final/adapter_config.json" ]; then
  st "ERROR=sft_no_adapter"; exit 1
fi
st "sft_done"

# ---------- 6. DPO ----------
st "STAGE=dpo_training"
python "$P/src/train_dpo.py" > "$LOG/train_dpo.log" 2>&1
[ $? -ne 0 ] && { st "ERROR=dpo"; tail -30 "$LOG/train_dpo.log" | tee -a "$STATUS"; exit 1; }
if [ ! -f "$P/outputs/dpo/final/adapter_config.json" ]; then
  st "ERROR=dpo_no_adapter"; exit 1
fi
st "dpo_done"

# ---------- 7. 评估 ----------
st "STAGE=evaluate"
python "$P/src/evaluate.py" > "$LOG/evaluate.log" 2>&1
[ $? -ne 0 ] && { st "ERROR=eval"; tail -30 "$LOG/evaluate.log" | tee -a "$STATUS"; exit 1; }
st "eval_done"

st "PIPELINE_COMPLETE"
