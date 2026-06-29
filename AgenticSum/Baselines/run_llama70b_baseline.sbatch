#!/bin/bash
#SBATCH -p hl_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 10:00:00
#SBATCH -J llama70b_baseline
#SBATCH -o /home/user/MLHC_AgenticSUM/outputs/agenticsum/llama70b_%j.log
#SBATCH -e /home/user/MLHC_AgenticSUM/outputs/agenticsum/llama70b_%j.err

# -------------------------------
# Hugging Face cache + auth
# -------------------------------
export HF_HOME=/home/user/.cache/huggingface
export HUGGINGFACE_HUB_TOKEN=$(cat $HF_HOME/token)
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p /home/user/MLHC_AgenticSUM/outputs/agenticsum/llama70b_baseline

# ===============================
# Environment (SBATCH-SAFE)
# ===============================
source $(conda info --base)/etc/profile.d/conda.sh
conda activate AgenticSUM_CHIL

# ===============================
# Run Llama-3.1-70B single-pass baseline
# ===============================
cd /home/user/MLHC_AgenticSUM/AgenticSum/Baselines

python run_llama70b_baseline.py
