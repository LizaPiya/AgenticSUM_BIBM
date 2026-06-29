#!/bin/bash
#SBATCH -p hl_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 24:00:00
#SBATCH -J agenticsum_70b
#SBATCH -o /home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum/agenticsum70b_%j.log
#SBATCH -e /home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum/agenticsum70b_%j.err

# -------------------------------
# Hugging Face cache + auth
# -------------------------------
export HF_HOME=/home/lizapiya/.cache/huggingface
export HUGGINGFACE_HUB_TOKEN=$(cat $HF_HOME/token)
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum/agenticsum_70b

# ===============================
# Environment (SBATCH-SAFE)
# ===============================
source $(conda info --base)/etc/profile.d/conda.sh
conda activate AgenticSUM_CHIL

# ===============================
# Run AgenticSum with 70B backbone
# ===============================
cd /home/lizapiya/MLHC_AgenticSUM/AgenticSum/Agents

python run_agenticsum_70b.py
