#!/bin/bash
#SBATCH -p gpu-v100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 10:00:00
#SBATCH -J agenticsum_test
#SBATCH -o /lustre/hl/users/4283/outputs/agenticsum/logs/run_%j.log
#SBATCH -e /lustre/hl/users/4283/outputs/agenticsum/logs/run_%j.err

# -------------------------------
# Hugging Face cache + auth
# -------------------------------
export HF_HOME=/lustre/hl/users/4283/.cache/huggingface
export HUGGINGFACE_HUB_TOKEN=$(cat $HF_HOME/token)
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p $HF_HOME

# ===============================
# Environment (SBATCH-SAFE)
# ===============================
vpkg_devrequire anaconda/2024.02:python3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate agentic

# ===============================
# Run AgenticSum
# ===============================
cd /home/lizapiya/MLHC_AgenticSUM/AgenticSum/Agents

python run_agenticsum.py
