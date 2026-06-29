#!/bin/bash
#SBATCH -p hl_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J mistral7b_soap
#SBATCH -o /home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum/mistral7b_soap_%j.log
#SBATCH -e /home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum/mistral7b_soap_%j.err

# -------------------------------
# Hugging Face cache + auth
# -------------------------------
export HF_HOME=/home/lizapiya/.cache/huggingface
export HUGGINGFACE_HUB_TOKEN=$(cat $HF_HOME/token)
export HF_HUB_DISABLE_TELEMETRY=1

# OOM fix recommended by PyTorch error message
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===============================
# Environment (SBATCH-SAFE)
# ===============================
source $(conda info --base)/etc/profile.d/conda.sh
conda activate AgenticSUM_CHIL

# ===============================
# Run AgenticSum Mistral-7B — SOAP only
# ===============================
cd /home/lizapiya/MLHC_AgenticSUM/AgenticSum/Agents

python run_mistral7b_agenticsum_soap.py
