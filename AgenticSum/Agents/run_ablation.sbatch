#!/bin/bash
#SBATCH -p hl_gpu_shared
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -J ablation_5docs
#SBATCH -o /home/user/MLHC_AgenticSUM/outputs/agenticsum/logs/ablation_%j.log
#SBATCH -e /home/user/MLHC_AgenticSUM/outputs/agenticsum/logs/ablation_%j.err

mkdir -p /home/user/MLHC_AgenticSUM/outputs/agenticsum/logs

# ===============================
# Environment
# ===============================
source $(conda info --base)/etc/profile.d/conda.sh
conda activate AgenticSUM_CHIL

# ===============================
# Run ablation
# ===============================
cd /home/user/MLHC_AgenticSUM/AgenticSum/Agents

python run_ablation.py
