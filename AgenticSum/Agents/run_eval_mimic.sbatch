#!/bin/bash
#SBATCH -p hl_gpu_shared
#SBATCH --gres=gpu:blackwell:1
#SBATCH --licenses=gpu_slot:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J eval_mimic
#SBATCH -o /home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum/eval_mimic_%j.log
#SBATCH -e /home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum/eval_mimic_%j.err

export HF_HOME=/home/lizapiya/.cache/huggingface
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p /home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum

source $(conda info --base)/etc/profile.d/conda.sh
conda activate AgenticSUM_CHIL

cd /home/lizapiya/MLHC_AgenticSUM/AgenticSum/Agents

python run_eval_mimic.py
