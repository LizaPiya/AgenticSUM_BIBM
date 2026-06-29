#!/bin/bash
#SBATCH -p hl_gpu_shared
#SBATCH --gres=gpu:blackwell:1
#SBATCH --licenses=gpu_slot:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J judge_abl_mimic
#SBATCH -o /home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum/judge_ablation_mimic_%j.log
#SBATCH -e /home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum/judge_ablation_mimic_%j.err

export HF_HOME=/home/lizapiya/.cache/huggingface
export HUGGINGFACE_HUB_TOKEN=$(cat $HF_HOME/token)
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p /home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum

source $(conda info --base)/etc/profile.d/conda.sh
conda activate AgenticSUM_CHIL

cd /home/lizapiya/MLHC_AgenticSUM/AgenticSum/Agents

python run_judge_ablation_mimic.py
