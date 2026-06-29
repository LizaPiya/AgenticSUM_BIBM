#!/bin/bash
#SBATCH -p hl_gpu_shared
#SBATCH --gres=gpu:blackwell:1
#SBATCH --licenses=gpu_slot:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH -J judge_soap
#SBATCH -o /home/user/MLHC_AgenticSUM/outputs/agenticsum/judge_soap_%j.log
#SBATCH -e /home/user/MLHC_AgenticSUM/outputs/agenticsum/judge_soap_%j.err

export HF_HOME=/home/user/.cache/huggingface
export HUGGINGFACE_HUB_TOKEN=$(cat $HF_HOME/token)
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p /home/user/MLHC_AgenticSUM/outputs/agenticsum

source $(conda info --base)/etc/profile.d/conda.sh
conda activate AgenticSUM_CHIL

cd /home/user/MLHC_AgenticSUM/AgenticSum/Agents

python run_judge_soap.py
