#!/bin/bash
#SBATCH -p hl_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:15:00
#SBATCH -J fix_bertscore
#SBATCH -o /home/user/MLHC_AgenticSUM/outputs/agenticsum/fix_bertscore_%j.log
#SBATCH -e /home/user/MLHC_AgenticSUM/outputs/agenticsum/fix_bertscore_%j.err

export HF_HOME=/home/user/.cache/huggingface
export HUGGINGFACE_HUB_TOKEN=$(cat $HF_HOME/token)

source $(conda info --base)/etc/profile.d/conda.sh
conda activate AgenticSUM_CHIL

cd /home/user/MLHC_AgenticSUM/AgenticSum/Agents

python fix_bertscore_mimic.py
