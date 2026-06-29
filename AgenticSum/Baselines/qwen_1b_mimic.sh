#!/bin/bash
#SBATCH --job-name=qwen1b_mimic
#SBATCH --partition=hl_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum/qwen1b_mimic_%j.log
#SBATCH --error=/home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum/qwen1b_mimic_%j.err

source /home/lizapiya/miniconda3/etc/profile.d/conda.sh
conda activate AgenticSUM_CHIL

cd /home/lizapiya/MLHC_AgenticSUM/AgenticSum/Baselines

python run_qwen_mimic.py --model_size 1b
