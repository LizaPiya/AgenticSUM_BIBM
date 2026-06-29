#!/bin/bash
#SBATCH --job-name=qwen1b_soap
#SBATCH --partition=hl_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/user/MLHC_AgenticSUM/outputs/agenticsum/qwen1b_soap_%j.log
#SBATCH --error=/home/user/MLHC_AgenticSUM/outputs/agenticsum/qwen1b_soap_%j.err

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate AgenticSUM_CHIL

cd /home/user/MLHC_AgenticSUM/AgenticSum/Baselines

python run_qwen_soap.py --model_size 1b
