#!/bin/bash
#SBATCH -J work
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -t 72:00:00
#SBATCH -w node01

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
echo ""


# # activate your conda virtual environment
# --learning_rate 1e-4 \

source activate work
export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=0
# for dataset in ds mpqa
# do
# for mask in no holder_target expression polarity
# do
python main.py --debug --do_train \
        --dataset ds \
        --num_train_epochs 50 \
        --pretrain_model_path /root/data2/bert-base-uncased \
        --mask no \
        --learning_rate 9e-5
# done
# done