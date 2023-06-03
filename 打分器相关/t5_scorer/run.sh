#!/bin/bash


# # activate your conda virtual environment
source activate lighting
export CUDA_VISIBLE_DEVICES="0,1"
export TOKENIZERS_PARALLELISM=true
# google/t5-v1_1-base --fast_dev_run --debug  ./flan-t5-base #SBATCH -w node01

# darmstadt_unis
for seed in 4234 #2234 3234 4234 5234
do
    python main.py --task asqp \
                --seed $seed \
                --dataset ds \
                --model_name_or_path /root/data3/t5-base \
                --n_gpu 1 \
                --do_direct_eval \
                --train_batch_size 16 \
                --gradient_accumulation_steps 3 \
                --eval_batch_size 1 \
                --learning_rate 2e-4 \
                --num_train_epochs 20 \
                --num_samples 2 
                # --do_train

    echo ""
    date
done