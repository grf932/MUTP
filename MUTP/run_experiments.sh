#!/bin/bash

# Define arrays of different parameter values
caption_rates=(0)
dataset_types=("ds" "af")
seeds=(0 1 2 3 4)

# Loop through each combination of parameters and execute the Python script
for caption_rate in "${caption_rates[@]}"
do
    for dataset_type in "${dataset_types[@]}"
    do
        for seed in "${seeds[@]}"
        do
            output_path="./result/${dataset_type}/seed${seed}"
            echo "Running with caption_rate: ${caption_rate}, dataset_type: ${dataset_type}, seed: ${seed}"
            python my_main_kfold.py --data_path ./trafficnet_dataset_v1/ --input_path ./trafficnet_dataset_v1/embedding_inputs --caption_rate ${caption_rate} --seq_len 80 --dataset_type ${dataset_type} --seed ${seed} --output_path ${output_path} --num_epoch 50
        done
    done
done