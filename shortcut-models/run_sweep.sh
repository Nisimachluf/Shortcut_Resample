#!/bin/bash

DTS_VALUES=(128 64 32 16 8)
LR_FACTORS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

for dts in "${DTS_VALUES[@]}"; do
    for lr_factor in "${LR_FACTORS[@]}"; do
        run_name="${dts}_steps_lr_fact_${lr_factor}"
        echo "Running: dts=${dts}, lr_factor=${lr_factor}, run_name=${run_name}"
        python3 restore_images.py \
            --dts "$dts" \
            --lr_factor "$lr_factor" \
            --run_name "$run_name" \
            --num_images 100 
    done
done
