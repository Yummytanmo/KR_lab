#!/bin/bash

# Define a list of parameter sets
window=3
vector_size=16
sg=1
epochs=2
seed=42
mkdir -p log
# Loop through the parameter sets and run the script
for vs in "${vector_size[@]}"; do
    for ep in "${epochs[@]}"; do
        output_file="log/window_${window}_vs_${vs}_sg_${sg}_ep_${ep}_seed_${seed}.txt"
        echo "Running with window=$window, vector_size=$vs, sg=$sg, epochs=$ep, seed=$seed" > "$output_file"
        python similar.py --window $window --vector_size $vs --sg $sg --epochs $ep --seed $seed >> "$output_file"
        echo "Done running with window=$window, vector_size=$vs, sg=$sg, epochs=$ep, seed=$seed"
    done
done