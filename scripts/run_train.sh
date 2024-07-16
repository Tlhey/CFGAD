#!/bin/bash

# 定义所有的异常检测方法
models=("AdONE" "ANOMALOUS" "AnomalyDAE" "CoLA" "CONAD" "DMGD" "DOMINANT" "DONE" "GAAN" "GADNR" "GAE" "GUIDE")

# 定义数据集路径
dataset_path="data/processed/bail_contextual.pkl"

# 定义结果输出目录
output_dir="results/bail"

# 创建结果输出目录
mkdir -p $output_dir

# 循环遍历所有模型并运行训练脚本
for model in "${models[@]}"; do
    echo "Running model: $model"
    python train.py --dataset $dataset_path --model $model --n_trials 20 --output_dir $output_dir
    echo "Finished running model: $model"
done

echo "All models have been trained and results are saved in the $output_dir directory."
