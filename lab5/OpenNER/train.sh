#!/bin/bash
#SBATCH -J test                               # 作业名为 test
#SBATCH -o test.out                           # stdout 重定向到 test.out
#SBATCH -e test.err                           # stderr 重定向到 test.err
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 20:00:00                            # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1          # 申请 4 卡 A100 80GB，如果只申请CPU可以删除本行

# 设置运行环境
. $HOME/soft/miniconda3/etc/profile.d/conda.sh
conda activate lab

python example/train_bag_cnn.py \
    --metric auc \
    --dataset nyt10m \
    --batch_size 4000 \
    --lr 0.1 \
    --weight_decay 1e-5 \
    --max_epoch 100 \
    --max_length 128 \
    --seed 42 \
    --encoder cnn \
    --aggr avg\
    > train_cnn_avg.log 2>&1