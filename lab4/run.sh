#!/bin/bash
#SBATCH -J train                               # 作业名为 test
#SBATCH -o train.out                           # stdout 重定向到 test.out
#SBATCH -e train.err                           # stderr 重定向到 test.err
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 6:00:00                            # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:tesla_v100-pcie-32gb:1          # 申请 4 卡 A100 80GB，如果只申请CPU可以删除本行

# 设置运行环境
. $HOME/soft/miniconda3/etc/profile.d/conda.sh
conda activate base

python main.py