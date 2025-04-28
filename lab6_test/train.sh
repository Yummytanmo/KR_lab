#!/bin/bash
#SBATCH -J train                               # 作业名为 test
#SBATCH -o train.out                           # stdout 重定向到 test.out
#SBATCH -e train.err                           # stderr 重定向到 test.err
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 5:00:00
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1

. $HOME/soft/miniconda3/etc/profile.d/conda.sh
conda activate base

python train.py > ./train.log 2>&1 