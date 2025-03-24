#!/bin/bash
#SBATCH -J note                               # 作业名为 test
#SBATCH -o note.out                           # stdout 重定向到 test.out
#SBATCH -e note.err                           # stderr 重定向到 test.err
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 3:00:00
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1

. $HOME/soft/miniconda3/etc/profile.d/conda.sh
conda activate LF
jupyter notebook --ip=0.0.0.0 --port=1234 --no-browser > notebook.log 2>&1