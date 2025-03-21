#!/bin/bash
#SBATCH -J test                               # 作业名为 test
#SBATCH -o test.out                           # stdout 重定向到 test.out
#SBATCH -e test.err                           # stderr 重定向到 test.err
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 2:00:00                            # 任务运行的最长时间为 1 小时
#SBATCH -w gpu06                              # 指定运行作业的节点是 gpu06，若不填写系统自动分配节点
#SBATCH --gres=tesla_v100-sxm2-16gb:1           # 申请 4 卡 A100 80GB，如果只申请CPU可以删除本行


# 输入要执行的命令，例如 ./hello 或 python test.py 等
python -V                    # 执行命令