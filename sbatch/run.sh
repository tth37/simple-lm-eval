#!/bin/bash
#SBATCH -J cluster-h800-llama2-7b-dense-boolq
#SBATCH -N 1
#SBATCH -p h01
#SBATCH -o sbatch-results/cluster-h800-llama2-7b-dense-boolq.out
#SBATCH -e sbatch-results/cluster-h800-llama2-7b-dense-boolq.err
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

source /home/fit/renju/WORK/tianhaodong/vllm-examples/.venv/bin/activate
cd /home/fit/renju/WORK/tianhaodong/vllm-examples/simple-lm-eval && python simple_lm_eval.py --backend transformers --model_name meta-llama/Llama-2-7b-hf --method dense --task boolq