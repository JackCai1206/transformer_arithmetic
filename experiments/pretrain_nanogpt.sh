set -e
export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0 python run.py \
    --wandb_run_name='pt-gpt-nano-openwebtext' \
    --wandb_project='pretrain-gpt-nano' \
    --dataset='openwebtext' \
    --max_iters=20000 \
    --num_test=1000 \
    --eval_interval=250 \
    --do_eval=False \
    --wandb_log=False
