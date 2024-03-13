set -e
export TOKENIZERS_PARALLELISM=false

for train_digit_dist in constant zipf:3.5 poisson:3; do
    CUDA_VISIBLE_DEVICES=1 python run.py \
        --out_dir='out/priming/gpt-nano-reverse-n_digit-3-dist-'$train_digit_dist \
        --ckpt_name='ckpt_5000.pt' \
        --wandb_run_name='gpt-nano-reverse-n_digit-3-dist-'$train_digit_dist \
        --wandb_project='priming' \
        --train_digit_dist=$train_digit_dist \
        --n_digit=3 \
        --max_iters=20000 \
        --num_test=1000 \
        --eval_interval=1000 \
        --resume=True \
        --always_save_checkpoint=True \
        --wandb_log=True \
        --do_length_gen_eval=True \
        --min_eval_len=2 \
        --max_eval_len=10
done
