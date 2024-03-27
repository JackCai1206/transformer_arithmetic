set -e

for n_digit in 5; do
    for trial in 1; do
        CUDA_VISIBLE_DEVICES=0 python run.py \
            --wandb_run_name='gpt-nano-waves-n_digit-'$n_digit'-trial-'$trial \
            --wandb_project='waves' \
            --wandb_group='waves-exp1' \
            --dataset='waves' \
            --n_digit=$n_digit \
            --batch_size=128 \
            --max_iters=10000 \
            --num_train=10000 \
            --num_test=1000 \
            --eval_interval=250 \
            --wandb_log=True \
            --online=True \
            --save_by_iter=False \
            --resume=False \
            --do_per_digit_eval=False \
            --do_carry_type_eval=False \
            --check_leak=False \
            --use_hookedtransformer=True \
            --pos_emb_type=alibi \
            --ary=80
    done
done
