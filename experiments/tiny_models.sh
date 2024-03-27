set -e
export TOKENIZERS_PARALLELISM=false

for n_digit in 10 20; do
    for n_embd in 24 48 72 96; do
        for n_head in 1; do
            for n_layer in 1 2 3 4 5; do
                for trial in 1; do
                    CUDA_VISIBLE_DEVICES=1 python run.py \
                        --wandb_run_name='reverse-binary-n_digit-'$n_digit'-n_embd-'$n_embd'-n_layer-'$n_layer'-n_head-'$n_head'-trial-'$trial \
                        --wandb_project='tiny_models' \
                        --wandb_group='baseline-exp1' \
                        --resume=False \
                        --data_format='binary' \
                        --n_digit=$n_digit \
                        --max_iters=20000 \
                        --num_train=10000 \
                        --num_test=1000 \
                        --eval_interval=250 \
                        --wandb_log=True \
                        --online=True \
                        --save_by_iter=False \
                        --do_per_digit_eval=False \
                        --do_carry_type_eval=False \
                        --n_layer=$n_layer \
                        --n_head=$n_head  \
                        --n_embd=$n_embd \
                        --ary=2 \
                        --use_hookedtransformer=True \
                        --check_leak=False
                done
            done
        done
    done
done
