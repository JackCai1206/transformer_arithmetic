set -e
export TOKENIZERS_PARALLELISM=false

for n_digit in 5; do
    for n_layer in 1; do
        for n_head in 3; do
            CUDA_VISIBLE_DEVICES=0 python run.py \
                --wandb_run_name='reverse-n_digit-'$n_digit'-n_layer-'$n_layer'-n_head-'$n_head \
                --wandb_project='tiny_models' \
                --wandb_group='baseline-exp1' \
                --resume=True \
                --n_digit=$n_digit \
                --max_iters=20000 \
                --num_train=10000 \
                --num_test=1000 \
                --eval_interval=250 \
                --wandb_log=True \
                --online=True \
                --save_by_iter=True \
                --do_per_digit_eval=True \
                --do_carry_type_eval=True \
                --n_layer=$n_layer \
                --n_head=$n_head
        done
    done
done
