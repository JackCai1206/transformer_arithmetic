set -e
export TOKENIZERS_PARALLELISM=false

for n_digit in 10; do
    for trial in {1..3}; do
        CUDA_VISIBLE_DEVICES=0 python run.py \
            --wandb_run_name='gpt-nano-reverse-n_digit-'$n_digit'-trial-'$trial \
            --wandb_project='baseline' \
            --wandb_group='baseline-exp1' \
            --n_digit=$n_digit \
            --max_iters=8000 \
            --num_train=10000 \
            --num_test=1000 \
            --eval_interval=250 \
            --wandb_log=True \
            --online=False \
            --save_by_iter=True \
            --resume=False \
            --do_per_digit_eval=True \
            --do_carry_type_eval=True 
    done
done

for n_digit in 10; do
    for trial in {1..3}; do
        CUDA_VISIBLE_DEVICES=0 python run.py \
            --wandb_run_name='gpt2-reverse-n_digit-'$n_digit'-trial-'$trial \
            --wandb_project='baseline' \
            --n_digit=$n_digit \
            --max_iters=8000 \
            --num_test=2000 \
            --eval_interval=250 \
            --wandb_log=True \
            --n_layer=12 \
            --n_head=12 \
            --n_embd=768 \
            --batch_size=64 \
            --eval_batch_size=1024 \
            --gradient_accumulation_steps=2 \
            --learning_rate=5e-4 \
            --online=False \
            --resume=False \
            --do_per_digit_eval=True \
            --do_carry_type_eval=True 
    done
done
