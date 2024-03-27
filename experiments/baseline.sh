set -e
export TOKENIZERS_PARALLELISM=false

# for n_digit in 4; do
#     for trial in 1; do
#         for pos_emb_type in absolute; do
#             CUDA_VISIBLE_DEVICES=0 python run.py \
#                 --wandb_run_name='gpt-nano-reverse-n_digit-'$n_digit'-trial-'$trial'-pos_emb_type-'$pos_emb_type \
#                 --wandb_project='baseline' \
#                 --wandb_group='baseline-exp1' \
#                 --n_digit=$n_digit \
#                 --batch_size=32 \
#                 --max_iters=5000 \
#                 --num_train=10000 \
#                 --num_test=1000 \
#                 --eval_interval=250 \
#                 --wandb_log=False \
#                 --online=False \
#                 --save_by_iter=True \
#                 --resume=False \
#                 --do_per_digit_eval=False \
#                 --do_carry_type_eval=False \
#                 --use_hookedtransformer=True \
#                 --pos_emb_type=$pos_emb_type \
#                 --do_train=True \
#                 --resume=True \
#                 --do_length_gen_eval=True \
#                 --min_eval_len=2 \
#                 --max_eval_len=10
#         done
#     done
# done

# for n_digit in 3; do
#     for trial in 1; do
#         CUDA_VISIBLE_DEVICES=1 python run.py \
#             --wandb_run_name='gpt2-reverse-n_digit-'$n_digit'-trial-'$trial \
#             --wandb_project='baseline' \
#             --n_digit=$n_digit \
#             --max_iters=8000 \
#             --num_test=2000 \
#             --eval_interval=250 \
#             --wandb_log=True \
#             --n_layer=12 \
#             --n_head=12 \
#             --n_embd=768 \
#             --batch_size=64 \
#             --eval_batch_size=1024 \
#             --gradient_accumulation_steps=2 \
#             --learning_rate=5e-4 \
#             --online=False \
#             --resume=False \
#             --do_per_digit_eval=True \
#             --do_carry_type_eval=True 
#     done
# done

set -e
export TOKENIZERS_PARALLELISM=false

for n_digit in 10; do
    for trial in 1; do
        for pos_emb_type in standard; do
            CUDA_VISIBLE_DEVICES=0 python run.py \
                --wandb_run_name='gpt-nano-ft-only-posenc-reverse-n_digit-'$n_digit'-trial-'$trial'-pos_emb_type-'$pos_emb_type \
                --wandb_project='baseline' \
                --wandb_group='baseline-exp1' \
                --n_digit=$n_digit \
                --batch_size=32 \
                --max_iters=10000 \
                --num_train=10000 \
                --num_test=1000 \
                --eval_interval=250 \
                --wandb_log=True \
                --online=False \
                --save_by_iter=True \
                --resume=False \
                --do_per_digit_eval=False \
                --do_carry_type_eval=False \
                --use_hookedtransformer=True \
                --pos_emb_type=$pos_emb_type \
                --do_train=True \
                --resume=True \
                --do_length_gen_eval=False \
                --min_eval_len=2 \
                --max_eval_len=10 \
                --resume_dir='out/baseline/gpt-nano-reverse-n_digit-4-trial-1-pos_emb_type-absolute' \
                --freeze_except=[\'pos_embed\',\'ln\',\'_orig_mod.blocks.0\']
        done
    done
done
