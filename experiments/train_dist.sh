set -e
export TOKENIZERS_PARALLELISM=false

# for prob in 0.1; do
#     CUDA_VISIBLE_DEVICES=1 python run.py \
#         --wandb_run_name='gpt-nano-reverse-n_digit-3-prob8-'$prob \
#         --wandb_project='priming' \
#         --train_digit_dist='{4:'$prob',6:'$prob',8:'$prob',10:'$prob',12:'$prob'}' \
#         --n_digit=3 \
#         --max_iters=2500 \
#         --num_test=1000 \
#         --eval_interval=500 \
#         --resume=False \
#         --always_save_checkpoint=True \
#         --wandb_log=True \
#         --do_length_gen_eval=True \
#         --min_eval_len=2 \
#         --max_eval_len=15 \
#         --batch_size=128 \
#         --eval_train=False
# done

for posenc in fire; do
    CUDA_VISIBLE_DEVICES=1 python run.py \
        --wandb_run_name='gpt-nano-reverse-n_digit-3-posenc-'$posenc \
        --wandb_project='priming' \
        --data_format='index-hints' \
        --n_digit=3 \
        --max_iters=2500 \
        --num_test=1000 \
        --eval_interval=500 \
        --resume=False \
        --always_save_checkpoint=False \
        --wandb_log=False \
        --do_length_gen_eval=True \
        --min_eval_len=2 \
        --max_eval_len=15 \
        --batch_size=128 \
        --eval_train=True \
        --pos_emb_type=$posenc
done

