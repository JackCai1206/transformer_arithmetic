set -e
export TOKENIZERS_PARALLELISM=false

# for n_digit in {3..5}; do
for num_train in 20000; do
    for n_digit in 10 20 30; do
        for data_format in space extra_toks; do
        # for data_format in space; do
            for init_pretrained in True False; do
                CUDA_VISIBLE_DEVICES=0 python run.py \
                --wandb_run_name='gpt2-reverse-n_digit-'$n_digit'-num_train-'$num_train'-data_format-'$data_format'-init_pretrained-'$init_pretrained \
                --wandb_project='nl-pretraining' \
                --wandb_group='nl-pretraining-init_pretrained-'$init_pretrained \
                --init_from='gpt2' \
                --tokenizer='gpt2' \
                --block_size=1024 \
                --init_pretrained=$init_pretrained \
                --resume=False \
                --max_iters=5000 \
                --batch_size=8 \
                --eval_batch_size=32 \
                --gradient_accumulation_steps=4 \
                --learning_rate=5e-4 \
                --dropout=0.1 \
                --weight_decay=0.01 \
                --data_format=$data_format \
                --wandb_log=True \
                --n_digit=$n_digit \
                --do_train=True \
                --do_eval=True \
                --online=False \
                --random_order=True \
                --num_train=$num_train \
                --do_length_gen_eval=False \
                --min_eval_len=2 \
                --max_eval_len=10 \
                --use_saved_config=False \
                --override_eval_file=False \
                --use_hookedtransformer=False
            done
        done
    done
done


# Length generalization
# for n_digit in 3; do
#     for data_format in space; do
#         for init_pretrained in True; do
#             CUDA_VISIBLE_DEVICES=0 python run.py \
#             --out_dir='out/nl-pretraining/gpt2-reverse-n_digit-'$n_digit'-data_format-'$data_format'-init_pretrained-'$init_pretrained \
#             --wandb_run_name='gpt2-reverse-n_digit-'$n_digit'-data_format-'$data_format'-init_pretrained-'$init_pretrained \
#             --wandb_project='nl-pretraining' \
#             --wandb_group='nl-pretraining-init_pretrained-'$init_pretrained \
#             --init_from='gpt2' \
#             --tokenizer='gpt2' \
#             --block_size=1024 \
#             --init_pretrained=$init_pretrained \
#             --resume=True \
#             --max_iters=5000 \
#             --batch_size=10 \
#             --eval_batch_size=128 \
#             --gradient_accumulation_steps=4 \
#             --learning_rate=5e-4 \
#             --dropout=0.1 \
#             --weight_decay=0.01 \
#             --data_format=$data_format \
#             --wandb_log=False \
#             --n_digit=$n_digit \
#             --do_train=False \
#             --do_eval=True \
#             --online=True \
#             --random_order=True \
#             --num_train=5000 \
#             --do_length_gen_eval=True \
#             --min_eval_len=2 \
#             --max_eval_len=10 \
#             --use_saved_config=False \
#             --override_eval_file=False \
#             --use_hookedtransformer=False
#         done
#     done
# done
