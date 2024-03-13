set -e
export TOKENIZERS_PARALLELISM=false

for n_digit in {3..5}; do
    for data_format in space reverse extra_toks; do
        for init_pretrained in True False; do
            CUDA_VISIBLE_DEVICES=1 python run.py \
            --out_dir='out/nl-pretraining/gpt2-reverse-n_digit-'$n_digit'-data_format-'$data_format'-init_pretrained-'$init_pretrained \
            --wandb_run_name='gpt2-reverse-n_digit-'$n_digit'-data_format-'$data_format'-init_pretrained-'$init_pretrained \
            --wandb_project='nl-pretraining' \
            --wandb_group='nl-pretraining-init_pretrained-'$init_pretrained \
            --init_from='gpt2' \
            --init_pretrained=$init_pretrained \
            --resume=False \
            --max_iters=10000 \
            --batch_size=6 \
            --eval_batch_size=32 \
            --gradient_accumulation_steps=6 \
            --learning_rate=1e-3 \
            --data_format=$data_format \
            --wandb_log=True \
            --n_digit=$n_digit \
            --do_train=True \
            --do_eval=True \
            --online=True \
            --do_length_gen_eval=False \
            --min_eval_len=2 \
            --max_eval_len=10 \
            --use_saved_config=False \
            --override_eval_file=True
        done
    done
done
