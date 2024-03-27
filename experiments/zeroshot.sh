set -e
export TOKENIZERS_PARALLELISM=false

for n_digit in 3; do
    for data_format in reverse_zeroshot; do
        for init_pretrained in True; do
            CUDA_VISIBLE_DEVICES=1 python run.py \
            --wandb_run_name='gptneo-reverse-n_digit-'$n_digit'-data_format-'$data_format'-init_pretrained-'$init_pretrained \
            --wandb_project='nl-pretraining' \
            --wandb_group='nl-pretraining-init_pretrained-'$init_pretrained \
            --init_from='microsoft/phi-2' \
            --tokenizer='microsoft/phi-2' \
            --init_pretrained=$init_pretrained \
            --resume=False \
            --max_iters=5000 \
            --batch_size=128 \
            --eval_batch_size=128 \
            --data_format=$data_format \
            --wandb_log=False \
            --n_digit=$n_digit \
            --do_train=True \
            --use_lora=True \
            --do_eval=True \
            --online=True \
            --random_order=True \
            --num_train=5000 \
            --do_length_gen_eval=False \
            --min_eval_len=2 \
            --max_eval_len=10 \
            --use_saved_config=False \
            --override_eval_file=False \
            --use_hookedtransformer=False \
            --pad_token=' ' \
            --eos_token='\n'
        done
    done
done
