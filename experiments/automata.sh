set -e

for n_digit in 20 40; do
    for data_format in cyclic dihedral; do
        for init_pretrained in True False; do
            CUDA_VISIBLE_DEVICES=1 python run.py \
                --wandb_run_name='gpt-2-automata-length-'$n_digit'-type-'$data_format'-init_pretrained-'$init_pretrained \
                --wandb_project='automata' \
                --wandb_group='automata-exp1' \
                --dataset='automata' \
                --init_from='gpt2' \
                --init_pretrained=$init_pretrained \
                --tokenizer='gpt2' \
                --prune_embeddings=True \
                --data_format=$data_format \
                --automaton_config='{}' \
                --n_digit=$n_digit \
                --batch_size=8 \
                --fill_block=True \
                --eval_batch_size=1024 \
                --block_size=1024 \
                --max_iters=8000 \
                --num_train=10000 \
                --num_test=1000 \
                --check_leak=False \
                --learning_rate=5e-4 \
                --dropout=0.2 \
                --weight_decay=0.1 \
                --eval_interval=250 \
                --wandb_log=True \
                --online=True \
                --save_by_iter=False \
                --resume=False \
                --do_per_digit_eval=False \
                --do_carry_type_eval=False 
        done
    done
done
