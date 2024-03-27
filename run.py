import os 
from lib.config import config
from lib.data import AdditionDataset, get_train_test_loaders
from lib.utils.addition_utils import get_char_encode_decode
from lib.train import init_model, init_optimizer, evaluate, train
from lib.logging import Logger
from transformer_lens import HookedTransformer, HookedTransformerConfig
import os
import wandb
import torch
import random
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType

torch.manual_seed(config['seed'])
random.seed(config['seed'])
np.random.seed(config['seed'])

if config['out_dir'] is None:
    config['out_dir'] = os.path.join('out', config['wandb_project'], config['wandb_run_name'])
if not os.path.exists(f"{config['out_dir']}"):
    os.makedirs(f"{config['out_dir']}")

model, it, best_val_acc, ckpt, encode, decode, vocab_size = init_model(config)
print('it:', it)
print('best_val_acc:', best_val_acc)

if config['use_lora']:
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False)
    model = get_peft_model(model, peft_config)

train_dataloader, test_dataloaders, test_train_dataloader = get_train_test_loaders(encode, config)

print('training data size:', len(train_dataloader.dataset))
print('test data size:', [len(test_dataloader.dataset) for test_dataloader in test_dataloaders])

# check if there is test set leakage
if config['check_leak']:
    for test_dataloader in test_dataloaders:
        for prompt in test_dataloader.dataset.test_data.keys():
            if prompt in train_dataloader.dataset.train_data:
                print('test set leakage:', prompt)
                exit(1)
    else:
        print('no test set leakage')

print("model has", sum(p.numel() for p in model.parameters()), "parameters")

optimizer, lr_scheduler = init_optimizer(model, config, ckpt)

if ckpt and 'wandb_run_id' in ckpt:
    config['wandb_run_id'] = ckpt['wandb_run_id']
else:
    config['wandb_run_id'] = wandb.util.generate_id()

logger = Logger(config['out_dir'], config=config)

if config['wandb_log']:
    wandb.init(
        project=config['wandb_project'],
        name=config['wandb_run_name'],
        id=config['wandb_run_id'],
        resume='allow' if config['resume'] else False,
        group=config['wandb_group'],
    )
    wandb.config.update(config, allow_val_change=True)

if config['do_eval'] and not config['do_train']:
    val_acc_total, val_acc, _, val_num_total, val_d_acc, val_c_acc = evaluate(model, it, test_dataloaders, encode, decode, config=config, verbose=True)
    train_acc_total, train_acc, _, train_num_total, _, _ = evaluate(model, it, [test_train_dataloader], encode, decode, config=config, verbose=False) if config['eval_train'] else None
    val_acc = {f'val_acc_{nd}': val_acc[nd] for nd in val_acc}
    train_num = {f'train_num_{nd}': train_num_total[nd] for nd in train_num_total}
    val_d_acc = {f'val_digit_acc_{i}': val_d_acc[i] for i in val_d_acc}
    val_c_acc = {f'val_carry_type_acc_{i}': val_c_acc[i] for i in val_c_acc}
    logger.log_metrics({'it': it, 'val_acc': val_acc_total, 'train_acc': train_acc_total} | val_acc | train_num | val_d_acc | val_c_acc)
elif config['do_train']:
    train(model, it, train_dataloader, test_dataloaders, test_train_dataloader, optimizer, lr_scheduler, encode, decode, logger, config=config, best_val_acc=0)
