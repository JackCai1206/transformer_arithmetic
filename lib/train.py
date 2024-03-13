from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, AutoConfig, GPT2LMHeadModel, GenerationConfig
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.loading_from_pretrained import get_pretrained_model_config
from lib.config import config
import string
import torch as t
from lib.data import AdditionDataset, get_char_encode_decode
from tqdm.notebook import tqdm
import wandb
import os
from itertools import islice
from collections import OrderedDict, defaultdict
from glob import glob

def init_model(config=config):
    ckpt = None
    if config['resume']:
        if config['ckpt_name']:
            ckpt = t.load(os.path.join(config['out_dir'], config['ckpt_name']), map_location=config['device'])
        else:
            ckpt_files = glob(os.path.join(config['out_dir'], 'ckpt_*.pt'))
            if ckpt_files:
                # sort the files by the number in the filename
                ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                ckpt = t.load(ckpt_files[-1], map_location=config['device'])
            else:
                ckpt = t.load(os.path.join(config['out_dir'], 'ckpt.pt'), map_location=config['device'])
        update_keys = ['resume', 'do_train', 'do_eval', 'wandb_log', 'device', 'max_iters'] # always update these keys
        for k in update_keys:
            ckpt['config'][k] = config[k]
        if config['use_saved_config']:
            config.update(ckpt['config'])
    
    if config['init_from']:
        if config['init_pretrained']:
            hf_model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(config['init_from'])
        else:
            hf_config = AutoConfig.from_pretrained(config['init_from'])
            hf_model: GPT2LMHeadModel = AutoModelForCausalLM.from_config(hf_config)
        hf_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(config['init_from'], add_bos_token=True)
        num_toks_added = hf_tokenizer.add_special_tokens({
            'pad_token': '<#>',
            'additional_special_tokens': ['<1>', '<2>', '<3>', '<4>', '<5>', '<6>', '<7>', '<8>', '<9>', '<0>', '<+>', '<=>', '<$>']
        })
        avg_emb = hf_model.transformer.wte.weight.data.mean(dim=0)
        hf_model.resize_token_embeddings(len(hf_tokenizer))
        # prevent performance degredataion from embedding resize by initilizing the new tokens with the average of the existing tokens
        hf_model.transformer.wte.weight.data[-num_toks_added:] = avg_emb + 0.0005 * t.randn(num_toks_added, avg_emb.shape[0])
        hf_model.config.vocab_size = hf_model.get_input_embeddings().weight.shape[0]
        if config['use_hookedtransformer']:
            model = HookedTransformer.from_pretrained(config['init_from'], hf_model=hf_model, tokenizer=hf_tokenizer)
        else:
            model = hf_model
        # assert (len(hf_tokenizer) == len(model.tokenizer))
        config['block_size'] = hf_model.config.n_positions
        config['n_embd'] = hf_model.config.n_embd
        config['n_layer'] = hf_model.config.n_layer
        config['n_head'] = hf_model.config.n_head
        config['pad_token'] = '<#>'
        config['eos_token'] = '\n'

        encode = hf_tokenizer.encode
        decode = hf_tokenizer.decode
        vocab_size = len(hf_tokenizer)
    else:
        encode, decode, vocab_size = get_char_encode_decode()
        model_cfg = HookedTransformerConfig(
            n_layers=config['n_layer'],
            n_heads=config['n_head'],
            d_model=config['n_embd'],
            n_ctx=config['block_size'],
            d_head=config['n_embd'] // config['n_head'],
            act_fn='relu',
            d_vocab=vocab_size,
            positional_embedding_type=config['pos_emb_type'],
            use_attn_result=False
        )
        model = HookedTransformer(model_cfg)
        config['pad_token'] = '#'
        config['eos_token'] = '\n'
        config['use_hookedtransformer'] = True

    model = model.to(config['device'])
    model = t.compile(model)

    it = 0
    best_val_acc = 0
    if config['resume']:
        if ckpt and 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
            it = ckpt['it']
            best_val_acc = ckpt['best_val_acc']
        else:
            model.load_state_dict(ckpt)

    return model, it, best_val_acc, ckpt, encode, decode, vocab_size


from torch.optim.lr_scheduler import CosineAnnealingLR

def init_optimizer(model, config=config, ckpt=None):
    learning_rate, weight_decay, beta2, lr_decay_iters = config['learning_rate'], config['weight_decay'], config['beta2'], config['lr_decay_iters']

    no_decay = ["bias", "layer_norm.weight"]
    # no_decay = []
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = t.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, betas=(0.9, beta2))

    if config['resume'] and ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])

    lr_scheduler = CosineAnnealingLR(optimizer, T_max=lr_decay_iters, eta_min=learning_rate/10)

    if config['resume'] and ckpt:
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])

    return optimizer, lr_scheduler


def evaluate(model: GPT2LMHeadModel, it, test_dataloaders, encode, decode, config=config, verbose=True):
    wandb_log, eval_iters, log_interval, device = config['wandb_log'], config['eval_iters'], config['log_interval'], config['device']
    pad_token_id, eos_token_id, space_token_id = encode(config['pad_token'])[0], encode(config['eos_token'])[0], encode(' ')[0]
    model.eval()
    val_acc = {}
    num_correct = defaultdict(int)
    num_total = defaultdict(int)
    digit_nc = defaultdict(int)
    digit_nt = defaultdict(int)
    carry_nc = defaultdict(int)
    carry_nt = defaultdict(int)
    with t.no_grad():
        for test_dataloader in test_dataloaders:
            tqdm_obj = tqdm(test_dataloader)
            for i, batch in enumerate(tqdm_obj):
                input = batch['input']
                # input = t.tensor(encode('twinkle twinkle little star, how I wonder')).unsqueeze(0).to(device)
                if config['use_hookedtransformer']:
                    output = model.generate(
                        input.to(device),
                        max_new_tokens=test_dataloader.dataset.ans_pad_length + 5,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                        return_type='tensor',
                        do_sample=False,
                        prepend_bos=False,
                        padding_side='left',
                        verbose=False,
                        # attention_mask=batch['attention_mask'].to(device)
                    )
                else:
                    output = model.generate(
                        input.to(device),
                        generation_config=GenerationConfig(
                            max_new_tokens=test_dataloader.dataset.ans_pad_length + 5,
                            do_sample=False,
                            pad_token_id=pad_token_id,
                            eos_token_id=eos_token_id
                        ),
                        return_dict_in_generate=False
                    )
                # input_str = [decode(i) for i in inp.cpu().numpy()]
                # output_str = [decode(o[inp.shape[1]:]).strip() for o in output.cpu().numpy()]
                # ans_str = [a.strip() for a in batch['ans']]
                for carries, nd, inp, out, ans in zip(batch['carries'].numpy(), batch['n_digit'].numpy(), input.cpu(), output[:, input.shape[1]:].cpu(), batch['ans_toks']):
                    out = t.atleast_1d(out[(out != eos_token_id) & (out != pad_token_id) & (out != space_token_id)].squeeze(0))
                    ans = t.atleast_1d(ans[(ans != eos_token_id) & (ans != pad_token_id) & (ans != space_token_id)].squeeze(0))
                    cor = len(out) == len(ans) and t.all(out == ans).item()
                    if i == 0 and not cor and verbose:
                        print('inp:', repr(decode(inp.cpu().numpy())),
                              'out:', repr(decode(out.cpu().numpy())),
                              'ans:', repr(decode(ans.cpu().numpy())))
                    num_correct[nd] += cor
                    num_total[nd] += 1
                    if nd == config['n_digit']:
                        for d in range(min(nd + 1, len(out), len(ans))):
                            digit_nc[d] += cor and (out[d] == ans[d]).item()
                            digit_nt[d] += 1
                        # for ci, c in enumerate(carries):
                        #     if c == 1:
                        #         carry_nc[ci] += cor
                        #         carry_nt[ci] += 1
                        carry_nc[sum(carries)] += cor
                        carry_nt[sum(carries)] += 1
                # wrong_ids = [i for i, (out, ans) in enumerate(zip(output_str, ans_str)) if out != ans]
                # wrong_ids = (output != batch['ans']).any(dim=1).nonzero()
                # num_correct += len(batch['ans']) - len(wrong_ids)
                # if i==0 and verbose:
                #     for wid in wrong_ids:
                #         print(repr(input_str[wid]), repr(output_str[wid]), repr(ans_str[wid]))
                        # print(
                        #     repr(decode(inp[wid].cpu.numpy)),
                        #     repr(decode(output[wid].cpu().numpy())),
                        #     repr(decode(batch['ans'][wid].cpu().numpy())))
                if eval_iters and i > eval_iters:
                    break
    for nd in num_correct:
        val_acc[nd] = num_correct[nd] / num_total[nd] if num_total[nd] > 0 else 0
    val_acc_total = sum(num_correct.values()) / sum(num_total.values())
    for d in range(config['n_digit'] + 1):
        digit_nc[d] = digit_nc[d] / digit_nt[d] if digit_nt[d] > 0 else 0
    for ci in range(config['n_digit'] + 1):
        carry_nc[ci] = carry_nc[ci] / carry_nt[ci] if carry_nt[ci] > 0 else 0
    print('digit_nc:', digit_nc)
    print('carry_nc:', carry_nc)

    print('val_acc:', val_acc_total)
    print('val_acc:', val_acc)
    return val_acc_total, val_acc, num_correct, num_total, digit_nc, carry_nc

def train(model: GPT2LMHeadModel, it, train_dataloader, test_dataloaders, test_train_dataloader, optimizer, lr_scheduler, encode, decode, logger, config=config, best_val_acc=0):
    max_iters, do_train, do_eval, log_interval, eval_interval, device = config['max_iters'], config['do_train'], config['do_eval'], config['log_interval'], config['eval_interval'], config['device']
    gradient_accumulation_steps, out_dir, num_train, wandb_log, always_save_checkpoint = config['gradient_accumulation_steps'], config['out_dir'], config['num_train'], config['wandb_log'], config['always_save_checkpoint']
    pbar = tqdm(total=max_iters)
    pbar.update(it)
    model.train()
    model.requires_grad_(True)
    while it < max_iters and do_train:
        for batch in train_dataloader:
            if it == 0:
                print(repr(decode(batch['input'][0].cpu().numpy())))
                print(len(decode(batch['input'][0].cpu().numpy())))
                print(batch['input'].shape)
            if config['use_hookedtransformer']:
                loss = model(batch['input'].to(device), return_type='loss', attention_mask=batch['attention_mask'].to(device))
            else:
                loss = model(batch['input'].to(device), labels=batch['input'].to(device), attention_mask=batch['attention_mask'].to(device)).loss
            loss = loss / gradient_accumulation_steps
            loss.backward()
            if (it + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            lr_scheduler.step()
            if it % log_interval == 0:
                print(f'it: {it}, loss: {loss.item()}')
                if wandb_log:
                    wandb.log({'train_loss': loss.item(), 'lr': lr_scheduler.get_last_lr()[0]}, step=it)
            if it % eval_interval == 0 and do_eval:
                print('evaluating...')
                val_acc_total, val_acc, _, val_num_total, val_d_acc, val_c_acc = evaluate(model, it, test_dataloaders, encode, decode, config=config, verbose=True)
                train_acc_total, train_acc, _, train_num_total, _, _ = evaluate(model, it, [test_train_dataloader], encode, decode, config=config, verbose=False) if config['eval_train'] else None
                val_acc = {f'val_acc_{nd}': val_acc[nd] for nd in val_acc}
                train_num = {f'train_num_{nd}': train_num_total[nd] for nd in train_num_total}
                val_d_acc = {f'val_digit_acc_{i}': val_d_acc[i] for i in val_d_acc}
                val_c_acc = {f'val_carry_type_acc_{i}': val_c_acc[i] for i in val_c_acc}
                logger.log_metrics({'it': it, 'val_acc': val_acc_total, 'train_acc': train_acc_total} | val_acc | train_num | val_d_acc | val_c_acc)
                model.train()
                if val_acc_total >= best_val_acc or always_save_checkpoint:
                    best_val_acc = val_acc_total
                    print(f'saving checkpoint to {out_dir}/ckpt_{num_train}.pt')
                    if config['save_by_iter']:
                        save_fn = f'{out_dir}/ckpt_{it}.pt'
                    else:
                        save_fn = f'{out_dir}/ckpt.pt'
                    t.save({
                        'it': it,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'best_val_acc': best_val_acc,
                        'wandb_run_id': wandb.run.id if wandb.run else None,
                        'config': config
                    }, save_fn)
                    if val_acc_total >= 0.99:
                        return
            it += 1
            pbar.update(1)
    
