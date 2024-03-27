import string
import torch as t
import numpy as np
from random import shuffle, randint
from lib.config import config
from collections import OrderedDict
from functools import partial
from itertools import islice
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from .utils.waves import WaveGenerator, WaveGenerator2
from .utils.addition_utils import sample_n_digit, generate_sample, int_to_base
from .utils.automata import dataset_map

class PromptAnswerDataset(Dataset):
    def __init__(self, is_train=True, encode=None, config=config, external_test_data=None, external_train_data=None):
        self.block_size = config['block_size']
        self.num_train = config['num_train']
        self.num_test = config['num_test']
        self.max_iters = config['max_iters']
        self.online = config['online']
        self.fill_block = config['fill_block']
        self.pad_token_id = encode(config['pad_token'])[0]
        self.eos_token_id = encode(config['eos_token'])[0] # encode('\n')[0]
        self.encode = encode
        self.is_train = is_train
        self.prompt_pad_length = 0
        self.ans_pad_length = 0
        self.has_external_test_data = False
        self.has_external_train_data = False
        self.check_leak = config['check_leak']

        if external_test_data is not None:
            self.has_external_test_data = True
            self.test_data = external_test_data
        elif not is_train:
            self.generate_test_data()
        for prompt, (ans, _) in self.test_data.items():
            self.prompt_pad_length = max(len(encode(prompt)), self.prompt_pad_length)
            self.ans_pad_length = max(len(encode(ans)), self.ans_pad_length)

        if external_train_data is not None:
            self.has_external_train_data = True
            self.train_data = external_train_data
        elif is_train:
            self.generate_train_data()
    
    def generate_test_data(self):
        raise NotImplementedError
    def generate_train_data(self):
        raise NotImplementedError
    
    def __len__(self):
        if self.is_train:
            if self.has_external_train_data:
                return len(self.train_data)
            else:
                return self.num_train
        else:
            if self.has_external_test_data:
                return len(self.test_data)
            else:
                return self.num_test

    def __getitem__(self, idx):
        if self.is_train:
            block = []
            c = randint(0, self.num_train-1)
            while len(block) <= self.block_size:
                if config['random_order']:
                    prompt, (ans, _) = iter(islice(self.train_data.items(), randint(0, self.num_train-1), None)).__next__()
                else:
                    prompt, (ans, _) = iter(islice(self.train_data.items(), c, c+1)).__next__()
                # for online training, regenerate data if we run out
                if self.online and idx >= self.num_train - 1:
                    self.generate_train_data()
                line = prompt + ans
                true_length = len(block)
                block += self.encode(''.join(line))
                c = (c + 1) % self.num_train
            if self.fill_block:
                # instead of padding, crop the block
                j = randint(0, len(block) - self.block_size)
                block = block[j:self.block_size+j]
                return {'input': t.tensor(block, dtype=t.long), 'attention_mask': t.ones(self.block_size, dtype=t.long)}
            else:
                block = [self.pad_token_id] * (self.block_size - true_length) + block[:true_length] # left pad
                return {
                    'input': t.tensor(block, dtype=t.long),
                    'attention_mask': t.cat((t.zeros(self.block_size - true_length, dtype=t.long), t.ones(true_length, dtype=t.long)))
                }
            
        else:
            if idx >= self.num_test:
                raise StopIteration
            prompt, (ans, carries) = self.test_data.popitem(last=False)
            self.test_data[prompt] = (ans, carries)
            # prompt, (ans, carries) = islice(self.test_data.items(), idx, idx+1).__next__()
            ans_toks = self.encode(ans)
            ans_toks_pad = ans_toks + [self.eos_token_id] * (self.ans_pad_length-len(ans_toks))
            prompt_toks = self.encode(prompt)
            prompt_toks_pad = [self.pad_token_id] * (self.prompt_pad_length-len(prompt_toks)) + prompt_toks
            line_toks_pad = prompt_toks_pad + ans_toks + [self.pad_token_id] * (self.ans_pad_length-len(ans_toks))
            return {
                'input': t.tensor(prompt_toks_pad, dtype=t.long),
                'ans_toks': t.tensor(ans_toks_pad, dtype=t.long),
                'ans': ans,
                'carries': t.tensor(carries, dtype=t.bool),
                'line': t.tensor(line_toks_pad, dtype=t.long),
                'n_digit': t.tensor(len(carries), dtype=t.long)
            }

class AdditionDataset(PromptAnswerDataset):
    def __init__(self, n_digit=None, is_train=True, encode=None, config=config, external_test_data=None, external_train_data=None):
        self.data_format = config['data_format']
        self.ary = config['ary']
        self.n_digit = config['n_digit'] if n_digit is None else n_digit
        self.train_digit_dist = config['train_digit_dist']
        self.check_leak = config['check_leak']
        super().__init__(is_train=is_train, encode=encode, config=config, external_test_data=external_test_data, external_train_data=external_train_data)
    
    def generate_train_data(self):
        if self.test_data is None:
            raise ValueError('Cannot generate train data without test data, because we need to avoid test set leakage')
        self.train_data = OrderedDict()
        skipped = 0
        while len(self.train_data) < config['num_train']:
            sampled_n_digit = sample_n_digit(self.n_digit, self.train_digit_dist)
            a, b, carries = generate_sample(sampled_n_digit, ary=self.ary)
            prompt, ans = self.get_line(a, b, carries)
            if self.check_leak:
                if prompt in self.test_data:
                    skipped += 1
                    if skipped > 1000:
                        raise ValueError('Cannot find unseen samples')
                    continue # don't want to train on test data
            self.train_data[prompt] = (ans, carries)
    
    def generate_test_data(self):
        self.test_data = OrderedDict()
        while len(self.test_data) < config['num_test']:
            # try covering all possible carries if we can, if not we are sampling carry randomly
            all_carries = list(range(2 ** self.n_digit))
            # random.shuffle(all_carries) 
            for carries in all_carries:
                carries = [int(x) for x in int_to_base(carries, 2).zfill(self.n_digit)]
                a, b, carries = generate_sample(self.n_digit, ary=self.ary, carries=carries)
                prompt, ans = self.get_line(a, b, carries)
                self.test_data[prompt] = (ans, carries)
    
    def get_line(self, a, b, carries):
        if self.data_format == 'reverse':
            ans = str(a + b)[::-1] + '$\n'
            prompt = '$' + str(a) + '+' + str(b) + '='
        elif self.data_format == 'reverse-no-dollar':
            ans = str(a + b)[::-1] + '\n'
            prompt = '\n' + str(a) + '+' + str(b) + '='
        elif self.data_format == 'binary':
            a_bin = bin(a)[2:]
            b_bin = bin(b)[2:]
            c_bin = bin(a + b)[2:]
            prompt = '$' + a_bin + '+' + b_bin + '=' 
            ans = c_bin[::-1] + '$\n'
        elif self.data_format == 'carry-hints':
            s = list(zip(str(a+b)[::-1], carries))
            ans = ''.join([f'{"c" if x[1] == 1 else "n"}{x[0]}' for x in s]) + '$\n'
            prompt = '$' + str(a) + '+' + str(b) + '='
        elif self.data_format == 'index-hints':
            a_hint = [f'{n}{h}' for n, h in zip(list(str(a)), list(string.ascii_letters)[:self.n_digit])]
            b_hint = [f'{n}{h}' for n, h in zip(list(str(b)), list(string.ascii_letters)[:self.n_digit])]
            c_hint = [f'{n}{h}' for n, h in zip(list(str(a+b)[::-1]), list(string.ascii_letters))]
            ans = ''.join(c_hint) + '$\n'
            prompt = '$' + ''.join(a_hint) + '+' + ''.join(b_hint) + '='
        elif self.data_format == 'extra_toks':
            ans = str(a + b)[::-1]
            a_str = ''.join([f'<{x}>' for x in str(a)])
            b_str = ''.join([f'<{x}>' for x in str(b)])
            c_str = ''.join([f'<{x}>' for x in ans])
            prompt = '<$>' + a_str + '<+>' + b_str + '<=>'
            ans = c_str + '<$>\n'
        elif self.data_format == 'space':
            ans = str(a + b)[::-1] + '$\n'
            ans = ans.replace('', ' ')[:-1] # remove space on either end
            prompt = '$' + str(a) + '+' + str(b) + '='
            prompt = prompt.replace('', ' ')[:-1] 
        elif self.data_format == 'reverse_zeroshot':
            prompt = 'The correct sum of the two numbers is ' + str(a) + '+' + str(b) + '='
            ans = str(a + b)
        else:
            prompt = '$' + str(a) + '+' + str(b) + '='
            ans = str(a + b) + '$' + '\n'

        return prompt, ans

class AutomataDataset(PromptAnswerDataset):
    def __init__(self, n_digit=None, is_train=True, encode=None, config=config, external_test_data=None, external_train_data=None):
        self.data_format = config['data_format']
        self.ary = config['ary']
        self.n_digit = config['n_digit'] if n_digit is None else n_digit
        automaton_config = {
            'seed': config['seed'],
            'length': config['n_digit'],
            'n': config['ary']
        } | config['automaton_config']
        self.automaton = dataset_map[self.data_format](automaton_config)
        super().__init__(is_train=is_train, encode=encode, config=config, external_test_data=external_test_data, external_train_data=external_train_data)

    def generate_test_data(self):
        self.test_data = OrderedDict()
        while len(self.test_data) < config['num_test']:
            x, fx = self.automaton.sample()
            x = '$' + ''.join(map(str, x.tolist())) + '='
            x = x.replace('', ' ')[:-1]
            fx = ''.join(map(str, fx.tolist())) + '$\n'
            fx = fx.replace('', ' ')[:-1]
            self.test_data[x] = (fx, [])
    
    def generate_train_data(self):
        self.train_data = OrderedDict()
        c = 0
        while len(self.train_data) < config['num_train']:
            x, fx = self.automaton.sample()
            x = '$' + ''.join(map(str, x.tolist())) + '='
            x = x.replace('', ' ')[:-1]
            fx = ''.join(map(str, fx.tolist())) + '$\n'
            fx = fx.replace('', ' ')[:-1]
            if x in self.test_data:
                c += 1
                if c > 10000:
                    raise ValueError('Cannot find unseen samples')
                continue
            self.train_data[x] = (fx, [])
            c = 0

class WaveDataset(PromptAnswerDataset):
    def __init__(self, n_digit=None, is_train=True, encode=None, config=config, external_test_data=None, external_train_data=None):
        self.data_format = config['data_format']
        self.ary = config['ary']
        self.n_digit = config['n_digit'] if n_digit is None else n_digit
        self.wave_generator = WaveGenerator2(length=config['block_size'], periods=[2, 3, 5])
        super().__init__(is_train=is_train, encode=encode, config=config, external_test_data=external_test_data, external_train_data=external_train_data)

    def generate_test_data(self):
        self.test_data = OrderedDict()
        while len(self.test_data) < config['num_test']:
            wave = self.wave_generator.sample()
            x = wave[:len(wave)//2]
            fx = wave[len(wave)//2:len(wave)//2+20]
            self.test_data[x] = (fx, [])
    
    def generate_train_data(self):
        self.train_data = OrderedDict()
        c = 0
        while len(self.train_data) < config['num_train']:
            wave = self.wave_generator.sample()
            x = wave[:len(wave)//2]
            fx = wave[len(wave)//2:]
            if self.check_leak:
                if x in self.test_data:
                    c += 1
                    if c > 10000:
                        raise ValueError('Cannot find unseen samples')
                    continue
            self.train_data[x] = (fx, [])
            c = 0
            # print(len(self.train_data))

def get_train_test_loaders(encode, config=config):
    batch_size, eval_batch_size = config['batch_size'], config['eval_batch_size']
    do_length_gen_eval, min_eval_len, max_eval_len = config['do_length_gen_eval'], config['min_eval_len'], config['max_eval_len']

    if config['dataset'] == 'addition' or config['dataset'] == None: # legacy
        DatasetClass = AdditionDataset
    elif config['dataset'] == 'automata':
        DatasetClass = AutomataDataset
    elif config['dataset'] == 'openwebtext':
        DatasetClass = partial(load_dataset, 'openwebtext', split='train')
    elif config['dataset'] == 'waves':
        DatasetClass = WaveDataset

    test_dataloaders = []
    if config['do_eval']:
        if not do_length_gen_eval:
            min_eval_len = max_eval_len = config['n_digit']
        eval_range = list(range(min_eval_len, max_eval_len + 1))
        eval_range.insert(0, eval_range.pop(eval_range.index(config['n_digit'])))
        for n_digit in eval_range:
            test_dataset = DatasetClass(n_digit=n_digit, is_train=False, encode=encode)
            test_dataloader = DataLoader(test_dataset, batch_size=min(eval_batch_size, len(test_dataset)), shuffle=False, drop_last=False)
            test_dataloaders.append(test_dataloader)

    all_test_data = {}
    if config['check_leak']:
        for test_dataloader in test_dataloaders:
            all_test_data |= test_dataloader.dataset.test_data
    train_dataset = DatasetClass(is_train=True, encode=encode, external_test_data=all_test_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)\
    
    test_train_dataloader = None
    if config['do_eval'] and config['eval_train']:
        # train_data_subset = OrderedDict(islice(train_dataloader.dataset.train_data.items(), 0, config['num_test']))
        train_data_subset = OrderedDict({k: v for k, v in islice(train_dataloader.dataset.train_data.items(), config['num_test'])})
        train_subset = DatasetClass(is_train=False, encode=train_dataloader.dataset.encode, config=config, external_test_data=train_data_subset)
        eval_batch_size = min(config['eval_batch_size'], len(train_subset))
        test_train_dataloader = DataLoader(train_subset, batch_size=eval_batch_size, shuffle=False, drop_last=False)

    return train_dataloader, test_dataloaders, test_train_dataloader
