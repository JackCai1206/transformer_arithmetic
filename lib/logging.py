import sys
import csv
import wandb
import glob
import os
import datetime

class Logger:
    def __init__(self, out_dir, config):
        self.terminal = sys.stdout
        if config['do_train']:
            folder = 'train_logs'
        else:
            folder = 'eval_logs'
        folder = os.path.join(out_dir, folder)
        if not os.path.exists(folder):
            os.makedirs(folder)
        log_files = glob.glob(os.path.join(folder, f'log_*.csv'))
        self.log = open(os.path.join(folder, f"log_run-{config['wandb_run_id']}.csv"), 'w')
        fieldnames = ['it', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
        min_eval_len = config['min_eval_len'] if config['do_length_gen_eval'] else config['n_digit']
        max_eval_len = config['max_eval_len'] if config['do_length_gen_eval'] else config['n_digit']
        fieldnames += [f'val_acc_{i}' for i in range(min_eval_len, max_eval_len + 1)]
        fieldnames += [f'train_num_total_{i}' for i in range(min_eval_len, max_eval_len + 1)]
        if config['do_per_digit_eval']: fieldnames += [f'val_digit_acc_{i}' for i in range(config['n_digit'])]
        if config['do_carry_type_eval']: fieldnames += [f'val_carry_type_acc_{i}' for i in range(config['n_digit'])]
        self.logwriter = csv.DictWriter(self.log, fieldnames=fieldnames)
        self.logwriter.writeheader()
        self.wandb_log = config['wandb_log']

    def log_metrics(self, obj):
        # filter out keys not in fieldnames
        obj = {k: v for k, v in obj.items() if k in self.logwriter.fieldnames}
        self.logwriter.writerow(obj)
        self.log.flush()
        print(obj)
        if self.wandb_log and wandb.run is not None:
            wandb.log(obj, step=obj['it'])
