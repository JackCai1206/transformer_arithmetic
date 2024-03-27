import string
# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

seed=42

out_dir = None # infer from wandb params
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = None
log_interval = 50 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True
save_by_iter = False

wandb_log = True # override via command line if you like
wandb_project = None
wandb_run_name = None
wandb_group = None

data_type='text'
data_format='reverse'
operator='+'
fill_block=True
dataset = 'addition'
automaton_config = {}

batch_size = 256
eval_batch_size = 256 * 4
block_size = 256 # context of up to 256 previous characters
# gradient_accumulation_steps = max(256 * 256 // (batch_size * block_size), 1)
gradient_accumulation_steps = 1


num_train = 10000
num_test = 1000
online = True
train_data_path = f'train_3digit_{num_train}.txt'
# val_data_path = 'val.bin'
resume = False
resume_dir = None
ckpt_name = None
reverse_c = True
eval_addition = True
start = "FILE:data/bal/test_10000.txt"
eval_addition_train = True
check_leak = True
# start_train = "FILE:data/one-sided-subtraction/plain/add_examples_10000_trainprompt.txt"

# n_layer = 16
# n_head = 4
# n_embd = 128
# dropout = 0

# n_layer = 4
# n_head = 4
# n_embd = 256
# dropout = 0

# baby GPT model :)
n_layer = 6
n_head = 6 * 6 // n_layer
n_embd = 384 * 6 // n_layer
dropout = 0.2

# GPT2 model
# n_layer = 12
# n_head = 12
# n_embd = 768
# dropout = 0.0

init_from = None
tokenizer = init_from
init_pretrained = False
pos_emb_type = 'standard'
prune_embeddings = False
chars = sorted(list(set(string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation + ' \n')))
freeze_keywords = []
freeze_except = []

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
weight_decay = 1e-1
# max_iters = round(256 * 256 / (batch_size * block_size) * 5000) + 1
max_iters = 20000
lr_decay_iters = max_iters # make equal to max_iters usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
grad_clip = 1.0
use_lora = True

warmup_iters = 100 # not super necessary potentially

device='cuda:0'

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

do_train = True
do_eval = True
eval_train = True

do_length_gen_eval = False
min_eval_len = None
max_eval_len = None

do_per_digit_eval = False
do_carry_type_eval = False

use_hookedtransformer = False
random_order = True

n_digit = 3
ary = 2 if data_format == 'binary' else 10
train_digit_dist = {}

pad_token = '#'
eos_token = '\n'
add_extra_tokens = False

use_saved_config = False
override_eval_file = False # override the saved config if resume is True

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None), dict, list))]
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
configurator_path = os.path.join(dir_path, 'configurator.py')
exec(open(configurator_path).read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
