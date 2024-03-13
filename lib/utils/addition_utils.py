import string
import random
import numpy as np
from random import shuffle, randint

def get_char_encode_decode():
    extra_tokens = []

    # get all the unique characters that occur in this text
    chars = sorted(list(set(string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation + ' \n')))
    if extra_tokens:
        assert all([c not in chars for c in extra_tokens])
        chars += extra_tokens
    vocab_size = len(chars)
    print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    return encode, decode, vocab_size

def sample_n_digit(n_digit, dist='constant', params=None):
    if dist == 'constant':
        return n_digit
    elif dist == 'uniform':
        return random.randint(2, params)
    elif dist == 'zipf':
        return min(np.random.zipf(params) + 2, 15)
    elif dist == 'poisson':
        return min(np.random.poisson(n_digit), 15)
    else:
        raise ValueError(f'Unknown distribution: {dist}')

def generate_sample(n_digit, ary=10, carries=None, sampling_method='carry-first'):
    if sampling_method == 'carry-first':
        a = 0
        b = 0
        carries = [randint(0, 1) for _ in range(n_digit)] if carries is None else carries
        prev_carry = 0
        upper = ary - 1
        for i, carry in enumerate(carries):
            a_i = random.randint(0, upper - prev_carry) if not carry else random.randint(1 - prev_carry, upper)
            b_i = random.randint(0, upper - a_i - prev_carry) if not carry else random.randint(ary - a_i - prev_carry, upper)
            if random.random() < 0.5:
                a_i, b_i = b_i, a_i
            # total = random.randint(0, upper - prev_carry) if not carry else random.randint(1 - prev_carry, upper)
            # a_i = random.randint(0, total)
            # b_i = total - a_i
            a += a_i * (ary ** i)
            b += b_i * (ary ** i)
            prev_carry = carry
    elif sampling_method == 'uniform':
        a = sum([random.randint(0, ary - 1) * (ary ** i) for i in range(n_digit)])
        b = sum([random.randint(0, ary - 1) * (ary ** i) for i in range(n_digit)])
        carries = detect_carry(a, b, ary=ary, n_digit=n_digit)[2]

    return a, b, carries

def detect_carry(a, b, ary=10, n_digit=None):
    a_digits = []
    b_digits = []
    while a > 0 or b > 0:
        a_digits.append(a % ary)
        b_digits.append(b % ary)
        a //= ary
        b //= ary
    carries = []
    carry = 0
    for a_d, b_d in zip(a_digits, b_digits):
        carry = (a_d + b_d + carry) // ary
        carries.append(carry)
    n_digit = max(len(a_digits), len(b_digits)) if n_digit is None else n_digit
    carries = carries + [0] * (n_digit - len(carries))
    
    return a_digits, b_digits, carries

def int_to_base(n, N):
    """ Return base N representation for int n. """
    base_n_digits = string.digits + string.ascii_lowercase + string.ascii_uppercase
    result = ""
    if n < 0:
        sign = "-"
        n = -n
    else:
        sign = ""
    while n > 0:
        q, r = divmod(n, N)
        result += base_n_digits[r]
        n = q
    if result == "":
        result = "0"
    return sign + "".join(reversed(result))
