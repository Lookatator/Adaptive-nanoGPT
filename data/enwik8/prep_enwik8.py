#!/usr/bin/env python
# coding=utf-8

import os
import sys
import zipfile

import numpy as np
import pickle

if os.path.exists('train.txt'):
    print('Tokenized enwik8 already exists - skipping processing')
    sys.exit()

data = zipfile.ZipFile('enwik8.zip').read('enwik8').decode('latin-1')

print('Length of enwik8: {}'.format(len(data)))

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):int(n*0.95)]
test_data = data[int(n*0.95):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
test_ids = encode(test_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
print(f"test has {len(test_ids):,} tokens")

# Decode train, validation, and test data to text
train_text = decode(train_ids)
val_text = decode(val_ids)
test_text = decode(test_ids)

# Save decoded texts to files
for data_split, text in [('train', train_text), ('val', val_text), ('test', test_text)]:
    with open(os.path.join(os.path.dirname(__file__), f'{data_split}.txt'), 'w', encoding='utf-8') as f:
        f.write(text)


# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
test_ids = np.array(test_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
test_ids.tofile(os.path.join(os.path.dirname(__file__), 'test.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

