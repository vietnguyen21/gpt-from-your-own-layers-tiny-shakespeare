"""
GPT From Your Own Layers: Tiny Shakespeare

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_data
import os
import tempfile
import urllib.request
import torch

def load_data(train_frac=0.9):
    """Return (chars, encode, decode, train_data, val_data) for Tiny Shakespeare, one id per character."""
    
    dir = tempfile.gettempdir()
    path = dir + 'tinyshakespeare.txt'

    download_data = urllib.request.urlretrieve('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',path)
    
    data = []
    with open(path,'r') as f:
        for lines in f:
            for characters in lines:
                data.append(characters)


    # character id 
    chars = []
    for i in data:
        if i not in chars:
            chars.append(i)
    chars.sort()
    # encode
    
    def encode(s):
        encode_characters = []
        for characters in s:
            for i in range(len(chars)):
                if characters == chars[i]:
                    encode_characters.append(int(i))
        return encode_characters
    #encode whold text
    encode_data = torch.tensor(encode(data),dtype=torch.long)
    train_data = encode_data[:int(len(encode_data)*train_frac)]
    val_data = encode_data[int(len(encode_data)*train_frac):]

    #decode
    def decode(ids):
        decode_characters = []
        for characters in ids:
            decode_characters.append(chars[characters])
        return ''.join(decode_characters)
    
    return (chars,encode,decode,train_data,val_data)

# Step 2 - get_batch
import torch

def get_batch(data, block_size, batch_size, generator=None):
    """Random (batch_size, block_size) chunks x of data and their next-token targets y."""
    
    offset = torch.randint(len(data) - block_size, (batch_size,), generator=generator)

    xs = []
    ys = []
    for i in offset:
        x = data[i:i+block_size]
        y = data[i+1:i+block_size+1]

        xs.append(x)
        ys.append(y)

    x = torch.stack(xs)
    y = torch.stack(ys)

    return x,y

# Step 3 - BigramLanguageModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        """One embedding table: row i holds the logits for the token after token i."""
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self, idx, targets=None):
        """Return (logits of shape (B, T, vocab_size), loss or None)."""
        
        logits = self.token_embedding_table(idx)
        B,T,C = logits.shape
        logits_2d = logits.view(B*T,C)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits_2d,targets.view(B*T,))

        return logits,loss

# Step 4 - generate (not yet solved)
# TODO: implement

# Step 5 - train (not yet solved)
# TODO: implement

# Step 6 - estimate_loss (not yet solved)
# TODO: implement

# Step 7 - average_past (not yet solved)
# TODO: implement

# Step 8 - Head (not yet solved)
# TODO: implement

# Step 9 - MultiHeadAttention (not yet solved)
# TODO: implement

# Step 10 - FeedForward (not yet solved)
# TODO: implement

# Step 11 - LayerNorm (not yet solved)
# TODO: implement

# Step 12 - Block (not yet solved)
# TODO: implement

# Step 13 - GPTLanguageModel (not yet solved)
# TODO: implement

