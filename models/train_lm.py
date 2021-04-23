import torch
import torch.nn as nn

from model import LanguageModel
import random
from data_loader import get_SrcMLLinear_loader
from pickle import load
import math

## hyper-parameters
learning_rate = 2e-4
train_epochs = 5
max_depth = 3
layers = 1
z_dim = 1000
batch_size = 32
bidirectional = False
train_file = '../data/java-med-bpe.pkl'
val_file = '../data/valfunc_parsed_BPE.pkl'
device = 'cuda'
vocab_file = '../tokenizing/bpe_data/jm-trainfunc_vocab_BPE.pkl'
with open(vocab_file, 'rb') as f:
    char2idx = load(f)
    vocab_size = (max(char2idx.values())+1+2)+1 # +1 for pad

lm = LanguageModel(vocab_size, hidden_size=z_dim, num_layers=layers, bidirectional=bidirectional)
lm.load_state_dict(torch.load('weights/SrcMLBPE_JMJavaFunc_LMl1_z1000.pth'))
lm.to(device)

optimizer = torch.optim.Adam(lm.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss(ignore_index = 0)

train_data_loader = get_SrcMLLinear_loader(train_file, vocab_file, batch_size, num_workers=8)
val_data_loader = get_SrcMLLinear_loader(val_file, vocab_file, batch_size)
train_size = len(train_data_loader)
print('Training data size:', train_size)
print('Vocab size:', vocab_size)

# ----- Training Part -----
for e_idx in range(0, train_epochs):
    for b_idx, (pad_x, lengths) in enumerate(train_data_loader):
        pad_x = pad_x.to(device)
        x_logits = lm(pad_x, lengths)

        x_logits_flat = x_logits.view(-1, vocab_size)
        if bidirectional:
            x_labels_flat = pad_x[1:-1].view(-1)
        else:
            x_labels_flat = pad_x[1:].view(-1)
        dec_loss = loss_fn(x_logits_flat, x_labels_flat)
        loss = dec_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b_idx % 500 == 0:
            print(f'{b_idx}/{e_idx}: total {loss.item():.2f} = dec {dec_loss.item():.2f} (1/{math.e**dec_loss.item():.1f})')
    
    with torch.no_grad():
        losses = []
        for pad_x, lengths in val_data_loader:
            pad_x = pad_x.to(device)
            x_logits = lm(pad_x, lengths)

            x_logits_flat = x_logits.view(-1, vocab_size)
            if bidirectional:
                x_labels_flat = pad_x[1:-1].view(-1)
            else:
                x_labels_flat = pad_x[1:].view(-1)
            dec_loss = loss_fn(x_logits_flat, x_labels_flat)
            losses.append(dec_loss.item())
        print(f'VALIDATION LOSS {sum(losses)/len(losses)}')
torch.save(lm.state_dict(), f'weights/SrcMLBPE_JMJavaFunc_LMl1_z1000.pth')
