#!/usr/bin/python

'''Loads data as numpy data form'''

import torch
import torchvision.datasets as dss
from torch.utils import data
import os
import numpy as np

import xml.etree.ElementTree as ET
import re, tqdm
from pickle import load

class SrcMLLinearBPEData(dss.DatasetFolder):
    def __init__(self, data_file, vocab_file):
        self.data = []
        with open(data_file, 'rb') as f:
            self.data = list(zip(*load(f)))[1]
            self.data = list(filter(lambda x: len(x) < 2000, self.data))
            self.data = list(map(lambda x: ['@SOS'] + x + ['@EOS'], self.data))

        with open(vocab_file, 'rb') as f:
            self.vocab2idx = load(f)
            vocab_size = max(self.vocab2idx.values())
            self.vocab2idx['@SOS'] = vocab_size+1
            self.vocab2idx['@EOS'] = vocab_size+2

        print(f'# datapoints: {len(self.data)}')
    
    def __getitem__(self, index):
        token_list = self.data[index]
        idx_list = [self.vocab2idx[t]+1 for t in token_list 
                    if len(t) != 0 and t in self.vocab2idx]
        idx_tensor = torch.LongTensor(idx_list).unsqueeze(1)
        return idx_tensor
    
    def __len__(self):
        return len(self.data)

############

def smll_collate(arg):
    lengths = [x.size(0)-1 for x in arg]
    pad_x = torch.nn.utils.rnn.pad_sequence(arg)
    return pad_x, lengths

def get_SrcMLLinear_loader(root_path, vocab_path, batch_size, num_workers = 2):
    dataset = SrcMLLinearBPEData(root_path, vocab_path)
    data_loader = data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        collate_fn = smll_collate
    )
    return data_loader
    
