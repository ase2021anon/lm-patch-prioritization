import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LanguageModel(nn.Module):
    '''Language model'''
    def __init__(self, vocab_size=1000, hidden_size=100, num_layers=1, bidirectional=False):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = hidden_size)
        self.decoder = nn.GRU(
            input_size = hidden_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            bidirectional = bidirectional
        )
        self.bd = bidirectional
        self.hs = hidden_size
        self.vs = vocab_size
        bd_multiplier = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.total_rnns = num_layers
        self.total_rnns *= bd_multiplier
        self.projector = nn.Linear(hidden_size*bd_multiplier, vocab_size)
        self.device = None
    
    def _decode_from_emb(self, x, z, lengths):
        if x.size(-1) == 1:
            x = self.embedding(x).squeeze()
        pack_x = pack_padded_sequence(x, [l+int(self.bd) for l in lengths], enforce_sorted=False)
        packed_output, _ = self.decoder(pack_x, z)
        hidden_states, output_lengths = pad_packed_sequence(packed_output)
        if self.bd:
            forward_states = hidden_states[:-2, :, :self.hs]
            backward_states = hidden_states[2:, :, self.hs:]
            hidden_states = torch.cat([forward_states, backward_states], dim=2)
        projected = self.projector(hidden_states)
        return projected
    
    def decode_cell(self, x, z):
        emb_x = self.embedding(x)[0]
        _, state = self.decoder(emb_x, z)
        projected = self.projector(state)
        return state, projected
    
    def forward(self, x, lengths):
        self.device = self.device if self.device is not None else next(self.parameters()).device

        emb_x = self.embedding(x).squeeze(2)
        z = torch.zeros(self.total_rnns, x.size(1), self.hs).to(self.device)
        x_logits = self._decode_from_emb(emb_x, z, lengths)
        return x_logits
    