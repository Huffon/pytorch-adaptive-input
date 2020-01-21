import numpy as np
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, params):
        super(PositionalEmbedding, self).__init__()
        self.device = params.device
        # PE(pos, 2i)     = sin(pos/10000 ** (2*i / hidden_dim))
        # PE(pos, 2i + 1) = cos(pos/10000 ** (2*i / hidden_dim))
        sinusoid = np.array([pos / np.power(10000, 2 * i / params.hidden_dim)
                            for pos in range(params.max_len) for i in range(params.hidden_dim)])
        # sinusoid = [max len * hidden dim]

        sinusoid = sinusoid.reshape(params.max_len, -1)
        # sinusoid = [max len, hidden dim]

        sinusoid[:, 0::2] = np.sin(sinusoid[:, 0::2])
        sinusoid[:, 1::2] = np.cos(sinusoid[:, 1::2])
        sinusoid = torch.FloatTensor(sinusoid).to(self.device)

        self.embedding = nn.Embedding.from_pretrained(sinusoid, freeze=True)
        
    def forward(self, x):
        " x = [batch size, sentence length] "
        
        x_pos = torch.arange(x.size(-1), dtype=torch.long).to(self.device)
        embed = self.embedding(x_pos)
        # embed = [batch size, sentence length, embed dim]
        return embed