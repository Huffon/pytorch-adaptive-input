import numpy as np
import torch
import torch.nn as nn


class AdaptiveInput(nn.Module):
    """Adaptive Input Representation Embedding Layer
    """
    def __init__(self, params):
        super(AdaptiveInput, self).__init__()
        self.cut_off = eval(params.cut_off)
        self.embed_dim = params.embed_dim
        self.padding_idx = 0

        # Add bucket, if vocab size is larger than last bucket
        if params.vocab_size > self.cut_off[-1]:
            self.cut_off += (params.vocab_size,)

        self.embeddings = nn.ModuleList()
        # Append Adaptive Embedding Layers
        for i in range(len(self.cut_off)):
            prev_size = self.cut_off[i-1] if i > 0 else 0  # 0, 20000, ...
            curr_size = self.cut_off[i] - prev_size          # 20000, 40000, ...
            embed_dim = int(params.embed_dim // (params.embed_factor ** i))

            embedding = nn.Sequential(
                nn.Embedding(curr_size, embed_dim, padding_idx=self.padding_idx),
                nn.Linear(embed_dim, params.embed_dim, bias=False)
            )
            self.embeddings.append(embedding)
            self.padding_idx = None
        
        self.padding_idx = 0
        self.device = params.device

    def forward(self, x):
        " x = [batch size, sentence length] "
        
        embed = torch.FloatTensor(x.shape + (self.embed_dim,)).to(self.device)
        # embed = [batch size, sentence length, embed dim]

        for i in range(len(self.cut_off)):
            # If 'id' could belong to current cluster, mask it as True
            mask = x.lt(self.cut_off[i])

            # Extract 'ids' only belonging to current cluster
            if i > 0:
                # Remove previously embedded 'ids'
                possible_ids = x.ge(self.cut_off[i-1])
                mask = mask & possible_ids
                # Substract previous 'cluster size' to start from index 0
                embed_portion = x[mask] - self.cut_off[i-1]
            else:
                embed_portion = x[mask]

            # Embed 'ids' belonging to current cluster
            embed[mask] = self.embeddings[i](embed_portion)
        
        # embed = [batch size, sentence length, embed dim]
        return embed


class AdaptiveSoftmax(nn.Module):
    """Adaptive Softmax Layer
    """
    def __init__(self, params):
        super(AdaptiveSoftmax, self).__init__()

    def forward(self, x):
        " x = [batch size, sentence length, hidden dim] "
        pass
