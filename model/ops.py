import torch
import torch.nn as nn


class AdaptiveInput(nn.Module):
    """Adaptive Input Representation Embedding Layer
    """
    def __init__(self, params):
        super(AdaptiveInput, self).__init__()
        self.cut_off = eval(params.cut_off)
        self.embed_dim = params.embed_dim

        # Add bucket, if vocab size is larger than last bucket
        if params.vocab_size > self.cut_off[-1]:
            self.cut_off += (params.vocab_size,)

        self.embeddings = nn.ModuleList()

        # Append Adaptive Embedding Layers
        for i in range(len(self.cut_off)):
            prev_size = self.cut_off[i - 1] if i > 0 else 0  # 0, 20000, ...
            curr_size = self.cut_off[i] - prev_size          # 20000, 40000, ...
            embed_dim = int(params.embed_dim // (params.embed_factor ** i))

            embedding = nn.Sequential(
                nn.Embedding(curr_size, embed_dim, padding_idx=0),
                nn.Linear(embed_dim, params.embed_dim, bias=False)
            )
            self.embeddings.append(embedding)

        self.device = params.device

    def forward(self, x):
        " x = [batch size, sentence length] "
        embed = torch.FloatTensor(x.shape + (self.embed_dim,)).to(self.device)
        # embed = [batch size, sentence length, embed dim]
        
        for i in range(len(self.cut_off)):
            # Make masking tensor "True", if 'ids' belong to current cluster
            mask = x.lt(self.cut_off[i])  # lower than

            # Extract 'ids' belonging to current cluster
            if i > 0:
                # Remove already embedded 'ids' belonging to previous cluster
                embedded_ids = ~x.lt(self.cut_off[i])
                print(embedded_ids)
                mask = mask & embedded_ids
                embed_portion = x[mask] - self.cut_off[i - 1]
            else:
                embed_portion = x[mask]

            # Embed 'ids' belonging to current cluster
            embed[mask] = self.embeddings[i](embed_portion)
        
        # embed = [batch size, sentence length, embed dim]
        return embed
