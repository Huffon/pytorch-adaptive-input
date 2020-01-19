import torch.nn as nn

from model.layers import AdaptiveInput, PositionalEmbedding
from model.block import DecoderLayer


class TransformerLM(nn.Module):
    """Adaptive Input Representations for Neural Language Modeling
    """
    def __init__(self, params):
        super(TransformerLM, self).__init__()
        self.embedding = AdaptiveInput(params)
        self.positional = PositionalEmbedding(params)
        self.layers = nn.ModuleList([DecoderLayer(params) for _ in range(params.num_layers)])
        self.decoder = nn.Linear(params.hidden_dim, params.vocab_size)

    def forward(self, input_ids):
        " input_ids = [batch size, sentence length] "
        
        outputs = self.embedding(input_ids) + self.positional(input_ids)
        # outputs = [batch size, sentence length, hidden dim]
        
        for layer in self.layers:
            outputs = layer(outputs)
        # outputs = [batch size, sentence length, hidden dim]

        outputs = self.decoder(outputs)
        # outputs = [batch size, sentence length, vocab size]
        return outputs
        

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
