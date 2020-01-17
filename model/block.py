import torch.nn as nn

from model.sublayers import MultiHeadAttention, FeedForward


class DecoderLayer(nn.Module):
    """Single Decoder Layer
    """
    def __init__(self, params):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(params)
        self.layer_norm1 = nn.LayerNorm(params.hidden_dim)
        self.feed_forward = FeedForward(params)
        self.layer_norm2 = nn.LayerNorm(params.hidden_dim)
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, x):
        " x = [batch size, sentence length, hidden dim] "

        # Conduct Multi-Head Self-Attention
        residual = x
        x = self.self_attn(x, x, x)
        x = self.dropout(x)
        x = residual + x
        x = self.layer_norm1(x)

        # Conduct Position-wise FeedForward
        residual = x
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x
        x = self.layer_norm2(x)

        return x