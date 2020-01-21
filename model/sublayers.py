import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layer
    """
    def __init__(self, params):
        super(MultiHeadAttention, self).__init__()
        assert params.hidden_dim % params.num_heads == 0, "hidden dimension must be divisible by the number of heads"
        self.num_heads = params.num_heads
        self.attn_dim = params.hidden_dim // self.num_heads

        self.q_w = nn.Linear(params.hidden_dim, self.num_heads * self.attn_dim)
        self.k_w = nn.Linear(params.hidden_dim, self.num_heads * self.attn_dim)
        self.v_w = nn.Linear(params.hidden_dim, self.num_heads * self.attn_dim)

        self.o_w = nn.Linear(self.num_heads * self.attn_dim, params.hidden_dim)
        
        self.scale_factor = self.attn_dim ** 0.5
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, q, k, v):
        " q, k, v = [batch size, sentence length, hidden dim] "

        batch_size = q.size(0)

        # Split hidden dimension into (num heads * attention dimension)
        q = self.q_w(q).view(batch_size, -1, self.num_heads, self.attn_dim).transpose(1, 2)
        k = self.k_w(k).view(batch_size, -1, self.num_heads, self.attn_dim).transpose(1, 2)
        v = self.v_w(v).view(batch_size, -1, self.num_heads, self.attn_dim).transpose(1, 2)
        # q, k, v = [batch size, num heads, sentence length, attn dim]

        attn = torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor
        # attn   = [batch size, num heads, sentence length, sentence length]
        score = self.dropout(F.softmax(attn, dim=-1))
        # score  = [batch size, num heads, sentence length, sentence length]

        output = torch.matmul(score, v)
        # output = [batch size, num heads, sentence length, attn dim]

        output = output.transpose(1, 2).contiguous()
        # output = [batch size, sentence length, num heads, attn dim]
        output = output.view(batch_size, -1, self.num_heads * self.attn_dim)
        # output = [batch size, sentence length, hidden dim]

        output = self.o_w(output)
        # output = [batch size, sentence length, hidden dim]
        return output


class FeedForward(nn.Module):
    """Position-wise Feed Forward Network
    """
    def __init__(self, params):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(params.hidden_dim, params.ffn_dim)
        self.fc2 = nn.Linear(params.ffn_dim, params.hidden_dim)
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, x):
        " x = [batch size, sentence length, hidden dim] "

        x = self.dropout(F.relu(self.fc1(x)))
        # x = [batch size, sentence legnth, feed forward dim]
        x = self.fc2(x)
        # x = [batch size, sentence legnth, hidden dim]
        return x
