import math

import torch
import numpy as np
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attention_mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale

        if attention_mask:
            attention = attention * attention_mask

        attention = self.softmax(attention)
        attention = self.dropout(attention)
        return torch.bmm(attention, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads=8, dropout=0.):
        super().__init__()
        assert model_dim % num_heads == 0
        self.dropout = nn.Dropout(dropout)
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.output_linear = nn.Linear(model_dim, model_dim)
        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, k, v, q, attention_mask=None):
        residual = q
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = k.shape[0]
        key = self.linear_k(k)
        value = self.linear_v(v)
        query = self.linear_q(q)

        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        scale = (key.shape[-1] // num_heads) ** 0.5
        context = self.dot_product_attention(query, key, value, scale, attention_mask)
        context = context.view(batch_size, -1, dim_per_head * num_heads)
        context = self.output_linear(context)
        context = self.dropout(context)
        return self.layer_norm(context + residual)


class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim, ffn_dim=2048, dropout=0.):
        super().__init__()
        self.w1 = nn.Linear(model_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.w2(self.relu(self.w1(x)))
        output = self.dropout(output)
        output = self.layer_norm(output + x)
        return output


class PositionalEncoding(nn.Module):
    """
        Require: expect the model_dim to be even
    """

    def __init__(self, model_dim, max_seq_len):
        super().__init__()
        assert model_dim % 2 == 0
        positional_encoding = torch.zeros(max_seq_len, model_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        # pe.requires_grad = False
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        return x + self.positional_encoding.to(x.device)


class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8, ffn_dim=2048, dropout=0.):
        super().__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attention_mask=None):
        context = self.attention(inputs, inputs, inputs, attention_mask)
        output = self.feed_forward(context)
        return output


class MetaEncoder(nn.Module):
    def __init__(self, subnet_layers, total_feature, num_layers=6, model_dim=512, num_heads=8,
                 ffn_dim=2048, dropout=0.):
        super().__init__()
        self.encoders = nn.ModuleList(
            [EncoderLayer(model_dim=model_dim, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout) for _ in
             range(num_layers)]
        )
        self.embedding = nn.Embedding(total_feature, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim=model_dim, max_seq_len=subnet_layers)

    def forward(self, input):
        out = self.positional_encoding(self.embedding(input))
        for encoder in self.encoders:
            out = encoder(out)
        return out
