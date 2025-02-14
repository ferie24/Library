import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class Attention(nn.Module): 
    def __init__(self, input_dim): 
        super(Attention, self).__init__()
        self.K = nn.Linear(input_dim, input_dim, bias=False)
        self.Q = nn.Linear(input_dim, input_dim, bias=False)
        self.V = nn.Linear(input_dim, input_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x): 
        scores = self.K(x) @ self.Q(x).transpose(-2, -1)
        att_weight = self.softmax(scores / np.sqrt(x.shape[-1]))
        out = att_weight @ self.V(x)
        return out

class MultiHeadAttention(nn.Module): 
    def __init__(self, dim, head_size, num_heads): 
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList(Attention(dim) for _ in range(num_heads))
        self.Z = nn.Linear(num_heads * head_size, dim)

    def forward(self, x): 
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.Z(out)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        #self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, dim, head_size, num_heads, ff_size): 
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(dim, head_size, num_heads)
        self.layer_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_size), 
            nn.GELU(), 
            nn.Linear(ff_size, dim)
        )  
    
    def forward(self, x): 
        out = x + self.mha(x)
        out = self.layer_norm(out)
        temp = out
        out = temp + self.ff(out)
        out = self.layer_norm(out)
        return out

class Encoder(nn.Module): 
    def __init__(self, dim, head_size, num_heads, ff_size, num_encoders, num_classes): 
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(EncoderLayer(dim, head_size, num_heads, ff_size) for _ in range(num_encoders))
        self.pe = PositionalEncoding(dim, max_len=128)
        self.linear = nn.Linear(dim, num_classes)
    
    def forward(self, x): 
        x = x + self.pe(x)
        for layer in self.layers: 
            x = layer(x)
        x = x.mean(dim=1) 
        return self.linear(x)

