import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# Functions to get slopes for ALiBi
def slopes_power_2(n_head):
    slope = (2**(-2**-(math.log2(n_head)-3)))
    slope1 = slope
    return [slope*slope1**n for n in range(n_head)]

def encoder_slopes(n_head):
    if math.log2(n_head).is_integer():
        return slopes_power_2(n_head)
    else:
        n = 2**math.floor(math.log2(n_head))
        return slopes_power_2(n) + encoder_slopes(2*n)[0::2][:n_head-n]
    
def decoder_slopes(slope):
    res = []
    for i in range(32):
        temp = []
        for j in range(0, i):
            temp.append(-j)
        temp = temp[::-1] + [0]*(32-i)
        res.append(temp)
    return slope*torch.Tensor(res)

# Scaled Dot-Product Self-Attention Head
class AttentionHead(nn.Module):
    def __init__(self, head_size, n_embd, block_size, decoder=False, dropout=0.2, alibi=False, n_head=2, device="cpu", slope=0.5):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.decoder = decoder
        self.alibi = alibi
        if self.alibi and not self.decoder:
            self.rel_pos = torch.arange(block_size)[None, :].to(device) - torch.arange(block_size)[:, None].to(device)
            self.rel_pos = torch.abs(self.rel_pos).unsqueeze(0).expand(n_head, -1,-1)
            self.slopes = torch.Tensor(encoder_slopes(n_head)).to(device)*(-1)
            self.bias = self.slopes.unsqueeze(1).unsqueeze(1)*self.rel_pos
            self.bias = self.bias.view(1, n_head, block_size, block_size)
            self.n_head = n_head
        if self.alibi and self.decoder:
            self.bias = decoder_slopes(slope).to(device)
            self.bias = self.bias.view(1, 1, block_size, block_size)

    def forward(self, x):
        batch, time, channels = x.shape

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        mat_mul = query @ key.transpose(-2, -1)

        if self.alibi:
            attn_weights = mat_mul.view(batch//self.n_head, self.n_head, 32, 32)
            attn_weights += self.bias[:,:,:32,:32].to(attn_weights)
            mat_mul = attn_weights.view(batch, 32, 32)

        mat_mul /= channels**(0.5)
        if self.decoder:
            mat_mul = mat_mul.masked_fill(self.tril[:time, :time] == 0, float("-inf"))
        mat_mul = F.softmax(mat_mul, dim=-1)
        attention_maps = mat_mul
        mat_mul = self.dropout(mat_mul)

        res = mat_mul @ value

        return res, attention_maps


# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size, n_embd, block_size, decoder=False, dropout=0.2, alibi=False):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size, n_embd, block_size, decoder=decoder, dropout=dropout, alibi=alibi, slope=(0.5)**(i+1)) for i in range(n_head)])
        self.projection_layer = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = []
        attention_maps = []
        for head in self.heads:
            h = head(x)
            res.append(h[0])
            attention_maps.append(h[1])
        res = torch.cat(res, dim=-1)
        res = self.projection_layer(res)
        res = self.dropout(res)

        return res, attention_maps


# Feedforward Network
class FeedForward(nn.Module):
    def __init__(self, n_input, n_hidden, dropout=0.2):
        super().__init__()
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(n_hidden, n_input)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        res = self.dropout(x)

        return res
    

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, n_head, n_embd, block_size, n_input, n_hidden, decoder=False, dropout=0.2, alibi=False):
        super().__init__()
        head_size = n_embd // n_head
        self.self_attention = MultiHeadAttention(n_head, head_size, n_embd, block_size, decoder=decoder, dropout=dropout, alibi=alibi)
        self.layernorm1 = nn.LayerNorm(n_embd)
        self.feedforward = FeedForward(n_input, n_hidden, dropout=dropout)
        self.layernorm2 = nn.LayerNorm(n_input)

    def forward(self, x):
        x, attention_maps = x
        y, attention_maps = self.self_attention(x)
        x = self.layernorm1(x + y)
        res = self.layernorm2(x + self.feedforward(x))

        return res, attention_maps


# Encoder only Transformer Model
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_input, n_hidden, n_output, n_layer, device="cpu", dropout=0.2, alibi=False):
        super().__init__()
        self.device = device
        self.input_embedding = nn.Embedding(vocab_size, n_embd)
        self.positional_encoding = nn.Embedding(block_size, n_embd)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(n_head, n_embd, block_size, n_input, n_hidden, dropout=dropout, alibi=alibi) for _ in range(n_layer)])
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(n_input*block_size, n_output)
        self.alibi = alibi
    
    def forward(self, x, targets=None):
        x, attention_maps = x
        batch, time = x.shape
        # print(x.shape, x)
        input_embd = self.input_embedding(x)
        if self.alibi:
            positional_embd = self.positional_encoding(torch.arange(time, device=self.device))
            y = input_embd + positional_embd
        else:
            y = input_embd
        y, attention_maps = self.transformer_blocks((y, None))
        y = self.flatten(y)
        y = self.linear(y)

        if targets == None:
            loss = None
        else:
            # batch, time, channels = y.shape
            # print(y.shape, targets.shape)
            # y = y.view(batch*time, channels)
            # targets = targets.view(batch*time)
            loss = F.cross_entropy(y, targets)

        return y, loss, attention_maps
    

# Decoder only Transformer Model
class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_input, n_hidden, n_layer, device="cpu", decoder=True, alibi=False):
        super().__init__()
        self.device = device
        self.input_embedding = nn.Embedding(vocab_size, n_embd)
        self.positional_encoding = nn.Embedding(block_size, n_embd)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(n_head, n_embd, block_size, n_input, n_hidden, decoder=decoder, dropout=0.2, alibi=alibi) for _ in range(n_layer)])
        self.linear = nn.Linear(n_input, vocab_size)

    def forward(self, x, targets=None):
        x, attention_maps = x
        batch, time = x.shape
        input_embd = self.input_embedding(x)
        positional_embd = self.positional_encoding(torch.arange(time, device=self.device))
        y = input_embd + positional_embd
        y, attention_maps = self.transformer_blocks((y, None))
        y = self.linear(y)

        if targets == None:
            loss = None
        else:
            batch, time, channels = y.shape
            y = y.view(batch*time, channels)
            targets = targets.view(batch*time)
            loss = F.cross_entropy(y, targets)

        return y, loss, attention_maps

