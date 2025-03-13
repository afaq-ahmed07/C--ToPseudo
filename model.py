import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # Learnable linear layers for queries, keys, and values.
        self.fc_q = nn.Linear(embed_size, embed_size)
        self.fc_k = nn.Linear(embed_size, embed_size)
        self.fc_v = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.fc_q(query)  # (B, seq_len, embed_size)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Reshape for multiple heads: (B, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention, V)  # (B, num_heads, seq_len, head_dim)

        # Concat heads and pass through final linear layer.
        out = out.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        out = self.fc_out(out)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, forward_expansion, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention))
        forward = self.feed_forward(x)
        out = self.norm2(x + self.dropout(forward))
        return out

class DecoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, forward_expansion, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.encoder_attention = MultiHeadAttention(embed_size, num_heads)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        # Self-attention with masking (for causal decoding)
        self_attn = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn))
        # Encoder-decoder attention
        enc_attn = self.encoder_attention(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(enc_attn))
        forward = self.feed_forward(x)
        out = self.norm3(x + self.dropout(forward))
        return out

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size=512, num_layers=6,
                 num_heads=8, forward_expansion=4, dropout=0.1, max_len=100):
        super(Transformer, self).__init__()
        self.embed_size = embed_size

        # Embedding layers for source and target languages.
        self.src_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, max_len)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(embed_size, num_heads, forward_expansion, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(embed_size, num_heads, forward_expansion, dropout) for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src):
        # src: (B, src_len)
        # Create mask to ignore padding tokens (assume padding index is 0)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        # tgt: (B, tgt_len)
        B, T = tgt.shape
        tgt_mask = torch.tril(torch.ones((T, T), device=tgt.device)).expand(B, 1, T, T)
        return tgt_mask

    def encode(self, src, src_mask):
        x = self.dropout(self.positional_encoding(self.src_embedding(src)))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_out, src_mask, tgt_mask):
        x = self.dropout(self.positional_encoding(self.tgt_embedding(tgt)))
        for layer in self.decoder_layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
        out = self.fc_out(dec_out)
        return out