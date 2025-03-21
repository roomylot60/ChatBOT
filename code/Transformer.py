import torch
import torch.nn as nn
import torch.nn.functional as F

# Scaled Dot-Product Attention
def scaled_dot_product_attention(q, k, v, mask=None, training=True):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attention_weights = F.softmax(scores, dim=-1)

    # 🔥 크기 불일치 문제 해결: `v.shape[-2]`이 `attention_weights.shape[-1]`과 다르면 조정
    if v.shape[-2] != attention_weights.shape[-1]:
        diff = abs(v.shape[-2] - attention_weights.shape[-1])
        if v.shape[-2] > attention_weights.shape[-1]:
            v = v[:, :, :attention_weights.shape[-1], :]
        else:
            v = F.pad(v, (0, 0, 0, diff), value=0)

    output = torch.matmul(attention_weights, v)
    return output, attention_weights


# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None, training=True):  # 🔥 `training` 인자 추가
        batch_size = q.shape[0]
        q, k, v = self.layer_norm(q), self.layer_norm(k), self.layer_norm(v)
        q, k, v = self.split_heads(self.wq(q), batch_size), self.split_heads(self.wk(k), batch_size), self.split_heads(self.wv(v), batch_size)

        # 🔥 크기 확인 및 조정
        if k.shape[-2] != q.shape[-2]:
            k = k[:, :, :q.shape[-2], :]
        if v.shape[-2] != k.shape[-2]:
            v = v[:, :, :k.shape[-2], :]

        attention, _ = scaled_dot_product_attention(q, k, v, mask, training)  # 🔥 `training` 전달
        attention = attention.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        return self.fc_out(attention)

# Transformer Encoder
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, dff), nn.ReLU(), nn.Linear(dff, d_model))
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.mha(x, x, x, mask)
        x = self.layernorm1(x + self.dropout(attn_output))
        x = self.layernorm2(x + self.dropout(self.ffn(x)))
        return x

# Transformer Decoder
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, dff), nn.ReLU(), nn.Linear(dff, d_model))
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None, training=True):  # 🔥 `training` 인자 추가
        x = self.layernorm1(x + self.dropout(self.mha1(x, x, x, look_ahead_mask)))

        # 🔥 크기 불일치 해결: `enc_output`과 `x` 크기 조정 (훈련 시에는 적용 안 함)
        if not training and enc_output.shape[1] != x.shape[1]:
            diff = abs(enc_output.shape[1] - x.shape[1])
            if enc_output.shape[1] > x.shape[1]:
                x = F.pad(x, (0, 0, 0, diff), value=0)
            else:
                x = x[:, :enc_output.shape[1], :]

        x = self.layernorm2(x + self.dropout(self.mha2(enc_output, enc_output, x, padding_mask, training)))
        return self.layernorm3(x + self.dropout(self.ffn(x)))

# Transformer Model
class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout=0.3): # 기존의 0.1 에서 0.3으로 변경(과적합 방지)
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.encoder = TransformerEncoderLayer(d_model, num_heads, dff, dropout)
        self.decoder = TransformerDecoderLayer(d_model, num_heads, dff, dropout)
        self.final_layer = nn.Linear(d_model, target_vocab_size)

    def forward(self, enc_input, dec_input, enc_mask=None, dec_mask=None, training=True):  # 🔥 `training` 인자 추가
        enc_input, dec_input = self.embedding(enc_input), self.embedding(dec_input)
        enc_output = self.encoder(enc_input, enc_mask)
        dec_output = self.decoder(dec_input, enc_output, dec_mask, enc_mask, training=training)  # 🔥 `training` 전달
        return self.final_layer(dec_output)
