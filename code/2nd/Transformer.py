import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout=0.1, embedding_matrix=None):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        
        # 🔹 Pretrained Embedding 적용
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.tensor(embedding_matrix, dtype=torch.float32))
            self.embedding.weight.requires_grad = False  # 필요시 True로 설정 가능
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, dff, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.linear = nn.Linear(d_model, target_vocab_size)

    def forward(self, enc_inputs, dec_inputs):
        enc_emb = self.embedding(enc_inputs)
        dec_emb = self.embedding(dec_inputs)
        enc_out = self.encoder(self.pos_encoding(enc_emb))
        dec_out = self.decoder(self.pos_encoding(dec_emb), enc_out)
        return self.linear(dec_out)
