# ✅ Transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout=0.1, embedding_matrix=None):
        super().__init__()
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
        else:
            self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=dff, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, dim_feedforward=dff, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)
        self.final_layer = nn.Linear(d_model, target_vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embedding(src)  # [B, T, D]
        tgt_emb = self.embedding(tgt)  # [B, T, D]
        memory = self.encoder(src_emb)
        output = self.decoder(tgt_emb, memory)
        return self.final_layer(output)