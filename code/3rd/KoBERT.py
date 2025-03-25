# ✅ KoBERT.py
import torch
import torch.nn as nn
from transformers import BertModel

class KoBERT_Transformer(nn.Module):
    def __init__(self, decoder_vocab_size, hidden_dim=768, num_layers=4, num_heads=8, dff=1024, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained("monologg/kobert")
        self.decoder_embedding = nn.Embedding(decoder_vocab_size, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=dff, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_dim, decoder_vocab_size)

    def forward(self, src_ids, src_mask, tgt_ids):
        # BERT 인코더
        enc_out = self.bert(input_ids=src_ids, attention_mask=src_mask).last_hidden_state

        # 디코더 입력
        tgt_emb = self.decoder_embedding(tgt_ids)
        tgt_mask = self._generate_square_subsequent_mask(tgt_ids.size(1)).to(tgt_ids.device)

        out = self.decoder(tgt=tgt_emb, memory=enc_out, tgt_mask=tgt_mask)
        return self.output_layer(out)

    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
