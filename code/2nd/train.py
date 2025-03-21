# ✅ train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from preprocess import *
from gensim.models import Word2Vec
from Transformer import Transformer

q_list, a_list = load_data("../../data/ChatbotData.csv")
corpus = build_corpus(q_list, a_list)
vocab = build_vocab(corpus)
w2v_model = Word2Vec(sentences=corpus, vector_size=300, window=5, min_count=1, workers=4)
embedding_matrix = build_embedding_matrix(vocab, w2v_model, dim=300)

class ChatDataset(Dataset):
    def __init__(self, q_list, a_list, vocab):
        self.enc = encode_sentences(q_list, vocab)
        self.dec_in = decode_sentences(a_list, vocab)
        self.dec_out = label_sentences(a_list, vocab)
    def __len__(self): return len(self.enc)
    def __getitem__(self, idx):
        return torch.tensor(self.enc[idx]), torch.tensor(self.dec_in[idx]), torch.tensor(self.dec_out[idx])

dataset = ChatDataset(q_list, a_list, vocab)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = Transformer(4, 300, 6, 512, len(vocab), len(vocab), embedding_matrix=embedding_matrix).to("cpu")
optimizer = optim.Adam(model.parameters(), lr=5e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])

for epoch in range(10):
    total_loss = 0
    model.train()
    for enc, dec_in, dec_out in dataloader:
        optimizer.zero_grad()
        output = model(enc, dec_in)
        loss = loss_fn(output.view(-1, output.size(-1)), dec_out.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")