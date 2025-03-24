# ✅ train.py (Korpora 기반 + Word2Vec + Transformer)
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec
from preprocess import *
from Transformer import Transformer

# 데이터 로딩 및 전처리
print("[INFO] Korpora에서 챗봇 및 메신저 데이터 로드")
q_list, a_list = load_korpora_corpus()
corpus = build_corpus(q_list, a_list)
vocab = build_vocab(corpus)
save_vocab(vocab, "../../models/vocab.json")

print("[DEBUG] word2vec 학습을 위한 corpus 토큰 수:", len(corpus))
if len(corpus) == 0:
    raise ValueError("❌ 수집된 데이터가 없어서 Word2Vec 학습 불가합니다.")

print(f"[INFO] Word2Vec 훈련 시작")
w2v_model = Word2Vec(sentences=corpus, vector_size=300, window=5, min_count=1, workers=4)
embedding_matrix = build_embedding_matrix(vocab, w2v_model, dim=300)

enc = encode_sentences(q_list, vocab)
dec_in = decode_sentences(a_list, vocab)
dec_out = label_sentences(a_list, vocab)

# Dataset 정의
class ChatDataset(Dataset):
    def __init__(self):
        self.enc = torch.tensor(enc)
        self.dec_in = torch.tensor(dec_in)
        self.dec_out = torch.tensor(dec_out)

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, idx):
        return self.enc[idx], self.dec_in[idx], self.dec_out[idx]

PAD_IDX = vocab[PAD]

def collate_fn(batch):
    enc, dec_in, dec_out = zip(*batch)
    return (
        pad_sequence(enc, batch_first=True, padding_value=PAD_IDX),
        pad_sequence(dec_in, batch_first=True, padding_value=PAD_IDX),
        pad_sequence(dec_out, batch_first=True, padding_value=PAD_IDX),
    )

# DataLoader
BATCH_SIZE = 32
dataset = ChatDataset()
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# 모델 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(
    num_layers=4,
    d_model=300,
    num_heads=6,
    dff=512,
    input_vocab_size=len(vocab),
    target_vocab_size=len(vocab),
    dropout=0.3
).to(device)

model.embedding.weight.data.copy_(torch.tensor(embedding_matrix, dtype=torch.float32))
model.embedding.weight.requires_grad = False

# Optimizer & Loss
optimizer = optim.Adam(model.parameters(), lr=5e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# 학습 함수
def train(epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"[Epoch {epoch+1}]")
        for enc_batch, dec_in_batch, dec_out_batch in progress_bar:
            enc_batch, dec_in_batch, dec_out_batch = (
                enc_batch.to(device), dec_in_batch.to(device), dec_out_batch.to(device)
            )

            optimizer.zero_grad()
            output = model(enc_batch, dec_in_batch)
            loss = loss_fn(output.view(-1, output.size(-1)), dec_out_batch.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f"\n✅ Epoch {epoch+1} | Avg Loss: {total_loss/len(dataloader):.4f}\n")

train(epochs=10)
torch.save(model.state_dict(), "../../models/pretrained_transformer_model.pth")
w2v_model.save("../../models/word2vec_ko.model")
print("✅ 학습된 모델과 Word2Vec 저장 완료")
