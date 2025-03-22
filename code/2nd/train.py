# 🔧 개선된 train_transformer.py (Word2Vec + 형태소 분석 기반)
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm
from Transformer import Transformer
from preprocess import encode_sentences, decode_sentences,label_sentences
from preprocess import load_data, build_corpus, build_vocab, build_embedding_matrix
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# 🔹 데이터 로드 및 전처리
DATA_PATH = "../../data/ChatbotData.csv"
q_list, a_list = load_data(DATA_PATH)
corpus = build_corpus(q_list, a_list)
vocab = build_vocab(corpus)
w2v_model = Word2Vec(sentences=corpus, vector_size=300, window=5, min_count=1, workers=4)
embedding_matrix = build_embedding_matrix(vocab, w2v_model, dim=300)

MAX_SEQ = 25
qst, ans = load_data(DATA_PATH)
# enc = encode_sentences(qst, vocab)
# dec_in = np.array(decode_sentences(ans, vocab))     # ✅ 확실하게 배열로 변환
# dec_out = np.array(label_sentences(ans, vocab))

enc = encode_sentences(qst, vocab, max_seq_length=25)
dec_in = decode_sentences(ans, vocab, max_seq_length=25)
dec_out = label_sentences(ans, vocab, max_seq_length=25)

print(f"[DEBUG] enc: {type(enc)}, shape: {np.array(enc).shape}")
print(f"[DEBUG] dec_in: {type(dec_in)}, shape: {np.array(dec_in).shape}")
print(f"[DEBUG] dec_out: {type(dec_out)}, shape: {np.array(dec_out).shape}")

for i, d in enumerate(decode_sentences(ans, vocab)):
    if len(d) != MAX_SEQ:
        print(f"[❌] index {i}: length {len(d)} → {d}")

# 🔹 ChatDataset 정의
class ChatDataset(Dataset):
    def __init__(self, q_list, a_list, vocab):
        self.enc = encode_sentences(q_list, vocab)
        self.dec_in = decode_sentences(a_list, vocab)
        self.dec_out = label_sentences(a_list, vocab)

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, idx):
        enc = self.enc[idx]
        dec_in = self.dec_in[idx]
        dec_out = self.dec_out[idx]

        if len(enc) != MAX_SEQ or len(dec_in) != MAX_SEQ or len(dec_out) != MAX_SEQ:
            print(f"[🔍] idx={idx} | enc: {len(enc)} | dec_in: {len(dec_in)} | dec_out: {len(dec_out)}")

        assert len(enc) == len(dec_in) == len(dec_out) == MAX_SEQ, "⚠️ 길이 불일치 발생"

        return torch.tensor(enc), torch.tensor(dec_in), torch.tensor(dec_out)

PAD = "<PAD>"
def collate_fn(batch):
    enc, dec_in, dec_out = zip(*batch)
    enc = pad_sequence(enc, batch_first=True, padding_value=vocab[PAD])
    dec_in = pad_sequence(dec_in, batch_first=True, padding_value=vocab[PAD])
    dec_out = pad_sequence(dec_out, batch_first=True, padding_value=vocab[PAD])
    return enc, dec_in, dec_out

# 🔹 DataLoader
BATCH_SIZE = 32
dataset = ChatDataset(q_list, a_list, vocab)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# 🔹 모델 설정
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

# Pretrained Embedding 적용
model.embedding.weight.data.copy_(torch.tensor(embedding_matrix, dtype=torch.float32))
model.embedding.weight.requires_grad = False  # 🔒 고정 (선택)

# 🔹 Optimizer & Loss
optimizer = optim.Adam(model.parameters(), lr=5e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])

# 🔹 훈련 루프

def train(epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        start = time.time()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, (enc, dec_in, dec_out) in enumerate(progress_bar):
            enc, dec_in, dec_out = enc.to(device), dec_in.to(device), dec_out.to(device)

            optimizer.zero_grad()
            output = model(enc, dec_in)
            loss = loss_fn(output.view(-1, output.size(-1)), dec_out.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Batch Loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        print(f"\n✅ Epoch {epoch+1} 완료 - 평균 Loss: {avg_loss:.4f}, 시간: {time.time() - start:.2f}초\n")

# 🔹 학습 실행
train(epochs=10)
torch.save(model.state_dict(), "../../models/pretrained_transformer_model.pth")

# 학습 후 Word2Vec 모델 저장
w2v_model.save("../../models/word2vec_ko.model")
print("✅ Word2Vec 모델이 word2vec_ko.model로 저장되었습니다.")
