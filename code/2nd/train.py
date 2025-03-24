# ✅ train.py (Korpora 기반 + Word2Vec + Transformer)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np
from preprocess import *
from Transformer import Transformer

# 경로 설정
VOCAB_PATH = "data/vocab.json"
CORPUS_PATH = "data/corpus_tokenized.json"
EMBED_PATH = "models/embedding_matrix.npy"
MODEL_PATH = "models/pretrained_transformer_model.pth"
W2V_PATH = "models/word2vec_ko.model"

# 통체 데이터 로드
q_list, a_list = load_korpora_corpus()
# corpus = build_corpus(q_list, a_list)
# save_corpus(corpus)
# vocab = build_vocab(corpus)
# save_vocab(vocab)

# # Word2Vec 과 결함 메트릭스 생성
# build_and_save_embedding_matrix(corpus, vocab, dim=300, w2v_path=W2V_PATH, emb_path=EMBED_PATH)

# ✅ 기존 전처리 파일 로드(전처리 과정 완료 후 학습을 이어서 진행할 때)
corpus = load_corpus()
vocab = load_vocab()
embedding_matrix = np.load(EMBED_PATH)

# 인코딩 데이터 만들기
enc = encode_sentences(q_list, vocab)
dec_in = decode_sentences(a_list, vocab)
dec_out = label_sentences(a_list, vocab)

class ChatDataset(Dataset):
    def __init__(self):
        self.enc = torch.tensor(enc)
        self.dec_in = torch.tensor(dec_in)
        self.dec_out = torch.tensor(dec_out)

    def __len__(self): return len(self.enc)
    def __getitem__(self, idx): return self.enc[idx], self.dec_in[idx], self.dec_out[idx]

PAD_IDX = vocab["<PAD>"]

def collate_fn(batch):
    enc, dec_in, dec_out = zip(*batch)
    return (
        pad_sequence(enc, batch_first=True, padding_value=PAD_IDX),
        pad_sequence(dec_in, batch_first=True, padding_value=PAD_IDX),
        pad_sequence(dec_out, batch_first=True, padding_value=PAD_IDX)
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(
    num_layers=4,
    d_model=300,
    num_heads=6,
    dff=512,
    input_vocab_size=len(vocab),
    target_vocab_size=len(vocab),
    embedding_matrix=embedding_matrix
).to(device)

optimizer = optim.Adam(model.parameters(), lr=5e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

def train(epochs=10):
    dataloader = DataLoader(ChatDataset(), batch_size=32, shuffle=True, collate_fn=collate_fn)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for enc_batch, dec_in_batch, dec_out_batch in pbar:
            enc_batch, dec_in_batch, dec_out_batch = (
                enc_batch.to(device), dec_in_batch.to(device), dec_out_batch.to(device)
            )
            optimizer.zero_grad()
            output = model(enc_batch, dec_in_batch)
            loss = loss_fn(output.view(-1, output.size(-1)), dec_out_batch.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        print(f"[Epoch {epoch+1}] Avg Loss: {total_loss/len(dataloader):.4f}")

train()
torch.save(model.state_dict(), MODEL_PATH)
print("✅ 학원 완료 및 모델 저장")
