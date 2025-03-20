import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from Transformer import Transformer

# 데이터 로드
df = pd.read_csv("../../data/ChatbotData.csv")

# 단어 사전 생성
vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
for sentence in df["Q"].tolist() + df["A"].tolist():
    for token in sentence.split():
        if token not in vocab:
            vocab[token] = len(vocab)

MAX_LENGTH = 50

# 데이터셋 정의
class ChatbotDataset(Dataset):
    def __init__(self, dataframe, vocab, max_length=MAX_LENGTH):
        self.data = dataframe
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data.iloc[idx]["Q"]
        answer = self.data.iloc[idx]["A"]

        question_tokens = [self.vocab.get(token, self.vocab["<UNK>"]) for token in question.split()]
        answer_tokens = [self.vocab["<SOS>"]] + [self.vocab.get(token, self.vocab["<UNK>"]) for token in answer.split()] + [self.vocab["<EOS>"]]

        question_tokens = question_tokens[:self.max_length] + [self.vocab["<PAD>"]] * (self.max_length - len(question_tokens))
        answer_tokens = answer_tokens[:self.max_length] + [self.vocab["<PAD>"]] * (self.max_length - len(answer_tokens))

        return torch.tensor(question_tokens, dtype=torch.long), torch.tensor(answer_tokens, dtype=torch.long)

# DataLoader 설정
dataset = ChatbotDataset(df, vocab)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 생성
transformer = Transformer(num_layers=4, d_model=256, num_heads=8, dff=512, input_vocab_size=len(vocab), target_vocab_size=len(vocab), dropout=0.1).to(device)
optimizer = torch.optim.Adam(transformer.parameters(), lr=5e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])

# 모델 훈련
def train(epochs=10):
    transformer.train()
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0

        for batch_idx, (enc_input, dec_input) in enumerate(train_loader):
            optimizer.zero_grad()
            enc_input, dec_input = enc_input.to(device), dec_input.to(device)

            output = transformer(enc_input, dec_input)

            # `output` 크기 확인
            assert output.shape[1] == dec_input.shape[1], f"Output size mismatch: {output.shape} vs Target: {dec_input.shape}"

            loss = loss_fn(output.view(-1, output.size(-1)), dec_input.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 🔹 배치 단위 진행 상태 출력
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        end_time = time.time()
        avg_loss = total_loss / len(train_loader)

        # 🔹 Epoch별 평균 Loss 및 학습 시간 출력
        print(f"\n✅ Epoch [{epoch+1}/{epochs}] 완료 - 평균 Loss: {avg_loss:.4f}, 학습 시간: {end_time - start_time:.2f}초\n")

train(epochs=10)
torch.save(transformer.state_dict(), "transformer_chatbot.pth")
