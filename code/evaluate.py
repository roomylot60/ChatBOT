import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from Transformer import Transformer  # Transformer 모델 import

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

        # 🔥 `MAX_LENGTH` 기준 패딩
        question_tokens = question_tokens[:self.max_length] + [self.vocab["<PAD>"]] * (self.max_length - len(question_tokens))
        answer_tokens = answer_tokens[:self.max_length] + [self.vocab["<PAD>"]] * (self.max_length - len(answer_tokens))

        return torch.tensor(question_tokens, dtype=torch.long), torch.tensor(answer_tokens, dtype=torch.long)

# DataLoader 설정 (batch_size=1로 변경)
dataset = ChatbotDataset(df, vocab)
eval_loader = DataLoader(dataset, batch_size=1, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 생성 및 로드
transformer = Transformer(num_layers=4, d_model=256, num_heads=8, dff=512, input_vocab_size=len(vocab), target_vocab_size=len(vocab), dropout=0.1).to(device)
transformer.load_state_dict(torch.load("transformer_chatbot.pth"))
transformer.eval()

loss_fn = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])

# 🔹 Perplexity 계산
def calculate_perplexity(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for enc_input, dec_input in dataloader:
            enc_input, dec_input = enc_input.to(device), dec_input.to(device)
            output = model(enc_input, dec_input)

            loss = loss_fn(output.view(-1, output.size(-1)), dec_input.view(-1))
            total_loss += loss.item() * enc_input.size(0)
            total_samples += enc_input.size(0)

    avg_loss = total_loss / total_samples
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

# 🔹 BLEU Score 계산
def calculate_bleu(model, dataloader, vocab, device):
    model.eval()
    total_bleu = 0
    total_samples = 0

    with torch.no_grad():
        for enc_input, dec_input in dataloader:
            enc_input, dec_input = enc_input.to(device), dec_input.to(device)
            output = model(enc_input, dec_input)
            predicted_ids = output.argmax(dim=-1)

            for i in range(enc_input.shape[0]):
                reference = [dec_input[i].cpu().tolist()]
                candidate = predicted_ids[i].cpu().tolist()

                # <PAD> 및 <EOS> 제거
                reference = [[token for token in ref if token not in [vocab["<PAD>"], vocab["<EOS>"]]] for ref in reference]
                candidate = [token for token in candidate if token not in [vocab["<PAD>"], vocab["<EOS>"]]]

                # BLEU Score 계산
                bleu_score = sentence_bleu(reference, candidate, smoothing_function=SmoothingFunction().method1)
                total_bleu += bleu_score
                total_samples += 1

    avg_bleu = total_bleu / total_samples
    return avg_bleu

# 🔹 챗봇 응답 테스트
def chatbot_response(model, user_input, vocab, device):
    model.eval()
    tokens = [vocab.get(token, vocab["<UNK>"]) for token in user_input.split()] + [vocab["<EOS>"]]
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    dec_input = torch.tensor([vocab["<SOS>"]], dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(MAX_LENGTH):
            output = model(input_tensor, dec_input, training=False)  # 🔥 평가 시 `training=False`
            next_word = output.argmax(-1)[:, -1].item()

            if next_word == vocab["<EOS>"]:
                break
            dec_input = torch.cat([dec_input, torch.tensor([[next_word]], dtype=torch.long).to(device)], dim=1)

    # 🔥 `<SOS>` 토큰 제거 후 응답 반환
    response_tokens = dec_input.squeeze(0).tolist()[1:]  # `<SOS>` 제거
    response = [word for word, idx in vocab.items() if idx in response_tokens]

    return " ".join(response)

# ✅ 평가 실행
print("\n🔹 모델 성능 평가 시작...\n")

# 🔹 Perplexity & Loss 평가
avg_loss, perplexity = calculate_perplexity(transformer, eval_loader, loss_fn, device)
print(f"✅ 평균 Loss: {avg_loss:.4f}")
print(f"✅ Perplexity: {perplexity:.4f}\n")

# 🔹 BLEU Score 평가
avg_bleu = calculate_bleu(transformer, eval_loader, vocab, device)
print(f"✅ 평균 BLEU Score: {avg_bleu:.4f}\n")

# 🔹 챗봇 응답 테스트
sample_inputs = ["안녕하세요?", "오늘 날씨 어때?", "너의 이름은?", "무슨 일을 할 수 있어?"]
for sample in sample_inputs:
    print(f"💬 질문: {sample}")
    print(f"🤖 챗봇: {chatbot_response(transformer, sample, vocab, device)}\n")

print("\n🎉 모델 평가 완료!")
