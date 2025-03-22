import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from Transformer import Transformer  # Transformer 모델 import
from rouge_score import rouge_scorer

# 데이터 로드
df = pd.read_csv("../data/ChatbotData.csv")

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

        # 🔥 `MAX_LENGTH` 기준 패딩
        question_tokens = question_tokens[:self.max_length] + [self.vocab["<PAD>"]] * (self.max_length - len(question_tokens))
        answer_tokens = answer_tokens[:self.max_length] + [self.vocab["<PAD>"]] * (self.max_length - len(answer_tokens))

        return torch.tensor(question_tokens, dtype=torch.long), torch.tensor(answer_tokens, dtype=torch.long)

# DataLoader 설정 (batch_size=1로 변경)
dataset = ChatbotDataset(df, vocab)
eval_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("⚠️ GPU 사용 불가. CPU 모드로 실행합니다.")

# 모델 생성 및 로드
transformer = Transformer(num_layers=4, d_model=256, num_heads=8, dff=512, input_vocab_size=len(vocab), target_vocab_size=len(vocab), dropout=0.1).to(device)
transformer.load_state_dict(torch.load("../models/transformer_chatbot.pth", map_location=torch.device("cpu"))) # GPU에서 학습한 모델을 CPU로 로드
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

def calculate_rouge(model, dataloader, vocab, device):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    model.eval()

    total_rouge1 = 0
    total_rougeL = 0
    total_samples = 0

    with torch.no_grad():
        for enc_input, dec_input in dataloader:
            enc_input, dec_input = enc_input.to(device), dec_input.to(device)
            output = model(enc_input, dec_input)
            predicted_ids = output.argmax(dim=-1)

            for i in range(enc_input.shape[0]):
                reference = [idx for idx in dec_input[i].cpu().tolist() if idx not in [vocab["<PAD>"], vocab["<EOS>"]]]
                candidate = [idx for idx in predicted_ids[i].cpu().tolist() if idx not in [vocab["<PAD>"], vocab["<EOS>"]]]

                # idx → 단어로 변환
                reference_text = " ".join([k for k, v in vocab.items() if v in reference])
                candidate_text = " ".join([k for k, v in vocab.items() if v in candidate])

                scores = scorer.score(reference_text, candidate_text)
                total_rouge1 += scores['rouge1'].fmeasure
                total_rougeL += scores['rougeL'].fmeasure
                total_samples += 1

    avg_rouge1 = total_rouge1 / total_samples
    avg_rougeL = total_rougeL / total_samples
    return avg_rouge1, avg_rougeL
# 🔹 챗봇 응답 테스트
# Greedy Decoding 방식으로 챗봇 응답 생성
# def chatbot_response(model, user_input, vocab, device):
#     model.eval()
#     tokens = [vocab.get(token, vocab["<UNK>"]) for token in user_input.split()] + [vocab["<EOS>"]]
#     input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

#     dec_input = torch.tensor([vocab["<SOS>"]], dtype=torch.long).unsqueeze(0).to(device)

#     with torch.no_grad():
#         for _ in range(MAX_LENGTH):
#             output = model(input_tensor, dec_input, training=False)  # 🔥 평가 시 `training=False`
#             next_word = output.argmax(-1)[:, -1].item()

#             if next_word == vocab["<EOS>"]:
#                 break
#             dec_input = torch.cat([dec_input, torch.tensor([[next_word]], dtype=torch.long).to(device)], dim=1)

#     # 🔥 `<SOS>` 토큰 제거 후 응답 반환
#     response_tokens = dec_input.squeeze(0).tolist()[1:]  # `<SOS>` 제거
#     response = [word for word, idx in vocab.items() if idx in response_tokens]

#     return " ".join(response)

# Beam Search Decoding 방식으로 챗봇 응답 생성
def chatbot_response(model, user_input, vocab, device, beam_size=3, max_length=50):
    model.eval()
    tokens = [vocab.get(token, vocab["<UNK>"]) for token in user_input.split()] + [vocab["<EOS>"]]
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    # 초기 디코더 입력값: <SOS>
    dec_input = torch.tensor([[vocab["<SOS>"]]], dtype=torch.long).to(device)
    sequences = [(dec_input, 0)]  # (현재 시퀀스, 누적 확률)

    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            with torch.no_grad():
                output = model(input_tensor, seq, training=False)  # Transformer 모델 예측
                topk_probs, topk_indices = torch.topk(output[:, -1, :], beam_size)  # Top-k 선택

            for i in range(beam_size):
                next_token = topk_indices[0, i].item()
                new_seq = torch.cat([seq, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)
                new_score = score + math.log(topk_probs[0, i].item())  # 로그 확률 합산
                all_candidates.append((new_seq, new_score))

        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]  # 상위 beam_size 개 선택

    # 최종 선택된 시퀀스 중 가장 높은 확률을 가진 문장 선택
    best_sequence = sequences[0][0].squeeze(0).tolist()[1:]  # `<SOS>` 제거
    response = [word for word, idx in vocab.items() if idx in best_sequence and idx not in [vocab["<EOS>"], vocab["<PAD>"]]]

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
