import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from KoBERT import KoBERT_Transformer
from preprocess import CORPUS_PATH, TOKENIZER_PATH

# 설정
MAX_LEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")

# 🔹 토크나이저 및 데이터 로드
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
    input_data = data["input"]
    target_data = data["target"]

# 🔹 모델 로드
model = KoBERT_Transformer(decoder_vocab_size=tokenizer.vocab_size).to(DEVICE)
model.load_state_dict(torch.load("models/kobert_chatbot.pt", map_location=DEVICE))
model.eval()

# 🔹 응답 생성 함수
def generate_response(input_text):
    input_ids = tokenizer.encode(input_text, truncation=True, max_length=MAX_LEN, padding="max_length")
    input_tensor = torch.tensor([input_ids]).to(DEVICE)
    enc_mask = (input_tensor != tokenizer.pad_token_id).long()

    dec_input = torch.tensor([[tokenizer.cls_token_id]]).to(DEVICE)
    result = []

    for _ in range(MAX_LEN):
        with torch.no_grad():
            output = model(input_tensor, enc_mask, dec_input)
        next_token = output.argmax(-1)[:, -1].item()
        if next_token == tokenizer.sep_token_id:
            break
        result.append(next_token)
        dec_input = torch.cat([dec_input, torch.tensor([[next_token]]).to(DEVICE)], dim=1)

    return tokenizer.decode(result, skip_special_tokens=True)

# 🔹 BLEU
def calculate_bleu_score():
    total_bleu = 0
    for input_enc, target_enc in tqdm(zip(input_data[:100], target_data[:100]), total=100, desc="[BLEU]"):
        question = tokenizer.decode(input_enc["input_ids"], skip_special_tokens=True)
        reference = tokenizer.decode(target_enc["input_ids"], skip_special_tokens=True).split()
        prediction = generate_response(question).split()
        total_bleu += sentence_bleu([reference], prediction, smoothing_function=SmoothingFunction().method1)
    return total_bleu / 100

# 🔹 ROUGE
def calculate_rouge_score():
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)
    total_r1, total_rl = 0, 0
    for input_enc, target_enc in tqdm(zip(input_data[:100], target_data[:100]), total=100, desc="[ROUGE]"):
        question = tokenizer.decode(input_enc["input_ids"], skip_special_tokens=True)
        reference = tokenizer.decode(target_enc["input_ids"], skip_special_tokens=True)
        prediction = generate_response(question)
        scores = scorer.score(reference, prediction)
        total_r1 += scores['rouge1'].fmeasure
        total_rl += scores['rougeL'].fmeasure
    return total_r1 / 100, total_rl / 100

# 🔹 Perplexity
def calculate_perplexity():
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    total_loss = 0
    for input_enc, target_enc in tqdm(zip(input_data[:100], target_data[:100]), total=100, desc="[PPL]"):
        enc_ids = torch.tensor([input_enc["input_ids"]]).to(DEVICE)
        enc_mask = torch.tensor([input_enc["attention_mask"]]).to(DEVICE)

        dec_ids = torch.tensor([target_enc["input_ids"]]).to(DEVICE)
        dec_input = dec_ids[:, :-1]
        dec_target = dec_ids[:, 1:]

        with torch.no_grad():
            output = model(enc_ids, enc_mask, dec_input)
            loss = criterion(output.reshape(-1, output.size(-1)), dec_target.reshape(-1))
        total_loss += loss.item()
    return torch.exp(torch.tensor(total_loss / 100)).item()

# 🔹 Accuracy
def calculate_accuracy():
    total_tokens, correct_tokens = 0, 0
    for input_enc, target_enc in tqdm(zip(input_data[:100], target_data[:100]), total=100, desc="[ACC]"):
        question = tokenizer.decode(input_enc["input_ids"], skip_special_tokens=True)
        reference = tokenizer.decode(target_enc["input_ids"], skip_special_tokens=True).split()
        prediction = generate_response(question).split()
        min_len = min(len(reference), len(prediction))
        correct_tokens += sum([r == p for r, p in zip(reference[:min_len], prediction[:min_len])])
        total_tokens += min_len
    return correct_tokens / total_tokens if total_tokens > 0 else 0.0

# ✅ 예시 출력
example_inputs = ["안녕", "요즘 날씨 어때?", "오늘 기분이 좋아", "영화 추천해줘"]
print("\n[예시 출력]")
for sent in example_inputs:
    print(f"Q: {sent}\nA: {generate_response(sent)}\n")

# ✅ 평가 실행
print("[INFO] 모델 평가 시작...")
bleu = calculate_bleu_score()
rouge1, rougeL = calculate_rouge_score()
ppl = calculate_perplexity()
acc = calculate_accuracy()

print("\n[✅ 성능 평가 결과]")
print(f"BLEU Score   : {bleu:.4f}")
print(f"ROUGE-1 F1   : {rouge1:.4f}")
print(f"ROUGE-L F1   : {rougeL:.4f}")
print(f"Perplexity   : {ppl:.4f}")
print(f"Accuracy     : {acc:.4f}")