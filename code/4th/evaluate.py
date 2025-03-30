# ✅ evaluate.py (KoBERT + Transformer Decoder 평가)
import os
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel
from train import TransformerDecoderModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tqdm import tqdm

# 설정
MODEL_PATH = "models/layer_6_epoch_12_kobert.pt"
CORPUS_PATH = "data/output/corpus_valid.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 토크나이저 및 KoBERT 인코더
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", use_fast=False, trust_remote_code=True)
pad_id = tokenizer.pad_token_id
vocab_size = tokenizer.vocab_size
kobert_enc = BertModel.from_pretrained("monologg/kobert", trust_remote_code=True)

# ✅ 평가용 디코더 모델 불러오기
model = TransformerDecoderModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("[INFO] ✅ 훈련된 모델 로드 완료")

# ✅ 샘플 데이터 로드
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
inputs = data["input"]
targets = data["target"]

# ✅ 샘플 생성 함수
def generate_response(question_ids):
    input_ids = torch.tensor([question_ids]).to(DEVICE)
    attention_mask = (input_ids != pad_id).long()
    generated = [tokenizer.cls_token_id]

    for _ in range(64):
        dec_input = torch.tensor([generated]).to(DEVICE)
        output = model(input_ids, attention_mask, dec_input)
        next_token = output.argmax(-1)[0, -1].item()
        if next_token == tokenizer.sep_token_id or len(generated) >= 64:
            break
        generated.append(next_token)
    return tokenizer.decode(generated[1:], skip_special_tokens=True)

# ✅ 평가 지표 초기화
smooth = SmoothingFunction().method1
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

bleu_scores = []
rouge1_scores = []
rougeL_scores = []
acc = 0
ppl_total = 0
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

# ✅ 전체 평가 루프
print("\n[INFO] 🧪 성능 평가 시작...\n")
for i in tqdm(range(len(inputs)), desc="Evaluating"):
    input_ids = inputs[i]["input_ids"]
    target_ids = targets[i]["input_ids"]

    question = tokenizer.decode(input_ids, skip_special_tokens=True)
    answer_gt = tokenizer.decode(target_ids, skip_special_tokens=True)
    answer_gen = generate_response(input_ids)

    # BLEU, ROUGE
    bleu = sentence_bleu([answer_gt.split()], answer_gen.split(), smoothing_function=smooth)
    rouge = scorer.score(answer_gt, answer_gen)

    bleu_scores.append(bleu)
    rouge1_scores.append(rouge['rouge1'].fmeasure)
    rougeL_scores.append(rouge['rougeL'].fmeasure)

    if answer_gt.strip() == answer_gen.strip():
        acc += 1

    # Perplexity
    input_tensor = torch.tensor([input_ids]).to(DEVICE)
    tgt_tensor = torch.tensor([target_ids]).to(DEVICE)
    attn_mask = (input_tensor != pad_id).long()
    dec_input = tgt_tensor[:, :-1]
    dec_target = tgt_tensor[:, 1:]
    with torch.no_grad():
        output = model(input_tensor, attn_mask, dec_input)
        loss = loss_fn(output.view(-1, vocab_size), dec_target.reshape(-1))
        ppl_total += torch.exp(loss).item()

# ✅ 결과 출력
print("\n[✅ 성능 평가 결과]")
print(f"BLEU Score   : {sum(bleu_scores) / len(bleu_scores):.4f}")
print(f"ROUGE-1 F1   : {sum(rouge1_scores) / len(rouge1_scores):.4f}")
print(f"ROUGE-L F1   : {sum(rougeL_scores) / len(rougeL_scores):.4f}")
print(f"Accuracy     : {acc / len(inputs):.4f}")
print(f"Perplexity   : {ppl_total / len(inputs):.4f}")

# ✅ 샘플 출력
print("\n[INFO] 💬 샘플 질문에 대한 응답 생성\n")
for i in range(5):
    input_ids = inputs[i]["input_ids"]
    target_ids = targets[i]["input_ids"]

    question = tokenizer.decode(input_ids, skip_special_tokens=True)
    answer_gt = tokenizer.decode(target_ids, skip_special_tokens=True)
    answer_gen = generate_response(input_ids)

    print(f"🟨 질문: {question}")
    print(f"🟦 실제 응답: {answer_gt}")
    print(f"🟧 생성 응답: {answer_gen}")
    print("-" * 80)
