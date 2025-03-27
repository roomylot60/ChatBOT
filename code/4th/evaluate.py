# ✅ evaluate.py (KoBERT 인코더 + Transformer 디코더 평가용)
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, BertModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# 설정
MODEL_PATH = "models/kobert_chatbot_epoch_5.pt"
CORPUS_PATH = "data/output/corpus_tokenized.json"
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] ✅ 디바이스: {DEVICE}")

# ✅ 모델 로드
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", use_fast=False, trust_remote_code=True)
kobert = BertModel.from_pretrained("monologg/kobert", trust_remote_code=True)

class TransformerDecoderModel(nn.Module):
    def __init__(self, hidden_size=768, vocab_size=tokenizer.vocab_size, num_layers=4, num_heads=8, ff_dim=1024):
        super().__init__()
        self.bert = kobert
        self.decoder_embedding = nn.Embedding(vocab_size, hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, enc_input_ids, enc_mask, dec_input_ids):
        enc_output = self.bert(input_ids=enc_input_ids, attention_mask=enc_mask).last_hidden_state
        dec_emb = self.decoder_embedding(dec_input_ids)
        tgt_mask = self.generate_square_subsequent_mask(dec_input_ids.size(1)).to(dec_input_ids.device)
        dec_output = self.decoder(tgt=dec_emb, memory=enc_output, tgt_mask=tgt_mask)
        return self.output_layer(dec_output)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

model = TransformerDecoderModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("[INFO] ✅ 학습된 모델 로드 완료")

# ✅ 디코딩 함수
def decode_response(input_text):
    enc = tokenizer.encode_plus(
        input_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    )
    enc_input_ids = enc["input_ids"].to(DEVICE)
    enc_mask = enc["attention_mask"].to(DEVICE)

    dec_input_ids = torch.tensor([[tokenizer.cls_token_id]], dtype=torch.long).to(DEVICE)
    result = []

    for _ in range(MAX_LEN):
        with torch.no_grad():
            outputs = model(enc_input_ids, enc_mask, dec_input_ids)
        next_token_id = outputs[:, -1, :].argmax(-1).item()
        if next_token_id == tokenizer.sep_token_id:
            break
        result.append(next_token_id)
        dec_input_ids = torch.cat([dec_input_ids, torch.tensor([[next_token_id]]).to(DEVICE)], dim=1)

    return tokenizer.decode(result, skip_special_tokens=True)

# ✅ 전처리된 데이터 로드
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

sample_inputs = data["input"][:100]
sample_targets = data["target"][:100]

smoothie = SmoothingFunction().method4
bleu_total, rouge1_total, rougel_total, acc_total = 0, 0, 0, 0

rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

print("\n[INFO] 모델 평가 시작...")

for idx in tqdm(range(len(sample_inputs))):
    input_ids = sample_inputs[idx]["input_ids"]
    target_ids = sample_targets[idx]["input_ids"]

    q = tokenizer.decode(input_ids, skip_special_tokens=True)
    ref = tokenizer.decode(target_ids, skip_special_tokens=True)
    hyp = decode_response(q)

    bleu = sentence_bleu([ref.split()], hyp.split(), smoothing_function=smoothie)
    scores = rouge.score(ref, hyp)

    bleu_total += bleu
    rouge1_total += scores["rouge1"].fmeasure
    rougel_total += scores["rougeL"].fmeasure
    acc_total += int(ref.strip() == hyp.strip())

# ✅ 평균 성능 출력
print("\n[✅ 성능 평가 결과]")
print(f"BLEU Score   : {bleu_total / len(sample_inputs):.4f}")
print(f"ROUGE-1 F1   : {rouge1_total / len(sample_inputs):.4f}")
print(f"ROUGE-L F1   : {rougel_total / len(sample_inputs):.4f}")
print(f"Accuracy     : {acc_total / len(sample_inputs):.4f}")

def generate_response(input_text):
    enc = tokenizer.encode_plus(input_text, return_tensors='pt', max_length=64, padding='max_length', truncation=True)
    input_ids = enc['input_ids'].to(DEVICE)
    attention_mask = enc['attention_mask'].to(DEVICE)

    decoder_input = torch.tensor([[tokenizer.cls_token_id]], device=DEVICE)
    result = []

    for _ in range(64):
        with torch.no_grad():
            output = model(input_ids, attention_mask, decoder_input)
        next_token = output.argmax(-1)[:, -1].item()
        if next_token == tokenizer.sep_token_id:
            break
        result.append(next_token)
        decoder_input = torch.cat([decoder_input, torch.tensor([[next_token]], device=DEVICE)], dim=1)

    return tokenizer.decode(result, skip_special_tokens=True)

prompt = ["오늘 기분이 어때?", "잘 지내셨습니까?"]
for x in prompt:
    print("[Prompt] : ", x)
    response = generate_response(x)
    print("[BOT] :", response)
