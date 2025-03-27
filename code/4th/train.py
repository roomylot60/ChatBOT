# ✅ train.py (KoBERT + Transformer Decoder)
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertModel

# 설정
CORPUS_PATH = "data/output/corpus_tokenized.json"
MODEL_SAVE_PATH = "models/kobert_chatbot_epoch_5.pt"
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] ✅ 사용 디바이스: {DEVICE}")

# ✅ KoBERT 인코더 로드
kobert_enc = BertModel.from_pretrained("monologg/kobert", trust_remote_code=True)
print(f"[INFO] ✅ KoBERT 모델 로드 완료 - Vocab Size: {kobert_enc.config.vocab_size}")
try:
    tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", use_fast=False, trust_remote_code=True)
    print(f"[INFO] 🔠 토크나이저 로드 완료 - Vocab Size: {tokenizer.vocab_size}")
    print(f"[INFO] ⛳ PAD ID: {tokenizer.pad_token_id}, CLS ID: {tokenizer.cls_token_id}, SEP ID: {tokenizer.sep_token_id}")
except Exception as e:
    print(f"[ERROR] ❌ 토크나이저 로딩 실패: {e}")
    exit(1)
vocab_size = tokenizer.vocab_size
pad_id = tokenizer.pad_token_id

# ✅ 디코더 정의 (Transformer)
class TransformerDecoderModel(nn.Module):
    def __init__(self, hidden_size=768, vocab_size=vocab_size, num_layers=4, num_heads=8, ff_dim=1024):
        super().__init__()
        self.bert = kobert_enc  # KoBERT 인코더
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

# ✅ 데이터셋 정의
class ChatDataset(Dataset):
    def __init__(self, data):
        self.inputs = data["input"]
        self.targets = data["target"]

    def __len__(self): return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx]["input_ids"]),
            torch.tensor(self.targets[idx]["input_ids"])
        )

def collate_fn(batch):
    input_ids, target_ids = zip(*batch)
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    target_ids = nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=pad_id)
    attention_mask = (input_ids != pad_id).long()
    return input_ids, attention_mask, target_ids

# ✅ 데이터셋 로드
try:
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    dataset = ChatDataset(data)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    print(f"[INFO] 📂 전처리된 데이터 로드 완료 - 총 샘플 수: {len(dataset.inputs)}")
except Exception as e:
    print(f"[ERROR] ❌ 전처리된 데이터 로딩 실패: {e}")
    exit(1)

# ✅ 학습 설정
model = TransformerDecoderModel().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
print(f"[INFO] ✅ 모델 구조 생성 완료 - 총 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

step = 0 # 배치 인덱스 추적용 변수

# ✅ 학습 루프
for epoch in range(5):
    model.train()
    total_loss = 0
    print(f"\n[Epoch {epoch+1}] 시작")

    for enc_ids, enc_mask, tgt_ids in tqdm(dataloader, desc=f"[Training {epoch+1}]"):
        enc_ids, enc_mask, tgt_ids = enc_ids.to(DEVICE), enc_mask.to(DEVICE), tgt_ids.to(DEVICE)
        dec_input = tgt_ids[:, :-1]
        dec_target = tgt_ids[:, 1:]
        
        try:
            output = model(enc_ids, enc_mask, dec_input)
            loss = loss_fn(output.view(-1, vocab_size), dec_target.reshape(-1))

        except Exception as e:
            print(f"[ERROR] ❌ 모델 순전파 오류 at step {step}: {e}")
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # 샘플 디버깅 출력 (1개 배치만)
        if step == 0:
            print(f"\n[DEBUG] 🔍 첫 배치 샘플:")
            print("🟩 질문 토큰 ID:", enc_ids[0].tolist())
            print("🟦 응답(입력) 토큰 ID:", dec_input[0].tolist())
            print("🟥 응답(타깃) 토큰 ID:", dec_target[0].tolist())
            print("🧾 예시 질문:", tokenizer.convert_ids_to_tokens(enc_ids[0].tolist()))
            print("🧾 예시 응답 입력:", tokenizer.convert_ids_to_tokens(dec_input[0].tolist()))
        
        step += 1

    avg_loss = total_loss / len(dataloader)
    print(f"[Epoch {epoch+1}] 평균 손실: {avg_loss:.4f}")

# ✅ 모델 저장
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"[✅ 저장 완료] 경로: {MODEL_SAVE_PATH}")