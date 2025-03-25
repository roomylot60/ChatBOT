# # ✅ train.py (Korpora 기반 + Word2Vec + Transformer)
# import os
# import json
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# from KoBERT import KoBERT_Transformer
# from tokenization_kobert import KoBertTokenizer

# # 경로
# DATA_DIR = "data"
# TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer")
# CORPUS_PATH = os.path.join(DATA_DIR, "corpus_tokenized.json")
# MODEL_SAVE_PATH = "models/kobert_chatbot.pt"

# # 디바이스 설정
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"[INFO] ✅ 사용 디바이스: {device}")

# # 토크나이저 로드
# try:
#     tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")
#     print(f"[INFO] 🔠 토크나이저 로드 완료 - Vocab Size: {tokenizer.vocab_size}")
#     print(f"[INFO] ⛳ PAD ID: {tokenizer.pad_token_id}, CLS ID: {tokenizer.cls_token_id}, SEP ID: {tokenizer.sep_token_id}")
# except Exception as e:
#     print(f"[ERROR] ❌ 토크나이저 로딩 실패: {e}")
#     exit(1)

# PAD_ID = tokenizer.pad_token_id

# # 전처리된 데이터 로드
# try:
#     with open(CORPUS_PATH, "r", encoding="utf-8") as f:
#         data = json.load(f)
#         input_data = data["input"]
#         target_data = data["target"]
#     print(f"[INFO] 📂 전처리된 데이터 로드 완료 - 총 샘플 수: {len(input_data)}")
# except Exception as e:
#     print(f"[ERROR] ❌ 전처리된 데이터 로딩 실패: {e}")
#     exit(1)
    
# class ChatDataset(Dataset):
#     def __init__(self, data):
#         self.inputs = data["input"]
#         self.targets = data["target"]

#     def __len__(self): return len(self.inputs)

#     def __getitem__(self, idx):
#         return (
#             torch.tensor(self.inputs[idx]["input_ids"]),
#             torch.tensor(self.inputs[idx]["attention_mask"]),
#             torch.tensor(self.targets[idx]["input_ids"]),
#         )

# def collate_fn(batch):
#     q_ids, q_mask, a_ids = zip(*batch)
#     return (
#         nn.utils.rnn.pad_sequence(q_ids, batch_first=True, padding_value=tokenizer.pad_token_id),
#         nn.utils.rnn.pad_sequence(q_mask, batch_first=True, padding_value=0),
#         nn.utils.rnn.pad_sequence(a_ids, batch_first=True, padding_value=tokenizer.pad_token_id),
#     )

# dataset = ChatDataset(corpus)
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"[INFO] 현재 학습 디바이스: {device}")
# model = KoBERT_Transformer(decoder_vocab_size=tokenizer.vocab_size).to(device)

# loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
# optimizer = optim.Adam(model.parameters(), lr=5e-5)

# for epoch in range(3):
#     model.train()
#     total_loss = 0
#     print(f"\n[Epoch {epoch+1}] 학습 시작")
#     for input_ids, attention_mask, target_ids in tqdm(dataloader):
#         input_ids, attention_mask, target_ids = input_ids.to(device), attention_mask.to(device), target_ids.to(device)
#         dec_input = target_ids[:, :-1]
#         dec_target = target_ids[:, 1:]

#         output = model(input_ids, attention_mask, dec_input)

#         loss = loss_fn(output.view(-1, output.size(-1)), dec_target.reshape(-1))
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"[Epoch {epoch+1}] 평균 Loss: {total_loss / len(dataloader):.4f}")

# # 저장
# os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
# torch.save(model.state_dict(), MODEL_SAVE_PATH)
# print("[✅ 모델 저장 완료]")

# ✅ train.py (Korpora 기반 + Word2Vec + Transformer)
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from KoBERT import KoBERT_Transformer
from tokenization_kobert import KoBertTokenizer

# 경로 설정
DATA_DIR = "data"
TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus_tokenized.json")
MODEL_SAVE_PATH = "models/kobert_chatbot.pt"

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] ✅ 사용 디바이스: {device}")

# 토크나이저 로드
try:
    tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")
    print(f"[INFO] 🔠 토크나이저 로드 완료 - Vocab Size: {tokenizer.vocab_size}")
    print(f"[INFO] ⛳ PAD ID: {tokenizer.pad_token_id}, CLS ID: {tokenizer.cls_token_id}, SEP ID: {tokenizer.sep_token_id}")
except Exception as e:
    print(f"[ERROR] ❌ 토크나이저 로딩 실패: {e}")
    exit(1)

PAD_ID = tokenizer.pad_token_id

# 전처리된 데이터 로드
try:
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        input_data = data["input"]
        target_data = data["target"]
    print(f"[INFO] 📂 전처리된 데이터 로드 완료 - 총 샘플 수: {len(input_data)}")
except Exception as e:
    print(f"[ERROR] ❌ 전처리된 데이터 로딩 실패: {e}")
    exit(1)

# 데이터셋 클래스
class ChatDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self): return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx]["input_ids"]),
            torch.tensor(self.targets[idx]["input_ids"])
        )

# Collate 함수
def collate_fn(batch):
    input_ids, target_ids = zip(*batch)
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=PAD_ID)
    target_ids = nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=PAD_ID)
    return input_ids, target_ids

# 데이터로더
dataset = ChatDataset(input_data, target_data)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# 모델 선언
model = KoBERT_Transformer(decoder_vocab_size=tokenizer.vocab_size).to(device)
print(f"[INFO] ✅ 모델 구조 생성 완료 - 총 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

# 손실함수 & 옵티마이저
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
optimizer = optim.Adam(model.parameters(), lr=5e-5)

# 학습 루프
for epoch in range(3):
    model.train()
    total_loss = 0
    print(f"\n[INFO] 🔁 Epoch {epoch + 1} 시작")

    for step, (input_ids, target_ids) in enumerate(tqdm(dataloader, desc=f"[TRAIN EPOCH {epoch+1}]")):
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)

        enc_mask = (input_ids != PAD_ID).long()
        dec_input = target_ids[:, :-1]
        dec_target = target_ids[:, 1:]

        try:
            output = model(input_ids, enc_mask, dec_input)
            loss = loss_fn(output.view(-1, output.size(-1)), dec_target.reshape(-1))
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
            print("🟩 질문 토큰 ID:", input_ids[0].tolist())
            print("🟦 응답(입력) 토큰 ID:", dec_input[0].tolist())
            print("🟥 응답(타깃) 토큰 ID:", dec_target[0].tolist())
            print("🧾 예시 질문:", tokenizer.convert_ids_to_tokens(input_ids[0].tolist()))
            print("🧾 예시 응답 입력:", tokenizer.convert_ids_to_tokens(dec_input[0].tolist()))

    avg_loss = total_loss / len(dataloader)
    print(f"[Epoch {epoch + 1}] 📉 평균 Loss: {avg_loss:.4f}")

# 모델 저장
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"[✅ 저장 완료] 모델이 '{MODEL_SAVE_PATH}'에 저장되었습니다.")
