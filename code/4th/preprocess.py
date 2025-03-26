# ✅ preprocess.py
import os
import re
import pandas as pd
import zipfile
import json
from transformers import AutoTokenizer
from tqdm import tqdm

# ✅ 설정
MAX_LEN = 64 # 답변을 너무 많이 자르는 것 같아서 128로 수정할까 고민 중
DATA_DIR = "data/Interview/Training/Labeling"
TOKENIZER_PATH = os.path.join("data/output", "tokenizer")
CORPUS_PATH = os.path.join("data/output", "corpus_tokenized.json")

# ✅ 데이터 정제
def clean_json_string(raw):
    # 허용되지 않는 제어 문자 제거
    return re.sub(r'[\x00-\x1f\x7f]', '', raw)

# ✅ 데이터 추출
def extract_qa_pairs():
    qa_pairs = []

    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if not file.endswith('.zip'):
                continue

            zip_path = os.path.join(root, file)
            print(f"[INFO] 🔍 처리 중: {zip_path}")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for json_file in sorted(zip_ref.namelist()):
                    if not json_file.endswith('.json'):
                        continue

                    try:
                        with zip_ref.open(json_file) as f:
                            raw = f.read().decode("utf-8", errors="ignore")
                            cleaned = clean_json_string(raw)
                            data = json.loads(cleaned)
                            df = pd.DataFrame(data)

                            question_raw = df.at['question', 'dataSet']['raw']['text']
                            answer_raw = df.at['answer', 'dataSet']['raw']['text']

                            if question_raw.strip() and answer_raw.strip():
                                qa_pairs.append((question_raw.strip(), answer_raw.strip()))
                    except Exception as e:
                        print(f"[WARN] ❗ {file}/{json_file} 처리 실패: {e}")

    print(f"[INFO] ✅ 추출된 QA 쌍 수: {len(qa_pairs)}")
    return qa_pairs

# ✅ 데이터 저장
def save_qa_pairs(qa_pairs):
    with open("data/output/processed_interview_data.json", "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        print("[INFO] 저장 완료: processed_interview_data.json")

# ✅ KoBERT 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", use_fast=False)
print("[INFO] ✅ KoBERT 토크나이저 로드 완료")
print(f"[INFO] 🔢 Vocab 크기: {tokenizer.vocab_size}")
print(f"[INFO] 🔠 예시 토큰화: {tokenizer.tokenize('기분이 안 좋아')}")
print(f"[INFO] 🔤 CLS ID: {tokenizer.cls_token_id}, SEP ID: {tokenizer.sep_token_id}, PAD ID: {tokenizer.pad_token_id}")

# ✅ KoBERT 토크나이징
def tokenize_and_encode(q_list, a_list):
    input_encodings = []
    target_encodings = []
    for q, a in tqdm(zip(q_list, a_list), total=len(q_list), desc="[TOKENIZING]"):
        q_ids = tokenizer.encode(q, max_length=MAX_LEN, padding="max_length", truncation=True)
        a_ids = tokenizer.encode(a, max_length=MAX_LEN, padding="max_length", truncation=True)
        input_encodings.append({"input_ids": q_ids})
        target_encodings.append({"input_ids": a_ids})
    return input_encodings, target_encodings

# ✅ 전처리 실행
def save_preprocessed_data():
    qa_pairs = extract_qa_pairs()
    q_list, a_list = zip(*qa_pairs)

    if len(q_list) == 0:
        print("[ERROR] 수집된 문장이 없습니다. 데이터를 확인해주세요.")
        return

    print("[INFO] ✨ 토크나이징 및 인코딩 시작...")
    input_encodings, target_encodings = tokenize_and_encode(q_list, a_list)

    print(f"[INFO] 🔍 샘플 QnA:\nQ: {q_list[0]}\nA: {a_list[0]}")
    print(f"[INFO] 🔍 샘플 인코딩:\nQ: {input_encodings[0]['input_ids']}\nA: {target_encodings[0]['input_ids']}")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump({"input": input_encodings, "target": target_encodings}, f, ensure_ascii=False, indent=2)

    tokenizer.save_vocabulary(TOKENIZER_PATH)
    print(f"[✅ 완료] 저장된 쌍: {len(input_encodings)}개")
    print(f"[✅ 완료] 토크나이저 저장 경로: {TOKENIZER_PATH}")
    print(f"[✅ 완료] 코퍼스 저장 경로: {CORPUS_PATH}")

if __name__ == "__main__":
    save_preprocessed_data()
