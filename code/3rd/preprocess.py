# ✅ preprocess.py (KoBERT 전용)
import os
import json
from tqdm import tqdm
from tokenization_kobert import KoBertTokenizer
from Korpora import Korpora

# 설정
MAX_SEQ_LEN = 64
DATA_DIR = "data"
TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus_tokenized.json")

# ✅ KoBERT 토크나이저 로드
tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")
print("[INFO] ✅ KoBERT 토크나이저 로드 완료")
print(f"[INFO] 🔢 Vocab 크기: {tokenizer.vocab_size}")
print(f"[INFO] 🔠 예시 토큰화: {tokenizer.tokenize('기분이 안 좋아')}")
print(f"[INFO] 🔤 CLS ID: {tokenizer.cls_token_id}, SEP ID: {tokenizer.sep_token_id}, PAD ID: {tokenizer.pad_token_id}")

# ✅ Korpora에서 데이터 로드
def process_korpora():
    print("[INFO] 📦 Korpora 데이터 로드 시작")
    q_list, a_list = [], []

    try:
        Korpora.fetch("open_subtitles")
        open_subs = Korpora.load("open_subtitles")
        open_q, open_a = [], []
        for pair in tqdm(open_subs.train, desc="[LOAD open_subtitles]"):
            if '\t' in pair.text:
                q, a = pair.text.split('\t')
                open_q.append(q.strip())
                open_a.append(a.strip())
        q_list += open_q
        a_list += open_a
        print(f"[OK] ✅ open_subtitles 로드 완료: {len(open_q)} 쌍")
    except Exception as e:
        print(f"[FAIL] ❌ open_subtitles 로딩 실패: {e}")

    try:
        nsmc = Korpora.load("nsmc")
        nsmc_q = [line.text.strip() for line in tqdm(nsmc.train, desc="[LOAD nsmc]")]
        nsmc_a = ["좋아요" if int(line.label) else "별로예요" for line in nsmc.train]
        q_list += nsmc_q
        a_list += nsmc_a
        print(f"[OK] ✅ NSMC 로드 완료: {len(nsmc_q)} 쌍")
    except Exception as e:
        print(f"[FAIL] ❌ NSMC 로딩 실패: {e}")

    try:
        root_dir = os.path.abspath("data")
        modu = Korpora.load("modu_messenger", root_dir=root_dir)
        modu_q = [line.text.strip() for line in tqdm(modu.train, desc="[LOAD modu_messenger]")]
        modu_a = ["응" for _ in modu_q]
        q_list += modu_q
        a_list += modu_a
        print(f"[OK] ✅ MODU 로드 완료: {len(modu_q)} 쌍")
    except Exception as e:
        print(f"[FAIL] ❌ MODU 로딩 실패: {e}")

    print(f"[INFO] 총 수집된 데이터: {len(q_list)} 쌍")
    return q_list, a_list

# ✅ KoBERT 토크나이징
def tokenize_and_encode(q_list, a_list):
    input_encodings = []
    target_encodings = []
    for q, a in tqdm(zip(q_list, a_list), total=len(q_list), desc="[TOKENIZING]"):
        q_ids = tokenizer.encode(q, max_length=MAX_SEQ_LEN, padding="max_length", truncation=True)
        a_ids = tokenizer.encode(a, max_length=MAX_SEQ_LEN, padding="max_length", truncation=True)
        input_encodings.append({"input_ids": q_ids})
        target_encodings.append({"input_ids": a_ids})
    return input_encodings, target_encodings

# ✅ 전처리 실행
def save_preprocessed_data():
    q_list, a_list = process_korpora()

    if len(q_list) == 0:
        print("[ERROR] 수집된 문장이 없습니다. 데이터를 확인해주세요.")
        return

    print("[INFO] ✨ 토크나이징 및 인코딩 시작...")
    input_encodings, target_encodings = tokenize_and_encode(q_list, a_list)

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
