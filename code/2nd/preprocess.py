# ✅ preprocess.py (Korpora 기반)
import os
import re
import json
import numpy as np
from tqdm import tqdm
from konlpy.tag import Okt
from gensim.models import Word2Vec
from Korpora import Korpora

# ✅ 설정
PAD, SOS, EOS, UNK = "<PAD>", "<SOS>", "<EOS>", "<UNK>"
MARKERS = [PAD, SOS, EOS, UNK]
MAX_SEQ = 25
okt = Okt()
FILTER = re.compile("[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9]")

VOCAB_PATH = "data/vocab.json"
CORPUS_PATH = "data/corpus_tokenized.json"
EMBEDDING_PATH = "models/embedding_matrix.npy"
W2V_MODEL_PATH = "models/word2vec_ko.model"


# ✅ 형태소 단위 토크나이징
def tokenize_morphs(sentence):
    return okt.morphs(re.sub(FILTER, "", sentence.replace(" ", "")))


# ✅ Korpora에서 데이터 로드
def load_korpora_corpus():
    print("[INFO] Korpora 데이터 로드 시작")
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
        print(f"[OK] open_subtitles 로드 완료: {len(open_q)} 쌍")
    except Exception as e:
        print(f"[FAIL] open_subtitles 로딩 실패: {e}")

    try:
        nsmc = Korpora.load("nsmc")
        nsmc_q = [line.text.strip() for line in tqdm(nsmc.train, desc="[LOAD nsmc]")]
        nsmc_a = ["좋아요" if int(line.label) else "별로예요" for line in nsmc.train]
        q_list += nsmc_q
        a_list += nsmc_a
        print(f"[OK] NSMC 로드 완료: {len(nsmc_q)} 쌍")
    except Exception as e:
        print(f"[FAIL] NSMC 로딩 실패: {e}")

    try:
        root_dir = os.path.abspath("data")
        modu = Korpora.load("modu_messenger", root_dir=root_dir)
        modu_q = [line.text.strip() for line in tqdm(modu.train, desc="[LOAD modu_messenger]")]
        modu_a = ["응" for _ in modu_q]
        q_list += modu_q
        a_list += modu_a
        print(f"[OK] MODU 로드 완료: {len(modu_q)} 쌍")
    except Exception as e:
        print(f"[FAIL] MODU 로딩 실패: {e}")

    print(f"[INFO] 총 수집된 데이터: {len(q_list)} 쌍")
    return q_list, a_list


# ✅ 토크나이즈 전체 코퍼스
def build_corpus(q_list, a_list):
    corpus = []
    for q, a in tqdm(zip(q_list, a_list), total=len(q_list), desc="[TOKENIZE CORPUS]"):
        q_tokens = tokenize_morphs(q)
        a_tokens = tokenize_morphs(a)
        corpus.append(q_tokens)
        corpus.append(a_tokens)
    print(f"[INFO] 총 토큰화 문장 수: {len(corpus)}")
    return corpus


# ✅ 단어장 생성
def build_vocab(corpus):
    vocab = {tok: idx for idx, tok in enumerate(MARKERS)}
    for sentence in tqdm(corpus, desc="[BUILD VOCAB]"):
        for token in sentence:
            if token not in vocab:
                vocab[token] = len(vocab)
    print(f"[INFO] Vocab 크기: {len(vocab)}")
    return vocab


# ✅ 임베딩 매트릭스 생성 및 저장
def build_and_save_embedding_matrix(corpus, vocab, dim=300, w2v_path=W2V_MODEL_PATH, emb_path=EMBEDDING_PATH):
    print("[INFO] Word2Vec 훈련 시작")
    w2v_model = Word2Vec(sentences=corpus, vector_size=dim, window=5, min_count=1, workers=4)
    w2v_model.save(w2v_path)
    print(f"[INFO] Word2Vec 저장 완료: {w2v_path}")

    matrix = np.random.normal(0, 1, (len(vocab), dim))
    for word, idx in vocab.items():
        if word in w2v_model.wv:
            matrix[idx] = w2v_model.wv[word]
    np.save(emb_path, matrix)
    print(f"[INFO] Embedding Matrix 저장 완료: {emb_path}")
    return matrix


# ✅ 인코딩 함수
def encode_sentences(sentences, vocab, max_len=MAX_SEQ):
    encoded = []
    for s in tqdm(sentences, desc="[ENCODE]"):
        tokens = tokenize_morphs(s)
        ids = [vocab.get(token, vocab[UNK]) for token in tokens][:max_len]
        ids += [vocab[PAD]] * (max_len - len(ids))
        encoded.append(ids)
    return encoded

# ✅ 디코딩 함수
def decode_sentences(sentences, vocab):
    decoded = []
    for sentence in tqdm(sentences, desc="[DECODE]"):
        tokens = tokenize_morphs(sentence)
        ids = [vocab.get(token, vocab[UNK]) for token in tokens[:MAX_SEQ-1]]
        ids = [vocab[SOS]] + ids
        ids += [vocab[PAD]] * (MAX_SEQ - len(ids))
        decoded.append(ids)
    return np.array(decoded)

# ✅ 레이블 함수
def label_sentences(sentences, vocab, max_len=MAX_SEQ):
    labeled = []
    for s in tqdm(sentences, desc="[LABEL]"):
        tokens = tokenize_morphs(s)
        ids = [vocab.get(token, vocab[UNK]) for token in tokens][:max_len-1]
        ids = [vocab[SOS]] + ids + [vocab[EOS]]
        ids += [vocab[PAD]] * (max_len + 1 - len(ids))
        labeled.append(ids)
    return labeled


# ✅ 저장 및 로드 함수들
def save_vocab(vocab):
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"[INFO] Vocab 저장 완료: {VOCAB_PATH}")

def save_corpus(corpus):
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)
    print(f"[INFO] Corpus 저장 완료: {CORPUS_PATH}")

def load_vocab():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_corpus():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)