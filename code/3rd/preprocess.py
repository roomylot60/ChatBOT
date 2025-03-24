# ✅ preprocess.py (Korpora 기반)
import re
import os
import numpy as np
from tqdm import tqdm
from konlpy.tag import Okt
from gensim.models import Word2Vec
from Korpora import Korpora
import json

# 기본 설정
FILTERS = "([~.,!?\"':;)(])"
PAD, SOS, END, UNK = "<PAD>", "<SOS>", "<END>", "<UNK>"
MARKER = [PAD, SOS, END, UNK]
MAX_SEQ = 25
okt = Okt()
CHANGE_FILTER = re.compile(FILTERS)

def load_korpora_corpus():
    print("[INFO] Korpora에서 챗봇 및 다른 말뭉치 데이터 로드 중...")

    open_q, open_a = [], []
    nsmc_q, nsmc_a = [], []
    modu_q, modu_a = [], []

    try:
        Korpora.fetch("open_subtitles")
        open_subs = Korpora.load("open_subtitles")
        for pair in open_subs.train:
            if '\t' in pair.text:
                q, a = pair.text.split('\t')
                open_q.append(q.strip())
                open_a.append(a.strip())
        print(f"[OK] open_subtitles 데이터 로드 완료: {len(open_q)}쌍")
    except Exception as e:
        print(f"[FAIL] open_subtitles 로딩 실패: {e}")

    try:
        Korpora.fetch("nsmc")
        nsmc = Korpora.load("nsmc")
        nsmc_q = [pair.text.strip() for pair in nsmc.train]
        nsmc_a = ["좋아요" if int(pair.label) else "별로예요" for pair in nsmc.train]
        print(f"[OK] nsmc 데이터 로드 완료: {len(nsmc_q)}쌍")
    except Exception as e:
        print(f"[FAIL] nsmc 로딩 실패: {e}")

    try:
        korpora_root = os.path.abspath("data")
        modu = Korpora.load("modu_messenger", root_dir=korpora_root)
        modu_q = [line.text.strip() for line in modu.train]
        modu_a = ["응" for _ in modu_q]
        print(f"[OK] modu_messenger 수동 데이터 로드 완료: {len(modu_q)}쌍")
    except Exception as e:
        print(f"[FAIL] modu_messenger 로딩 실패: {e}")

    q_list = open_q + nsmc_q + modu_q
    a_list = open_a + nsmc_a + modu_a

    print(f"[INFO] 총 수집된 질문-응답 쌍: {len(q_list)}")

    return q_list, a_list

def tokenize_morphs(sentence):
    return okt.morphs(re.sub(CHANGE_FILTER, "", sentence.replace(" ", "")))

def build_corpus(q_list, a_list):
    print("[DEBUG] build_corpus(): 형태소 분석 시작")
    corpus = []
    all_sentences = q_list + a_list

    for i, sent in enumerate(tqdm(all_sentences, desc="🧼 Tokenizing")):
        tokens = tokenize_morphs(sent)
        if i < 3:  # 샘플 3개 출력
            print(f"[샘플 {i}] 원문: {sent}")
            print(f"[샘플 {i}] 형태소: {tokens}")
        corpus.append(tokens)

    print(f"[INFO] 전체 문장 수: {len(corpus)} | 예: {corpus[0] if corpus else '없음'}")
    return corpus

def build_vocab(corpus):
    print("[DEBUG] build_vocab(): vocab 생성 시작")
    vocab = {token: idx for idx, token in enumerate(MARKER)}

    for sentence in tqdm(corpus, desc="🔤 Vocab Building"):
        for word in sentence:
            if word not in vocab:
                vocab[word] = len(vocab)

    print(f"[INFO] vocab 크기: {len(vocab)}")
    print(f"[샘플 vocab] {list(vocab.items())[:10]}")
    return vocab

def save_vocab(vocab, path="vocab.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

def load_vocab(path="vocab.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def build_embedding_matrix(vocab, w2v_model, dim=300):
    embedding_matrix = np.random.normal(scale=0.6, size=(len(vocab), dim))
    for word, idx in vocab.items():
        if word in w2v_model.wv:
            embedding_matrix[idx] = w2v_model.wv[word]
    return embedding_matrix

def encode_sentences(sentences, vocab, max_seq_length=MAX_SEQ):
    output = []
    print("[INFO] encode_sentences 진행 중...")
    for s in tqdm(sentences, desc="Encoding"):
        tokens = tokenize_morphs(s)
        ids = [vocab.get(token, vocab[UNK]) for token in tokens][:max_seq_length]
        ids += [vocab[PAD]] * (max_seq_length - len(ids))
        output.append(ids)
    return np.array(output)

def decode_sentences(sentences, vocab, max_seq_length=MAX_SEQ):
    output = []
    print("[INFO] decode_sentences 진행 중...")
    for s in tqdm(sentences, desc="Decoding"):
        tokens = tokenize_morphs(s)
        ids = [vocab[SOS]] + [vocab.get(token, vocab[UNK]) for token in tokens][:max_seq_length - 1]
        ids += [vocab[PAD]] * (max_seq_length - len(ids))
        output.append(ids)
    return np.array(output)

def label_sentences(sentences, vocab, max_seq_length=MAX_SEQ):
    output = []
    print("[INFO] label_sentences 진행 중...")
    for s in tqdm(sentences, desc="Labeling"):
        tokens = tokenize_morphs(s)
        ids = [vocab.get(token, vocab[UNK]) for token in tokens][:max_seq_length - 1] + [vocab[END]]
        ids += [vocab[PAD]] * (max_seq_length - len(ids))
        output.append(ids)
    return np.array(output)