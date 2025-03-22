# ✅ preprocess.py
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from konlpy.tag import Okt
from gensim.models import Word2Vec

FILTERS = "([~.,!?\"':;)(])"
PAD, STD, END, UNK = "<PAD>", "<SOS>", "<END>", "<UNK>"
MARKER = [PAD, STD, END, UNK]
CHANGE_FILTER = re.compile(FILTERS)
MAX_SEQ = 25

okt = Okt()

def load_data(path):
    data_df = pd.read_csv(path)
    return list(data_df['Q']), list(data_df['A'])

def tokenize_morphs(sentence):
    return okt.morphs(re.sub(CHANGE_FILTER, "", sentence.replace(" ", "")))

def build_corpus(q_list, a_list):
    return [tokenize_morphs(sent) for sent in q_list + a_list]

def build_vocab(corpus):
    vocab = {token: idx for idx, token in enumerate(MARKER)}
    for sentence in corpus:
        for word in sentence:
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

def build_embedding_matrix(vocab, w2v_model, dim=300):
    embedding_matrix = np.random.normal(scale=0.6, size=(len(vocab), dim))
    for word, idx in vocab.items():
        if word in w2v_model.wv:
            embedding_matrix[idx] = w2v_model.wv[word]
    return embedding_matrix

def encode_sentences(sentences, vocab, max_seq_length=25):  # ✅ 인자 이름 수정
    output = []
    for s in sentences:
        tokens = tokenize_morphs(s)
        ids = [vocab.get(token, vocab[UNK]) for token in tokens]
        ids = ids[:max_seq_length]
        ids += [vocab[PAD]] * (max_seq_length - len(ids))
        output.append(ids)
    return np.array(output)

def decode_sentences(sentences, vocab, max_seq_length=25):
    output = []
    for s in sentences:
        tokens = tokenize_morphs(s)
        ids = [vocab[STD]] + [vocab.get(token, vocab[UNK]) for token in tokens]
        ids = ids[:max_seq_length]
        ids += [vocab[PAD]] * (max_seq_length - len(ids))
        output.append(ids)
    return np.array(output)

def label_sentences(sentences, vocab, max_seq_length=25):
    output = []
    for s in sentences:
        tokens = tokenize_morphs(s)
        ids = [vocab.get(token, vocab[UNK]) for token in tokens]
        ids = ids[:max_seq_length - 1] + [vocab[END]]  # 최대 길이 - 1까지 자르고 END 토큰 추가
        ids += [vocab[PAD]] * (max_seq_length - len(ids))  # 패딩 추가
        output.append(ids)
    return np.array(output)
