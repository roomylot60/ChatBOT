import os
import re
import numpy as np
import pandas as pd

from konlpy.tag import Okt

FILTERS = "([~.,!?\"':;)(])"
PAD = "<PAD>"
STD = "<SOS>"
END = "<EOS>"
UNK = "<UNK>"
MARKER =[PAD, STD, END, UNK]
MAX_SEQ = 25

# 데이터 로드 함수
def load_data(path):
    '''
    Args:
        path: String of data file(csv) path
    Returns:
        qst, ans : Lists of sentences of question ans answer
    '''
    df = pd.read_csv(path, encoding='utf-8')
    qst = list(df['Q'])
    ans = list(df['A'])
    return qst, ans

# 정규화 함수
def normalize(sentence, pattern):
    normalizer = re.compile(pattern)
    n_sentence = normalizer.sub("", sentence)
    return n_sentence

# 형태소 단위로 토큰화하는 함수
def morpheming(sentences):
    '''
    Args:
        sentences: List of string(sentences)
    Return:
        morpheme_seq: List of morphemes made from sentences removing special characters(FILTERS)
    '''
    morpheme_seq = list()
    okt = Okt()
    for sentence in sentences:
        sentence = re.sub(FILTERS, '', sentence)
        morpheme_seq.extend(okt.morphs(sentence))
    morpheme_seq = list(set(morpheme_seq))
    return morpheme_seq

def dictGenerator(path):
    '''
    Args:
    Returns:
        dictionary: List of morphemes made with questions and answers
    '''
    qst, ans = load_data(path)
    dictionary = list()
    dictionary.extend(morpheming(qst))
    dictionary.extend(morpheming(ans))
    return dictionary

def vocabLoader(path, vocab_path):
    '''
    '''
    if not os.path.exists(vocab_path):
        words = dictGenerator(path)
        words[:0] = MARKER
        with open(vocab_path, 'w', encoding='utf-8') as vocabulary_file:
            for word in words:
                vocabulary_file.write(word + '\n')
    else:
        words = list()
        with open(vocab_path, 'r', encoding='utf-8') as vocabulary_file:
            lines = vocabulary_file.readlines()
            for line in lines:
                words.append(line.strip())
    return words

def indexing(words):
    words_index = dict([(word, i+1) for i, word in range(enumerate(words))])
    return words_index

def encoding(sentences):
    '''
    Args:
    Returns: Encoded vector(List) made of vocabulary index
    '''
    for sentence in sentences:
        sentence
    return