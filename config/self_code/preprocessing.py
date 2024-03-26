import os
import re
import numpy as np
import pandas as pd

from konlpy.tag import Okt

FILTERS = "([~.,!?\"':;)(])"

MAX_SEQ = 25

# 데이터 로드 함수
def load_data(path, file):
    '''
    Args:
        path: String of data file(csv) path
    Returns:
        qst, ans: Lists of sentences of question ans answer
    '''
    df = pd.read_csv(os.path.join(path, file), encoding='utf-8')
    qst = list(df['Q'])
    ans = list(df['A'])
    return qst, ans

# 정규화 함수
def normalize(sentence):
    '''
    Args:
        sentence: String
    Return:
        n_sentence: Normalized String
    '''
    K_pattern = r'[^ ?,.!A-Za-z0-9가-힣+]'
    normalizer = re.compile(K_pattern)
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
        sentence = normalize(sentence)
        morpheme_seq.extend(okt.morphs(sentence))
    morpheme_seq = list(set(morpheme_seq))
    return morpheme_seq

PAD = 0
SOS = 1
EOS = 2
UNK = 3

word2index = {
    "<PAD>":PAD,
    "<SOS>":SOS,
    "<EOS>":EOS,
    "<UNK>":UNK
}
index2word = {
    PAD:"<PAD>",
    SOS:"<SOS>",
    EOS:"<EOS>",
    UNK:"<UNK>"
}

def dictGenerator(path):
    '''
    Args:
    Returns:
        dictionary: List of morphemes made with questions and answers
    '''
    qst, ans = load_data(path)
    dictionary = list()
    dictionary.extend(qst)
    dictionary.extend(ans)
    dictionary = morpheming(dictionary)
    dictionary[:0] = MARKER
    return dictionary

def vocabLoader(path, vocab_path):
    '''
    Args:
        path: String of data path
        vocab_path: String of dictionary path
    Return:
        words: List of morphemes loaded from vocabulary.txt file
    '''
    if not os.path.exists(vocab_path):
        words = dictGenerator(path)
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
