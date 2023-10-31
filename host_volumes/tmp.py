
# Module import
import os
import re
import json

import numpy as np
import pandas as pd

from tqdm import tqdm
from konlpy.tag import Okt

# Settings(특수문자 제거, 특수 토큰 정의)
FILTERS = "([~.,!?\"':;)(])"
PAD = "<PAD>"
STD = "<SOS>"
END = "<END>"
UNK = "<UNK>"

PAD_INDEX = 0
STD_INDEX = 1
END_INDEX = 2
UNK_INDEX = 3

MARKER =[PAD, STD, END, UNK]
CHANGE_FILTER = re.compile(FILTERS)

MAX_SEQ = 25

def data_tokenizer(data):
    words = [] # array to append tokens
    for sentence in data:
        # FILTERS 내의 값들을 정규화 표현식을 통해서 모두 제거
        sentence = re.sub(CHANGE_FILTER, "", sentence)
        for word in sentence.split():
            words.append(word) # 각 문장을 단어로 분리하고 이를 배열에 추가
    return [word for word in words if word] # tokeninzing, 정규화를 거친 값들을 반환

def make_vocabulary(vocab_list):
    chr2idx = {chr:idx for idx, chr in enumerate(vocab_list)}
    idx2chr = {idx:chr for idx, chr in enumerate(vocab_list)}
    return chr2idx, idx2chr

def prepro_like_morpheme(data):
    morph_analyzer = Okt()
    result_data = []
    for seq in tqdm(data): # 데이터에 대한 진행도 확인
        morphemic_seq = " ".join(morph_analyzer.morphs(seq.replace(' ', '')))
        result_data.append(morphemic_seq)
    return result_data


def load_vocabulary(path, vocab_path, tokenize_as_morph=False):
    # List for dictionary
    vocab_list = []
    # Checking the file presence on the path
    if not os.path.exists(vocab_path):
        if (os.path.exists(path)):
            # load data
            data_df = pd.read_csv(path, encoding='utf-8')
            # array for questions and answers
            qst, ans = list(data_df['Q']), list(data_df['A'])
            if tokenize_as_morph:
                # 형태소에 따른 tokeninzing
                qst = prepro_like_morpheme(qst)
                ans = prepro_like_morpheme(ans)
            # data = []
            # .extend(전달값) : 전달값(iterable)을 하나씩 분해해서(리스트 구성을 제거하고) 리스트에 각 원소로 저장(!! .append()와 구분)
            vocab_list.extend(qst)
            vocab_list.extend(ans)
            
            # vocab_list = data_tokenizer(data)
            return qst, ans
            # vocab_list = list(set(vocab_list)) # 중복 단어 제거를 위해 set

            # # MARKER를 dictionary에 추가
            # # 순서대로 넣기 위해서 인덱스 0 에 추가
            # # PAD = "<PADDING>"
            # # STD = "<START>"
            # # END = "<END>"
            # # UNK = "<UNKOWN>"
            # vocab_list[:0] = MARKER

    
        
            # # 리스트로 된 단어 사전을 vocabulary_file에 저장
            # # 각 단어는 개행 문자('\n')를 오른쪽에 달고 저장
            # with open(vocab_path, 'w', encoding='utf-8') as vocabulary_file:
            #     for word in words:
            #         vocabulary_file.write(word + '\n')
    # else:
    #     # 사전 파일이 존재하면 파일을 불러서 리스트에 추가
    #     # 각 개행 문자를 .strip()으로 제거 후 사전 리스트로 저장
    #     with open(vocab_path, 'r', encoding='utf-8') as vocabulary_file:
    #         lines = vocabulary_file.readlines()
    #         for line in lines:
    #             vocab_list.append(line.strip())
        
    # # 리스트 내용을 key:value 형태의 dictionary 구조로 생성
    # chr2idx, idx2chr = make_vocabulary(vocab_list)
    # # 두 가지 형태의 key, value 형태를 리턴
    # return chr2idx, idx2chr, len(chr2idx)

PATH = 'D:\Programming\Project\chatBOT\ChatBOT\host_volumes\data\ChatbotData.csv'
VOCAB_PATH = 'D:\Programming\Project\chatBOT\ChatBOT\host_volumes\data\\vocabulary.txt'
print(load_vocabulary(PATH, VOCAB_PATH, tokenize_as_morph=False))
