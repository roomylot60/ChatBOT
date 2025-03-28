# Module import
import os
import re
import json

import numpy as np
import pandas as pd

# 종종 pip 이나 anaconda를 통해 module을 설치해도
# python의 라이브러리 내부에 설치되지 않아 인식이 되지 않을 수 있음
# Scripts 디렉토리에서 재설치하면 해결
# 일괄적인 해결을 위해 가상환경 설정 후 가상환경 내부에서 설치하는 것이 바람직
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

# Data Load
def load_data(path):
    data_df = pd.read_csv(path, header=0)
    qst, ans = list(data_df['Q']), list(data_df['A'])
    return qst, ans

# 특수문자 제거 및 전체 문장에 대한 단어 사전 리스트로 반환
def data_tokenizer(data):
    words = [] # array to append tokens
    for sentence in data:
        # FILTERS 내의 값들을 정규화 표현식을 통해서 모두 제거
        sentence = re.sub(CHANGE_FILTER, "", sentence)
        for word in sentence.split():
            words.append(word) # 각 문장을 단어로 분리하고 이를 배열에 추가
    return [word for word in words if word] # tokeninzing, 정규화를 거친 값들을 반환

# 형태소 기준 txt_tokenizing 후 각 문장을 형태소들의 리스트로 반환
def prepro_like_morpheme(data):
    morph_analyzer = Okt()
    result_data = []
    for seq in tqdm(data): # 데이터에 대한 진행도 확인
        morphemic_seq = " ".join(morph_analyzer.morphs(seq.replace(' ', '')))
        result_data.append(morphemic_seq)
    return result_data

# 단어:인덱스, 인덱스:단어, 단어 개수 사전 반환 함수
def make_vocabulary(vocab_list):
    chr2idx = {chr:idx for idx, chr in enumerate(vocab_list)}
    idx2chr = {idx:chr for idx, chr in enumerate(vocab_list)}
    return chr2idx, idx2chr

# Generate Word Dictionary
# Return the values (word:index), (index:word), (num_of_total_word)
def load_vocabulary(path, vocab_path, tokenize_as_morph=False):
    # List for dictionary
    vocab_list = []
    # Checking the file presence on the path
    # Generating vocabulary.txt file
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
            data = []
            # .extend(전달값) : 전달값(iterable)을 하나씩 분해해서(리스트 구성을 제거하고) 리스트에 각 원소로 저장(!! .append()와 구분)
            # 리스트 내부에 리스트 단위로 들어있는 문장 별 형태소를 data 리스트에 원소별로 저장
            data.extend(qst)
            data.extend(ans)
            
            words = data_tokenizer(data)
            
            words = list(set(words)) # 중복 단어 제거를 위해 set

            # MARKER를 dictionary에 추가
            # 순서대로 넣기 위해서 인덱스 0 에 추가
            # PAD = "<PADDING>"
            # STD = "<START>"
            # END = "<END>"
            # UNK = "<UNKOWN>"
            words[:0] = MARKER
        
            # 리스트로 된 단어 사전을 vocabulary_file에 저장
            # 각 단어는 개행 문자('\n')를 오른쪽에 달고 저장
            with open(vocab_path, 'w', encoding='utf-8') as vocabulary_file:
                for word in words:
                    vocabulary_file.write(word + '\n')

    # 사전 파일이 존재하면 파일을 불러서 리스트에 추가
    # 각 개행 문자를 .strip()으로 제거 후 사전 리스트로 저장
    with open(vocab_path, 'r', encoding='utf-8') as vocabulary_file:
        lines = vocabulary_file.readlines()
        for line in lines:
            vocab_list.append(line.strip())
    # 리스트 내용을 key:value 형태의 dictionary 구조로 생성
    chr2idx, idx2chr = make_vocabulary(vocab_list)
    # 두 가지 형태의 key, value 형태를 리턴
    return chr2idx, idx2chr, len(chr2idx)

# input data(list about indexed sentences, list for length of sentences)
def enc_processing(value, dictionary, tokenize_as_morph=False):
    # 인덱스 값.(누적된다.)
    sequences_input_index = []
    # 하나의 인코딩 되는 문장 길이.(누적된다.)
    sequences_length = []
    # 형태소 토크나이징 사용 유무
    if tokenize_as_morph:
        value = prepro_like_morpheme(value)

    # 한줄씩 불어온다.
    for sequence in value:
        # 특수문자 제거
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        # 하나의 문장을 인코딩 할때 잠시 사용할 저장공간 : 매 문장마다 새로 만들어 사용할 것임
        sequence_index = []
        # 문장을 띄어쓰기 단위로 분리(단어가 됨)
        for word in sequence.split():
            # 잘려진 단어들이 단어 사전에 존재 하는지 보고
            # 그 값을 가져와 sequence_index에 추가
            if dictionary.get(word) is not None:
                sequence_index.extend([dictionary[word]])
            # 잘려진 단어가 딕셔너리에 존재 하지 않는
            # 경우 UNK(인덱스 : 3)를 넣어 준다.
            else:
                sequence_index.extend([dictionary[UNK]])
        # 문장 제한 길이보다 길어질 경우 뒤에 토큰을 자름
        if len(sequence_index) > MAX_SEQ:
            sequence_index = sequence_index[:MAX_SEQ]
        # 하나의 문장의 길이값 누적
        sequences_length.append(len(sequence_index))
        # max_sequence_length보다 문장 길이가
        # 작다면 빈 부분에 PAD(인덱스 0)를 넣어줌.
        sequence_index += (MAX_SEQ - len(sequence_index)) * [dictionary[PAD]]
        # 인덱스화 되어 있는 값을
        # sequences_input_index에 넣어줌.
        sequences_input_index.append(sequence_index)
    # 인덱스화된 일반 배열을 넘파이 배열로 변경한다.
    # 이유는 텐서플로우 dataset에 넣어 주기 위한
    # 사전 작업.
    # 넘파이 배열에 인덱스화된 배열과 그 길이를 넘김.
    return np.asarray(sequences_input_index), sequences_length

# Function returning decoder input data(indexed sentences list, length of sentences)
def dec_output_processing(value, dictionary, tokenize_as_morph=False):
    # indexed sentences list
    sequences_output_index = []
    # length of each sentence
    sequences_length = []
    # morpheme tokenizing
    if tokenize_as_morph:
        value = prepro_like_morpheme(value)
    
    # load each sentence
    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = []
        # 디코딩 입력의 처음에는 START가 와야 함
        # 문장에서 공백 단위 별로 단어를 가져와서 사전의 값인 인덱스를 입력
        sequence_index = [dictionary[STD]] + [dictionary[word] if word in dictionary else dictionary[UNK] for word in sequence.split()]
        # 문장 제한 길이보다 길어질 경우, 뒤의 token을 제거
        if len(sequence_index) > MAX_SEQ:
            sequence_index = sequence_index[:MAX_SEQ]
        # 하나의 문장에 길이를 입력
        sequences_length.append(len(sequence_index))
        # max_sequence_length 보다 문장길이가 짧을 경우,
        # PAD(index = 0)을 입력
        sequence_index += (MAX_SEQ - len(sequence_index)) * [dictionary[PAD]]
        # 인덱스 화 되어있는 값을 sequences_output_index에 추가
        sequences_output_index.append(sequence_index)
    # 인덱스 화 된 일반 배열을 numpy 배열로 변경
    return np.asarray(sequences_output_index), sequences_length

# Function returning decoder label data(indexed sentences list)
def dec_target_processing(value, dictionary, tokenize_as_morph=False):
    # indexed sentences list
    sequences_target_index = []
    # morpheme tokenizing
    if tokenize_as_morph:
        value = prepro_like_morpheme(value)
    # load each sentence
    for sequence in value:
        # remove special letter
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        # sentence indexing and add END token
        sequence_index = [dictionary[word] if word in dictionary else dictionary[UNK] for word in sequence.split()]
        
        # 문장 제한 길이보다 길 경우,
        # 토큰을 자르고 END 토큰을 넣음
        if len(sequence_index) >= MAX_SEQ:
            sequence_index = sequence_index[:MAX_SEQ - 1] + [dictionary[END]]
        else:
            sequence_index += [dictionary[END]]
        
        # max_sequence_length보다 문장 길이가
        # 작다면 빈 부분에 PAD(0)를 넣어준다.
        sequence_index += (MAX_SEQ - len(sequence_index)) * [dictionary[PAD]]
        
        # Add indexed values to sequences_target_index
        sequences_target_index.append(sequence_index)
    return np.asarray(sequences_target_index)
