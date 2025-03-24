import numpy as np
from gensim.models import Word2Vec

# 기존 저장된 vocab 불러오기 (ex: vocab.json 등으로 저장했다면 json으로 로드)
import json
with open("../../models/vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

# Word2Vec 모델 로딩
w2v_model = Word2Vec.load("../../models/word2vec_ko.model")

# 임베딩 차원
dim = 300
embedding_matrix = np.random.normal(scale=0.6, size=(len(vocab), dim))

# vocab 기준으로 embedding_matrix 생성
for word, idx in vocab.items():
    if word in w2v_model.wv:
        embedding_matrix[idx] = w2v_model.wv[word]

# 저장
np.save("../../models/embedding_matrix.npy", embedding_matrix)
print("✅ embedding_matrix.npy 저장 완료")