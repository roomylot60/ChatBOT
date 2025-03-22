# # evaluate.py
# import math
# import torch
# import torch.nn as nn
# import pandas as pd
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from rouge_score import rouge_scorer
# from gensim.models import Word2Vec
# from preprocess import load_data, build_corpus, build_vocab, encode_sentences, build_embedding_matrix
# from Transformer import Transformer

# # ⚙️ 설정
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"✅ Device: {DEVICE}")

# # 📌 데이터 로딩
# q_list, a_list = load_data("../../data/ChatbotData.csv")
# corpus = build_corpus(q_list, a_list)
# vocab = build_vocab(corpus)
# w2v_model = Word2Vec.load("../../models/word2vec_ko.model")
# embedding_matrix = build_embedding_matrix(vocab, w2v_model, dim=300)

# # 🔺 모델 로드
# model = Transformer(
#     num_layers=4, d_model=300, num_heads=6, dff=512,
#     input_vocab_size=len(vocab), target_vocab_size=len(vocab),
#     embedding_matrix=embedding_matrix
# ).to(DEVICE)
# model.load_state_dict(torch.load("../../models/pretrained_transformer_model.pth", map_location=DEVICE))
# model.eval()

# loss_fn = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])
# MAX_SEQ = 25

# # 🔹 응답 생성 (Greedy)
# def generate_response(model, sentence, vocab):
#     model.eval()
#     input_ids = encode_sentences([sentence], vocab, max_seq_length=MAX_SEQ)
#     input_tensor = torch.tensor(input_ids[0]).unsqueeze(0).to(DEVICE)
    
#     dec_input = torch.tensor([[vocab["<SOS>"]]], dtype=torch.long).to(DEVICE)
#     for _ in range(MAX_SEQ):
#         output = model(input_tensor, dec_input)
#         next_token = output.argmax(-1)[:, -1].item()
#         if next_token == vocab["<END>"]:
#             break
#         dec_input = torch.cat([dec_input, torch.tensor([[next_token]]).to(DEVICE)], dim=1)

#     # idx2word 변환
#     idx2word = {v: k for k, v in vocab.items()}
#     response = [idx2word.get(idx.item(), "<UNK>") for idx in dec_input[0][1:]]  # <SOS> 제외
#     return " ".join(response)

# # 🔹 BLEU 계산
# def calculate_bleu_score():
#     total_bleu = 0
#     total = 0
#     for q, a in zip(q_list, a_list):
#         reference = encode_sentences([a], vocab, max_seq_length=MAX_SEQ)[0]
#         candidate = encode_sentences([generate_response(model, q, vocab)], vocab, max_seq_length=MAX_SEQ)[0]
#         bleu = sentence_bleu([reference], candidate, smoothing_function=SmoothingFunction().method1)
#         total_bleu += bleu
#         total += 1
#     return total_bleu / total

# # 🔹 ROUGE 계산
# def calculate_rouge_score():
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
#     total_rouge1 = 0
#     total_rougeL = 0
#     total = 0
#     for q, a in zip(q_list, a_list):
#         prediction = generate_response(model, q, vocab)
#         scores = scorer.score(a, prediction)
#         total_rouge1 += scores['rouge1'].fmeasure
#         total_rougeL += scores['rougeL'].fmeasure
#         total += 1
#     return total_rouge1 / total, total_rougeL / total

# # 🔍 실행
# print("\n🔍 예제 질문에 대한 응답")
# samples = ["안녕", "오늘 뭐해?", "너의 이름은 뭐야?", "기분 어때?"]
# for q in samples:
#     print(f"Q: {q}\nA: {generate_response(model, q, vocab)}\n")

# print("🔍 성능 평가 시작...")
# bleu = calculate_bleu_score()
# rouge1, rougeL = calculate_rouge_score()
# print(f"✅ BLEU Score: {bleu:.4f}")
# print(f"✅ ROUGE-1 F1: {rouge1:.4f}")
# print(f"✅ ROUGE-L F1: {rougeL:.4f}")

import torch
from preprocess import load_data, build_corpus, build_vocab, build_embedding_matrix, encode_sentences
from gensim.models import Word2Vec
from Transformer import Transformer
import pandas as pd

# 설정
MAX_SEQ = 25
EMBED_DIM = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 및 사전 로드
q_list, a_list = load_data("../../data/ChatbotData.csv")
corpus = build_corpus(q_list, a_list)
vocab = build_vocab(corpus)

# ✅ 디버깅 출력
print(f"[INFO] 총 vocab 크기: {len(vocab)}")

# Word2Vec 로드
w2v_path = "../../models/word2vec_ko.model"
try:
    w2v_model = Word2Vec.load(w2v_path)
    print(f"[INFO] Word2Vec 모델 로드 성공: {w2v_path}")
except Exception as e:
    print(f"[ERROR] Word2Vec 모델 로드 실패: {e}")
    exit()

# 임베딩 매트릭스 생성
embedding_matrix = build_embedding_matrix(vocab, w2v_model, dim=EMBED_DIM)

# Transformer 모델 생성
model = Transformer(
    num_layers=4,
    d_model=EMBED_DIM,
    num_heads=6,
    dff=512,
    input_vocab_size=len(vocab),
    target_vocab_size=len(vocab),
    embedding_matrix=embedding_matrix
).to(DEVICE)

# 모델 가중치 로드
model_path = "../../models/pretrained_transformer_model.pth"
try:
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"[INFO] Transformer 모델 로드 성공: {model_path}")
except Exception as e:
    print(f"[ERROR] Transformer 모델 로드 실패: {e}")
    exit()

# idx2word 매핑
idx2word = {v: k for k, v in vocab.items()}

# 챗봇 응답 생성 함수 (디버깅용)
def generate_response(model, sentence, vocab, max_seq=MAX_SEQ):
    input_ids = encode_sentences([sentence], vocab, max_seq_length=max_seq)
    input_tensor = torch.tensor(input_ids[0], dtype=torch.long).unsqueeze(0).to(DEVICE)

    dec_input = torch.tensor([[vocab["<SOS>"]]], dtype=torch.long).to(DEVICE)
    output_tokens = []

    print(f"[DEBUG] 질문: {sentence}")
    print(f"[DEBUG] 입력 토큰: {input_ids[0]}")

    output_tokens = []

    for i in range(max_seq):
        with torch.no_grad():
            output = model(input_tensor, dec_input)  # (1, t, vocab_size)
            probs = torch.softmax(output[:, -1, :], dim=-1)
            topk = torch.topk(probs, k=5)

            print(f"[STEP {i}] Top-5 다음 토큰: {[(idx.item(), idx2word.get(idx.item(), '<UNK>'), prob.item()) for idx, prob in zip(topk.indices[0], topk.values[0])]}")

            next_token = torch.argmax(probs[0, :]).item()

        # ✅ 첫 토큰이 <END>이면 무시하고 진행
        if next_token == vocab["<END>"]:
            if i == 0:
                print(f"[STEP {i}] <END> 예측되었으나 첫 step — 무시하고 계속")
            else:
                print(f"[STEP {i}] <END> 예측됨 — 종료")
                break

        output_tokens.append(next_token)

        # 디코더 입력에 다음 토큰 추가
        next_token_tensor = torch.tensor([[next_token]], dtype=torch.long).to(DEVICE)
        dec_input = torch.cat([dec_input, next_token_tensor], dim=1)



    decoded_words = [idx2word.get(token, "<UNK>") for token in output_tokens]

    print(f"[DEBUG] 출력 토큰: {output_tokens}")
    print(f"[DEBUG] 응답 단어: {decoded_words}")

    return " ".join(decoded_words)

# 테스트
sample_questions = ["안녕", "너 뭐해", "오늘 날씨 어때?"]

for q in sample_questions:
    print(f"Q: {q}")
    answer = generate_response(model, q, vocab)
    print(f"A: {answer}\n")

for name, param in model.named_parameters():
    if "embedding" in name:
        print(f"[CHECK] {name} 평균값: {param.data.mean().item():.4f}, std: {param.data.std().item():.4f}")

print(f"[CHECK] embedding.weight 평균값: {model.embedding.weight.data.mean().item():.4f}, std: {model.embedding.weight.data.std().item():.4f}")