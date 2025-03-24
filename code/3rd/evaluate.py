import torch
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from gensim.models import Word2Vec
from preprocess import (
    load_korpora_corpus, build_corpus, load_vocab,
    build_embedding_matrix, encode_sentences, label_sentences
)
from Transformer import Transformer
import numpy as np

# ⚙️ 설정
MAX_SEQ = 25
EMBED_DIM = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")

# 📌 데이터 및 모델 로드
q_list, a_list = load_korpora_corpus()
corpus = build_corpus(q_list, a_list)
vocab = load_vocab(vocab, "../../models/vocab.json")

w2v_model = Word2Vec.load("../../models/word2vec_ko.model")
embedding_matrix = build_embedding_matrix(vocab, w2v_model, dim=EMBED_DIM)

model = Transformer(
    num_layers=4,
    d_model=EMBED_DIM,
    num_heads=6,
    dff=512,
    input_vocab_size=len(vocab),
    target_vocab_size=len(vocab),
    embedding_matrix=embedding_matrix
).to(DEVICE)
model.load_state_dict(torch.load("../../models/pretrained_transformer_model.pth", map_location=DEVICE))
model.eval()

idx2word = {v: k for k, v in vocab.items()}


# 🔹 응답 생성
def generate_response(model, sentence, vocab, max_seq=MAX_SEQ):
    input_ids = encode_sentences([sentence], vocab, max_seq_length=max_seq)
    input_tensor = torch.tensor(input_ids[0], dtype=torch.long).unsqueeze(0).to(DEVICE)
    dec_input = torch.tensor([[vocab["<SOS>"]]], dtype=torch.long).to(DEVICE)

    output_tokens = []
    for _ in range(max_seq):
        with torch.no_grad():
            output = model(input_tensor, dec_input)
            next_token = output.argmax(-1)[:, -1].item()
        if next_token == vocab["<END>"]:
            break
        output_tokens.append(next_token)
        dec_input = torch.cat([dec_input, torch.tensor([[next_token]], dtype=torch.long).to(DEVICE)], dim=1)

    return " ".join([idx2word.get(tok, "<UNK>") for tok in output_tokens])


# 🔹 BLEU
def calculate_bleu_score():
    total_bleu = 0
    for q, a in zip(q_list[:100], a_list[:100]):
        ref = encode_sentences([a], vocab)[0]
        gen = encode_sentences([generate_response(model, q, vocab)], vocab)[0]
        total_bleu += sentence_bleu([ref], gen, smoothing_function=SmoothingFunction().method1)
    return total_bleu / 100


# 🔹 ROUGE
def calculate_rouge_score():
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)
    total_r1, total_rl = 0, 0
    for q, a in zip(q_list[:100], a_list[:100]):
        pred = generate_response(model, q, vocab)
        scores = scorer.score(a, pred)
        total_r1 += scores['rouge1'].fmeasure
        total_rl += scores['rougeL'].fmeasure
    return total_r1 / 100, total_rl / 100


# 🔹 Perplexity
def calculate_perplexity():
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])
    total_loss = 0
    for q, a in zip(q_list[:100], a_list[:100]):
        enc_input = torch.tensor(encode_sentences([q], vocab), dtype=torch.long).to(DEVICE)
        dec_input = torch.tensor(label_sentences([a], vocab), dtype=torch.long).to(DEVICE)
        dec_target = torch.tensor(label_sentences([a], vocab), dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            output = model(enc_input, dec_input[:, :-1])  # (B, T, vocab)
            loss = criterion(output.reshape(-1, output.size(-1)), dec_target[:, 1:].reshape(-1))
        total_loss += loss.item()
    avg_loss = total_loss / 100
    return np.exp(avg_loss)  # Perplexity = exp(loss)


# 🔹 Accuracy (토큰 단위)
def calculate_accuracy():
    total_tokens = 0
    correct_tokens = 0
    for q, a in zip(q_list[:100], a_list[:100]):
        pred = generate_response(model, q, vocab).split()
        true = a.split()
        min_len = min(len(pred), len(true))
        correct_tokens += sum([p == t for p, t in zip(pred[:min_len], true[:min_len])])
        total_tokens += min_len
    return correct_tokens / total_tokens if total_tokens > 0 else 0.0


# ✅ 예시 출력
sample_questions = ["안녕", "요즘 어때?", "영화 추천해줘"]
for q in sample_questions:
    print(f"Q: {q}\nA: {generate_response(model, q, vocab)}\n")

# ✅ 평가 수행
print("[INFO] 모델 평가 중...")
bleu = calculate_bleu_score()
rouge1, rougeL = calculate_rouge_score()
perplexity = calculate_perplexity()
acc = calculate_accuracy()

print(f"✅ BLEU Score : {bleu:.4f}")
print(f"✅ ROUGE-1 F1 : {rouge1:.4f}")
print(f"✅ ROUGE-L F1 : {rougeL:.4f}")
print(f"✅ Perplexity : {perplexity:.4f}")
print(f"✅ Accuracy   : {acc:.4f}")
