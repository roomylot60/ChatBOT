import torch
import torch.nn as nn
import numpy as np
from preprocess import load_vocab, encode_sentences, label_sentences, tokenize_morphs
from Transformer import Transformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 로드
vocab = load_vocab("data/vocab.json")
embedding_matrix = np.load("models/embedding_matrix.npy")
model = Transformer(4, 300, 6, 512, len(vocab), len(vocab)).to(DEVICE)
model.load_state_dict(torch.load("models/pretrained_transformer_model.pth", map_location=DEVICE))
model.eval()
idx2word = {v: k for k, v in vocab.items()}

# 응답 생성
def generate_response(model, sentence, vocab, max_len=25):
    input_ids = encode_sentences([sentence], vocab)
    input_tensor = torch.tensor(input_ids).to(DEVICE)
    dec_input = torch.tensor([[vocab["<SOS>"]]], dtype=torch.long).to(DEVICE)
    output_tokens = []
    for _ in range(max_len):
        with torch.no_grad():
            output = model(input_tensor, dec_input)
            next_token = output.argmax(-1)[:, -1].item()
        if next_token == vocab["<END>"]:
            break
        output_tokens.append(next_token)
        dec_input = torch.cat([dec_input, torch.tensor([[next_token]], device=DEVICE)], dim=1)
    return " ".join([idx2word.get(tok, "<UNK>") for tok in output_tokens])

# 🔸 평가 지표
def calculate_bleu(pairs):
    total = 0
    for ref, q in pairs:
        hyp = generate_response(model, q, vocab).split()
        total += sentence_bleu([ref.split()], hyp, smoothing_function=SmoothingFunction().method1)
    return total / len(pairs)

def calculate_rouge(pairs):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)
    r1, rl = 0, 0
    for q, ref in pairs:
        hyp = generate_response(model, q, vocab)
        scores = scorer.score(ref, hyp)
        r1 += scores["rouge1"].fmeasure
        rl += scores["rougeL"].fmeasure
    return r1 / len(pairs), rl / len(pairs)

def calculate_perplexity(pairs):
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])
    loss = 0
    for q, a in pairs:
        enc = torch.tensor(encode_sentences([q], vocab)).to(DEVICE)
        dec_in = torch.tensor(label_sentences([a], vocab)).to(DEVICE)
        dec_out = torch.tensor(label_sentences([a], vocab)).to(DEVICE)
        with torch.no_grad():
            output = model(enc, dec_in[:, :-1])
            loss += criterion(output.view(-1, output.size(-1)), dec_out[:, 1:].reshape(-1)).item()
    return np.exp(loss / len(pairs))

def calculate_accuracy(pairs):
    total, correct = 0, 0
    for q, a in pairs:
        pred = generate_response(model, q, vocab).split()
        label = a.split()
        match = sum(p == l for p, l in zip(pred, label))
        correct += match
        total += len(label)
    return correct / total if total else 0

# 평가 샘플
eval_pairs = [("좋아요", "이 영화 어때?"), ("별로예요", "지루한 영화야?")]

print("✅ BLEU:", round(calculate_bleu(eval_pairs), 4))
r1, rl = calculate_rouge(eval_pairs)
print("✅ ROUGE-1:", round(r1, 4))
print("✅ ROUGE-L:", round(rl, 4))
print("✅ Perplexity:", round(calculate_perplexity(eval_pairs), 4))
print("✅ Accuracy:", round(calculate_accuracy(eval_pairs), 4))

# 예시 질의응답
for q in ["안녕", "요즘 어때", "심심한데 뭐해"]:
    print(f"Q: {q}")
    print(f"A: {generate_response(model, q, vocab)}\n")
