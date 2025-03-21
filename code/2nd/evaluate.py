# ✅ evaluate.py
import torch
from preprocess import *
from gensim.models import Word2Vec
from Transformer import Transformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd

q_list, a_list = load_data("../../data/ChatbotData.csv")
corpus = build_corpus(q_list, a_list)
vocab = build_vocab(corpus)
w2v_model = Word2Vec.load("word2vec_ko.model")
embedding_matrix = build_embedding_matrix(vocab, w2v_model, dim=300)

model = Transformer(4, 300, 6, 512, len(vocab), len(vocab), embedding_matrix=embedding_matrix)
model.load_state_dict(torch.load("transformer_chatbot.pth", map_location="cpu"))
model.eval()

def generate_response(model, sentence, vocab):
    model.eval()
    input_ids = encode_sentences([sentence], vocab)
    input_tensor = torch.tensor(input_ids[0]).unsqueeze(0)
    dec_input = torch.tensor([[vocab["<SOS>"]]])
    for _ in range(MAX_SEQ):
        output = model(input_tensor, dec_input)
        next_token = output.argmax(-1)[:, -1].item()
        if next_token == vocab["<END>"]:
            break
        dec_input = torch.cat([dec_input, torch.tensor([[next_token]])], dim=1)
    idx2word = {v: k for k, v in vocab.items()}
    return " ".join([idx2word[idx.item()] for idx in dec_input[0][1:]])

# 예제 질문
questions = ["안녕", "너 뭐해", "오늘 날씨 어때?"]
for q in questions:
    print(f"Q: {q}\nA: {generate_response(model, q, vocab)}\n")
