# import os
# import json
# import torch
# import torch.nn as nn
# from tqdm import tqdm
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from rouge_score import rouge_scorer
# from KoBERT import KoBERT_Transformer
# from preprocess import CORPUS_PATH, TOKENIZER_PATH
# from tokenization_kobert import KoBertTokenizer
# # 설정
# MAX_LEN = 64
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"[INFO] Device: {DEVICE}")

# tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")

# # 🔹 모델 로드
# model = KoBERT_Transformer(decoder_vocab_size=tokenizer.vocab_size).to(DEVICE)
# model.load_state_dict(torch.load("models/kobert_chatbot.pt", map_location=DEVICE))
# model.eval()

# def generate_response(input_text):
#     model.eval()
#     # 🔹 입력 문장 및 토큰 확인
#     print("[INPUT]", input_text)
#     token_ids = tokenizer.encode(input_text)
#     print("[ENCODED TOKEN IDS]", token_ids)
#     print("[ENCODED TOKENS]", tokenizer.convert_ids_to_tokens(token_ids))

#     enc = tokenizer.encode_plus(
#         input_text,
#         return_tensors="pt",
#         truncation=True,
#         max_length=MAX_LEN,
#         padding="max_length"
#     )

#     input_tensor = enc["input_ids"].to(DEVICE)
#     enc_mask = enc["attention_mask"].to(DEVICE)

#     START_TOKEN = tokenizer.cls_token_id
#     SEP_TOKEN = tokenizer.sep_token_id

#     dec_input = torch.tensor([[START_TOKEN]], dtype=torch.long).to(DEVICE)
#     result = []

#     for _ in range(MAX_LEN):
#         with torch.no_grad():
#             output = model(input_tensor, enc_mask, dec_input)
#         next_token = output.argmax(-1)[:, -1].item()
#         print(f"[DEBUG] next_token: {next_token} → {tokenizer.convert_ids_to_tokens([next_token])}")
#         if next_token == SEP_TOKEN and len(result) > 1:
#             break
#         result.append(next_token)
#         dec_input = torch.cat([dec_input, torch.tensor([[next_token]]).to(DEVICE)], dim=1)

#     return tokenizer.decode(result, skip_special_tokens=True)

# print("\n[디코딩 테스트]")
# print("[MODEL PARAMS]", sum(p.numel() for p in model.parameters()))
# tokens = tokenizer.tokenize("기분이 안 좋아")
# print(tokens)
# ids = tokenizer.convert_tokens_to_ids(tokens)
# print(ids)
# response = generate_response("기분이 안 좋아")  # 감정이 뚜렷한 문장
# print(f"응답: '{response}'")  # ← 여기 공란이면 바로 위 디버깅 포인트로 점검

from tokenization_kobert import KoBertTokenizer

tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")
sample_text = "안녕하세요, 반갑습니다."
tokens = tokenizer.tokenize(sample_text)
print(tokens)