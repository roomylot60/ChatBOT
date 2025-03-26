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

import os
import zipfile
import json
from tqdm import tqdm

def process_zip_files(directory, zip_prefix, output_file):
    """
    지정된 디렉토리에서 특정 접두사를 가진 ZIP 파일들을 처리하여 (질문, 답변) 쌍을 추출합니다.

    Parameters:
    - directory (str): ZIP 파일들이 위치한 디렉토리 경로
    - zip_prefix (str): 선택할 ZIP 파일들의 접두사
    - output_file (str): 추출된 (질문, 답변) 쌍을 저장할 파일 경로
    """
    qa_pairs = []

    # 디렉토리 내의 모든 파일에 대해 반복
    for filename in tqdm(os.listdir(directory), desc="Processing ZIP files"):
        if filename.startswith(zip_prefix) and filename.endswith('.zip'):
            zip_path = os.path.join(directory, filename)
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # ZIP 파일의 무결성 검사
                    if zip_ref.testzip() is not None:
                        print(f"Warning: {zip_path} is corrupted.")
                        continue

                    # ZIP 파일 내의 모든 JSON 파일에 대해 반복
                    for json_file in zip_ref.namelist():
                        if json_file.endswith('.json'):
                            with zip_ref.open(json_file) as f:
                                try:
                                    # JSON 파일의 인코딩을 고려하여 읽기
                                    data = json.load(f)
                                    # 데이터 구조에 따라 'utterance' 키를 확인
                                    if 'utterance' in data:
                                        utterances = data['utterance']
                                        for i in range(len(utterances) - 1):
                                            question = utterances[i]['text'].strip()
                                            answer = utterances[i + 1]['text'].strip()
                                            if question and answer:
                                                qa_pairs.append({"question": question, "answer": answer})
                                except json.JSONDecodeError:
                                    print(f"Error decoding JSON from file: {json_file} in {zip_path}")
            except zipfile.BadZipFile:
                print(f"Error: {zip_path} is not a valid ZIP file.")

    # 추출된 (질문, 답변) 쌍을 JSONL 형식으로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in qa_pairs:
            json.dump(pair, f, ensure_ascii=False)
            f.write('\n')

    print(f"총 {len(qa_pairs)}개의 질문-답변 쌍이 저장되었습니다.")

# 사용 예시
directory = 'data/Interview/Training/Labeling'  # ZIP 파일들이 위치한 디렉토리 경로
zip_prefix = 'TL_'         # 선택할 ZIP 파일들의 접두사
output_file = 'data/output.jsonl'          # 결과를 저장할 파일 경로

process_zip_files(directory, zip_prefix, output_file)