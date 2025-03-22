# Chatbot_data.          
Chatbot_data_for_Korean v1.0             


## Data description.    

인공데이터입니다. 일부 이별과 관련된 질문에서 다음카페 "사랑보다 아름다운 실연( http://cafe116.daum.net/_c21_/home?grpid=1bld )"에서 자주 나오는 이야기들을 참고하여 제작하였습니다. 
가령 "이별한 지 열흘(또는 100일) 되었어요"라는 질문에 챗봇이 위로한다는 취지로 답변을 작성하였습니다. 


1. 챗봇 트레이닝용 문답 페어 11,876개           
2. 일상다반사 0, 이별(부정) 1, 사랑(긍정) 2로 레이블링                
                      
                     
## Quick peek.                
                                     
![quick_peek](./data/img/data.png)


## 관련 코드 : [Korean Language Model for Wellness Conversation](https://github.com/nawnoes/WellnessConversationAI?fbclid=IwAR3ZhXYW_DwI2RXP1mbHzvafGXF80QWERa4t6TTz_m2NQug5QwjOwQt6Hvw)
- 이 곳에 저장된 데이터를 만들면서 누군가에게 위로가 되는 모델이 나오면 좋겠다고 생각했었는데 제 생각보다 더 잘 만든 모델이 있어서 링크 걸어 둡니다. 부족한 데이터지만 이곳에 저장된 데이터와 [AI 허브 정신건강 상담 데이터](http://www.aihub.or.kr/keti_data_board/language_intelligence)  를 토대로 만들었다고 합니다. 
- 전창욱 외(2020), 텐서플로2와 머신러닝으로 시작하는 자연어처리, 위키북스( http://cafe116.daum.net/_c21_/home?grpid=1bld )의 챗봇 부분에도 이 데이터가 사용된 것으로 알고 있습니다. 빠르게 챗봇 만들고 싶으신 분들은 참고하셔도 좋을 것 같습니다.
- 데이터 로더를 통한 다운로드는 다음 링크 [Korpora: Korean Corpora Archives](https://github.com/ko-nlp/Korpora)를 참고하시면 편하게 사용하실 수 있을 듯합니다.



#인용

Youngsook Song.(2018). Chatbot_data_for_Korean v1.0)[Online]. Available : https://github.com/songys/Chatbot_data (downloaded 2022. June. 29.)
---
## 1. Sequence-to-sequence Model(seq2seq)
- 입력된 시퀀스로부터 다른 도메인의 시퀀스를 출력하는 다양한 분야에서 사용
- 챗봇과 기계번역, 내용 요약, STT(Speach to Text) 등에서 활용
- 내부가 보이지 않는 블랙 박스에서 RNN의 조립을 통해 구조를 생성
- **인코더**와 **디코더**라는 두 개의 모듈로 구성되며 각각의 RNN 아키텍쳐를 보유
- 각각의 모듈은 RNN셀로 구성된 아키텍쳐로 성능을 위해 바닐라 RNN이 아니라 LSTM 셀 혹은 GRU 셀들로 구성

### Encoder
- 입력 문장의 모든 단어들을 순차적으로 입력받은 뒤에 마지막에 이 모든 단어 정보들을 압축해서 **하나의 Context vector**를 생성
- 인코더 RNN 셀의 마지막 시점의 은닉상태를 context vector라 함

### Decoder
- 인코더로부터 전송받은 Context vector를 변환된 단어로 하나씩 순차적으로 출력
- 디코더는 기본적으로 RNNLM(RNN Language Model)
- 테스트 과정과 훈련 과정의 작동방식에서 차이가 있음

1. `테스트 과정`
- Context vector와 초기 입력으로 문장의 시작을 의미하는 심볼 `<sos>`을 입력으로 받음
- 다음에 등장 확률이 높은 단어 예측하고 예측된 단어를 다음 입력으로 활용
- 문장의 끝을 의미하는 `<eos>`가 예측될 때까지 위의 과정을 반복

2. `훈련 과정`
- 디코어에게 인코더가 보낸 Context vector와 실제 정답을 입력받고, 나와야 하는 정답을 알려주면서 훈련(실제 정답은 `<sos>`로 시작하는 문장, 나와야 하는 정답은 `<eos>`로 끝나는 문장)
- 해당 과정을 교사 강요(teaching forcing)이라고 함
![인코더에서 디코더까지의 문장 변환 흐름](./data/img/encoder_to_decoder.png)

Ref.[Blog wikidocs.net - seq2seq](https://wikidocs.net/24996)

---
## Transformer

#### BLEU Score가 낮으면 성능이 좋지 않다는 의미인가?
- BLEU Score: 0.0613 (≈ 6.13%) → 매우 낮은 점수
- 평균 Loss: 0.0013 / Perplexity: 1.0013 → 매우 낮음 (이상적으로 낮아야 하지만, 너무 낮으면 과적합 가능성 있음)

#### BLEU Score: 0.0613의 의미
- 일반적으로 챗봇 모델에서 BLEU Score가 0.2~0.4(20%~40%) 정도면 양호한 성능
- 0.0613(6.13%)은 모델이 정답과 거의 유사하지 않은 문장을 생성하고 있다는 의미
- BLEU Score가 낮다면 모델이 적절한 응답을 생성하지 못하고 있을 가능성이 높음
- 즉, 현재 모델의 응답 품질이 좋지 않다는 것을 의미할 가능성이 큼

#### Loss와 Perplexity는 좋은데 BLEU Score가 낮은 이유?
1. BLEU Score는 단순한 단어 매칭 기반 지표
    - BLEU는 단순히 생성된 응답과 정답 간의 n-gram(단어 조합) 일치를 측정
    - 의미적으로는 적절한 응답이라도 단어가 다르면 BLEU Score가 낮게 나올 수 있음
2. Perplexity(PPL)가 너무 낮으면 과적합 가능성
    - PPL이 1.0013이면 모델이 거의 완벽하게 데이터를 학습했음을 의미
    - 하지만 과적합이 발생하면 새로운 입력에서 적절한 답변을 생성하지 못함
3. 데이터셋이 충분하지 않거나 일반적인 대화를 반영하지 못함
    - 학습 데이터가 충분하지 않거나, 너무 한정적인 패턴을 가지고 있으면 성능 저하
    - 모델이 너무 특정한 패턴에 맞춰 학습되었을 가능성 있음
4. 디코딩 방식(Beam Search, Temperature, Top-k Sampling) 개선 필요
    - 현재 Greedy Decoding(argmax 사용)으로 문장을 생성 중
    - Beam Search, Top-k Sampling 등 개선된 방법 적용 가능

#### 해결 방법
1. BLEU Score만으로 성능을 평가하지 않고, 직접 대화 테스트 수행

```python
sample_inputs = ["안녕하세요?", "오늘 날씨 어때?", "너의 이름은?", "무슨 일을 할 수 있어?"]
for sample in sample_inputs:
    print(f"💬 질문: {sample}")
    print(f"🤖 챗봇: {chatbot_response(transformer, sample, vocab, device)}\n")
```

- BLEU Score가 낮아도 의미적으로 적절한 응답인지 직접 확인 필요

2. Greedy Decoding 대신 Beam Search 적용 (더 다양한 문장 생성)
- 현재는 Greedy Decoding (argmax) 방식으로 단순히 가장 높은 확률의 단어를 선택하는데, 이것은 단순하고 정확도가 떨어질 수 있음.
- Beam Search 또는 Top-k Sampling을 사용하면 더 다양한 답변 생성 가능
- Beam Search 적용

```python
def beam_search_decoding(model, input_tensor, vocab, device, beam_size=3, max_length=50):
    model.eval()
    dec_input = torch.tensor([[vocab["<SOS>"]]], dtype=torch.long).to(device)
    sequences = [(dec_input, 0)]

    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            with torch.no_grad():
                output = model(input_tensor, seq, training=False)
                topk_probs, topk_indices = torch.topk(output[:, -1, :], beam_size)

            for i in range(beam_size):
                next_token = topk_indices[0, i].item()
                new_seq = torch.cat([seq, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)
                new_score = score + torch.log(topk_probs[0, i].item())  # 로그 확률 합산
                all_candidates.append((new_seq, new_score))

        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]  # 상위 beam_size 개 선택

    response_tokens = sequences[0][0].squeeze(0).tolist()[1:]  # `<SOS>` 제거
    response = [word for word, idx in vocab.items() if idx in response_tokens]
    return " ".join(response)
```

- Beam Search를 적용하면 더 다양한 문장을 탐색하여 최적의 응답 생성 가능
- BLEU Score 상승 가능성 있음

3. 모델 재훈련 - 과적합 방지 (데이터 확장, Dropout 조정)
- Dropout을 증가시켜 과적합 방지

```python
class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout=0.3):  # 🔥 dropout 증가
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.encoder = TransformerEncoderLayer(d_model, num_heads, dff, dropout)
        self.decoder = TransformerDecoderLayer(d_model, num_heads, dff, dropout)
        self.final_layer = nn.Linear(d_model, target_vocab_size)

    def forward(self, enc_input, dec_input, enc_mask=None, dec_mask=None, training=True):
        enc_input, dec_input = self.embedding(enc_input), self.embedding(dec_input)
        enc_output = self.encoder(enc_input, enc_mask)
        dec_output = self.decoder(dec_input, enc_output, dec_mask, enc_mask, training)
        return self.final_layer(dec_output)
```

- Dropout을 0.1 → 0.3으로 증가하여 과적합 방지
- PPL이 너무 낮은 문제 해결 가능성 있음

4. 데이터 증강 - Synonym Replacement 적용
- 훈련 데이터가 너무 한정적이라면 데이터 증강(Synonym Replacement, Paraphrasing 등)을 적용하여 학습 데이터 다양화
- 예제 코드 (NLTK WordNet 사용):

```python
from nltk.corpus import wordnet

def synonym_replacement(sentence):
    words = sentence.split()
    new_sentence = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            new_sentence.append(synonyms[0].lemmas()[0].name())  # 첫 번째 동의어 사용
        else:
            new_sentence.append(word)
    return " ".join(new_sentence)

# 예제 적용
original_sentence = "오늘 날씨 어때?"
augmented_sentence = synonym_replacement(original_sentence)
print(f"Original: {original_sentence} → Augmented: {augmented_sentence}")
```

- 데이터 증강을 통해 모델이 다양한 표현을 학습하도록 유도
- BLEU Score 개선 가능성 있음

#### 결론
- BLEU Score 0.0613은 매우 낮은 값으로, 모델의 응답 성능이 좋지 않을 가능성이 높음
- Loss와 Perplexity가 너무 낮은 것은 과적합 가능성이 있으며, Dropout을 증가시켜 방지 필요
- Beam Search를 적용하여 더 다양한 응답을 생성할 수 있도록 개선 가능
- 데이터 증강(Synonym Replacement, Paraphrasing)으로 모델이 더 다양한 패턴을 학습하도록 유도
- 다음 단계는 Beam Search, Dropout 조정, 데이터 증강을 적용하여 BLEU Score를 개선

## Pre-trained Embedding을 반영하여 Word2Vec Model을 사용한 Transformer Model
- BLEU Score: 0.7928
- ROUGE-1 F1: 0.0000
- ROUGE-L F1: 0.0000

### 모델 성능 평가 결과 (정정)
|지표|값|해석|
|--------|------|---------------|
|BLEU Score|0.7928|매우 높음, 일반적으로 우수한 번역/생성 결과로 간주됨|
|ROUGE-1 F1|0.0000|n-gram(단어 단위)의 recall 기반, 의미 단위 겹침 없음|
|ROUGE-L F1|0.0000|LCS 기반 문장 구조 유사도, 일치 없음|

- 해석 상 충돌
    - BLEU는 정답과 생성된 문장의 n-gram precision을 보며, ROUGE는 정답과의 recall, 유사 구조를 보는데:
    - BLEU는 매우 높음: 모델이 reference 문장의 단어들을 상당히 잘 맞췄음을 의미
    - ROUGE는 완전 0점: 생성 문장이 reference와 한 글자도 일치하지 않았다는 것

#### 해석 : 실제 BLEU가 잘못 계산되었을 가능성 높음
1. 평가 대상의 일관성 문제 : BLEU는 모델 출력 vs 정답 비교를 제대로 한 반면, ROUGE는 reference 없음 또는 잘못된 reference 상태로 평가되었을 가능성
    - Ex: " " 빈 리스트, <PAD>만 포함된 텐서, 혹은 tokenizer mismatch
2. BLEU 계산 방식 오류 가능성
    - BLEU는 한 문장 기준 sentence_bleu()로 측정 시, smoothing function 없이 사용하면 1.0 혹은 0.0으로 극단적 결과가 나오는 경우 많음
    - smoothing을 쓰더라도, target 문장과 candidate가 거의 유사할 때만 0.7~0.9대가 나옴. 그런데 실제 inference 출력은 “이 이 이 사람…” 이런 반복이므로 실제 reference와 유사하지 않음

#### 확인해야 할 사항
- BLEU 계산 함수 내부 확인

```python
sentence_bleu([ref], hyp) vs corpus_bleu()
```

- smoothing 없이 사용했는지
- <PAD> 또는 <END> 처리 포함 여부
- ROUGE 입력 포맷 확인 : rouge_scorer에 전달한 reference/candidate 리스트가 정확한지
- empty string 또는 tokenized mismatch 가능성
- 실제 출력 문장 확인
- 출력된 응답을 human readable하게 복원해 직접 비교
    - Ex: "이 이 이 이 사람 을 사람…" vs "안녕하세요."

#### 평가 지표 재정비
|항목|권장 조치|
|----|--------------------|
|BLEU|corpus_bleu() 사용 + smoothing_function=method4|
|ROUGE|ROUGE-1, ROUGE-2, ROUGE-L 모두 적용, tokenizer 확인|
|정제|<PAD>, <EOS> 제거 후 평가|
|추가 지표|METEOR 또는 BERTScore 도입 (정확도 + 의미 유사도)|

#### 결론
- BLEU 0.79는 실제 성능보다 높게 평가된 가능성 높음
- ROUGE 0.0은 평가 대상 오류 또는 정답 불일치를 나타냄
- 평가 함수 내부를 재검토하고, 출력 문자열을 직접 확인하여 신뢰도 높은 평가 기준을 구축하는 것이 우선입니다

---

### 데이터/모델 학습 설정 기반 분석
1. 데이터 양: 11,823쌍 (적은 편)
    - 자연어 처리에서는 일반적으로 수십만~수백만 쌍이 필요합니다.
    - 1만 쌍 정도면 간단한 도메인(예: 특정 챗봇 태스크)에 한정적으로는 사용 가능하지만, 다양성 부족, 표현력 한계, 학습 안정성 저하 등이 발생하기 쉬움
    - 결론: 데이터 양은 적은 편이며, 특히 자유로운 대화를 다루기엔 부족합니다.

2. Epoch 수: 10 (짧은 편일 가능성 있음)
    - 일반적으로 작은 데이터셋에서는 **오히려 더 많은 epoch (30~100)**가 필요합니다.
    - 하지만 epoch을 늘린다고 해서 무조건 성능이 오르지는 않습니다:
    - 모델이 단순한 패턴만 반복 학습해 과적합할 수도 있음
    - loss가 plateau 상태가 되면 더 이상 학습 효과가 없음
    - 결론: 10 epoch은 부족할 가능성이 높으며, 학습 로그(예: val loss 추이)를 함께 확인해야 합니다.

3. Word2Vec 임베딩
    - Word2Vec은 각 단어 간 의미 관계를 잘 반영하지만, 학습 초기에 모델이 임베딩에 과도하게 의존하면서 디코딩 성능이 고착될 수 있습니다.
    - 특히 <END> 토큰을 바로 예측하거나, 특정 단어(예: "이", "사람")를 반복 출력하는 것과 같은 현상은 Word2Vec embedding이 잘못 사용됐을 가능성을 시사합니다
    - 결론: Word2Vec embedding 자체는 좋은 선택이지만, fine-tuning되지 않으면 오히려 모델이 고정된 표현에 갇힐 수 있습니다.

4. 디코딩 방식: Greedy (or argmax)
    - 생성 시 diversity가 부족하고 가장 가능성 높은 토큰만 반복적으로 출력함
    - 이로 인해 "이 이 이 이 이..." 같은 현상이 발생함
    - 해결: Beam Search 또는 Top-k Sampling을 사용해야 함

#### 최종 요약: 모델 성능 저하 요인 정리
|항목|상태|설명|
|------|-----|--------------------|
|데이터 양|🔻 부족|11,823쌍은 자유 대화에 부족|
|Epoch 수|🔻 부족|최소 30~50 이상 필요|
|Embedding|	⚠️ 제한|Word2Vec은 좋지만 fine-tuning이 안됐거나 vocabulary mismatch 가능|
|디코딩 방식|	🔻 한정|Greedy decoding은 diversity 부족|
|지표 성능|🔻 낮음|BLEU 0.06은 비정상 수준|

#### 조치

|조치|설명|
|----|------|
|Epoch 50 이상으로 재학습|EarlyStopping 또는 Validation Loss 추이 기반|
|Beam Search 또는 Top-k Decoding|답변의 다양성과 질 개선|
|Fine-tune Embedding Layer|embedding.weight.requires_grad = True로 설정|
|Validation Loss 기록|학습 중 overfitting 여부 판단 가능|
|데이터 증강|paraphrasing 또는 GPT 기반 synthetic QA 생성|
