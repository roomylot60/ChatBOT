## BERT 기반 한국어 챗봇 모델 생성

- 한국어에 특화된 BERT 모델을 활용
    - KoBERT: 한국어에 최적화된 BERT 모델로, 다양한 한국어 자연어 처리 작업에 활용 ​
    - KB-BERT: 금융 도메인에 특화된 한국어 BERT 모델로, 금융 관련 자연어 처리 작업에 활용 ​
    - KPF-BERT: 20년치의 기사 데이터를 학습한 AI 언어모델로, 한국언론진흥재단에서 공개 ​

- 모델 파인튜닝: 
    - 선택한 BERT 모델을 챗봇의 목적에 맞게 파인튜닝
    - 이를 위해서는 대화 데이터셋이 필요하며, 해당 데이터셋을 활용하여 모델을 미세 조정

- 챗봇 구축 및 배포: 
    - 파인튜닝된 모델을 활용하여 챗봇 시스템을 구축하고, 사용자와의 상호작용을 위해 배포

### KoBERT 모델 불러오기 및 설정

1. 전처리 파일(preprocess.py) 기반 데이터셋 구성
2. KoBERT 입력 형식에 맞게 인코딩 및 Dataloader 구성
3. KoBERT 모델에 파인튜닝용 Classifier 레이어 추가
4. 학습 및 저장
5. 챗봇 응답 생성

### kobert_chatbot.pt
- epoch = 3
- MEX_LEN = 64
[✅ 성능 평가 결과]
BLEU Score   : 0.0067
ROUGE-1 F1   : 0.0000
ROUGE-L F1   : 0.0000
Accuracy     : 0.0000 

### kobert_chatbot_best.pt
- epoch = 5
- MEX_LEN = 64
[✅ 성능 평가 결과]
BLEU Score   : 0.0138
ROUGE-1 F1   : 0.0000
ROUGE-L F1   : 0.0000
Accuracy     : 0.0000

- epoch = 5
- MEX_LEN = 128
- validation 수행
[✅ 성능 평가 결과]
BLEU Score   : 0.0104
ROUGE-1 F1   : 0.0000
ROUGE-L F1   : 0.0000
Accuracy     : 0.0000
Perplexity   : 39.2820

### kobert_chatbot_best_epoch_10.pt
- epoch = 10
- MEX_LEN = 128
- validation 수행
[✅ 성능 평가 결과]
BLEU Score   : 0.0104
ROUGE-1 F1   : 0.0000
ROUGE-L F1   : 0.0000
Accuracy     : 0.0000
Perplexity   : 39.2820