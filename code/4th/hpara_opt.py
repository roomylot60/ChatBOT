import os
import json
import optuna
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertModel

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 로거 설정
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "optuna_search.log"),
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Optuna 로깅 설정
optuna.logging.disable_default_handler()
optuna.logging.enable_propagation()

# KoBERT 모델과 토크나이저 로드
kobert_model = BertModel.from_pretrained("monologg/kobert", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", use_fast=False, trust_remote_code=True)

# 데이터셋 정의
class ChatDataset(Dataset):
    def __init__(self, data):
        self.inputs = data["input"]
        self.targets = data["target"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx]["input_ids"]),
            torch.tensor(self.targets[idx]["input_ids"])
        )

def collate_fn(batch):
    input_ids, target_ids = zip(*batch)
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return input_ids, attention_mask, target_ids

# 데이터 로드
data_path = "data/output/corpus_train.json"
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)
dataset = ChatDataset(data)

# 모델 정의
class TransformerDecoderModel(nn.Module):
    def __init__(self, hidden_size=768, vocab_size=None, num_layers=4, num_heads=8, ff_dim=1024):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(f"`hidden_size` ({hidden_size}) must be divisible by `num_heads` ({num_heads})")

        self.bert = kobert_model
        self.decoder_embedding = nn.Embedding(vocab_size, hidden_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, enc_input_ids, enc_mask, dec_input_ids):
        enc_output = self.bert(input_ids=enc_input_ids, attention_mask=enc_mask).last_hidden_state
        dec_emb = self.decoder_embedding(dec_input_ids)
        tgt_mask = self.generate_square_subsequent_mask(dec_input_ids.size(1)).to(dec_input_ids.device)
        dec_output = self.decoder(tgt=dec_emb, memory=enc_output, tgt_mask=tgt_mask)
        return self.output_layer(dec_output)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

# 학습 함수
def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for enc_ids, enc_mask, tgt_ids in dataloader:
        enc_ids, enc_mask, tgt_ids = enc_ids.to(device), enc_mask.to(device), tgt_ids.to(device)
        dec_input = tgt_ids[:, :-1]
        dec_target = tgt_ids[:, 1:]

        optimizer.zero_grad()
        output = model(enc_ids, enc_mask, dec_input)
        loss = loss_fn(output.view(-1, output.size(-1)), dec_target.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 검증 함수
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for enc_ids, enc_mask, tgt_ids in dataloader:
            enc_ids, enc_mask, tgt_ids = enc_ids.to(device), enc_mask.to(device), tgt_ids.to(device)
            dec_input = tgt_ids[:, :-1]
            dec_target = tgt_ids[:, 1:]
            output = model(enc_ids, enc_mask, dec_input)
            loss = loss_fn(output.view(-1, output.size(-1)), dec_target.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Optuna 목적 함수
def objective(trial):
    num_layers = trial.suggest_int('num_layers', 2, 6)
    num_heads = trial.suggest_int('num_heads', 2, 8)
    ff_dim = trial.suggest_int('ff_dim', 512, 2048)
    hidden_size = trial.suggest_categorical('hidden_size', [512, 768])  # 호환 가능한 크기
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])

    if hidden_size % num_heads != 0:
        raise optuna.exceptions.TrialPruned()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    model = TransformerDecoderModel(
        hidden_size=hidden_size,
        vocab_size=tokenizer.vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for epoch in range(10):
        train_loss = train(model, dataloader, optimizer, loss_fn, device)
        val_loss = evaluate(model, dataloader, loss_fn, device)
        logger.info(f"Trial {trial.number} | Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return val_loss


# 스터디 저장을 위한 SQLite 스토리지 설정
study_name = "kobert_chatbot_study"
storage_name = f"sqlite:///{study_name}.db"

# 스터디 생성 또는 기존 스터디 로드
study = optuna.create_study(
    study_name=study_name,
    storage=storage_name,
    direction='minimize',
    load_if_exists=True
)

# 최적화 실행
study.optimize(objective, n_trials=20)

# 결과 출력
logger.info(f"Best Trial: {study.best_trial.number}")
logger.info(f"Best Value (Validation Loss): {study.best_trial.value:.4f}")
logger.info(f"Best Params: {study.best_trial.params}")
