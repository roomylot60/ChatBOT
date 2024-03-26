import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, n_vocabs, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        # 단어 사전의 개수 지정
        self.n_vocabs = n_vocabs
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        # embedding: n_vocab(input)을 emb_dim vector로 변경
        # 임베딩 레이어 정의 (number of vocabs, embedding dimension)
        self.embedding = nn.Embedding(n_vocabs, emb_dim)

        # GRU (embedding dimension): LSTM에서 사용하는 cell state, hidden state을 하나로 합쳐 hidden state으로 사용
        self.gru = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # embedding(): input shape, embedding vector의 차원 수를 출력
        # permute(): 입력된 tensor의 차원 순서를 변경
        # (1,2,3)으로 구성된 3차원 tensor에 permute(2,1,0)를 적용하면 (3,2,1) 순서로 tensor가 변경
        x = self.embedding(x).permute(1, 0, 2)

        output, hidden = self.gru(x)
        return output, hidden