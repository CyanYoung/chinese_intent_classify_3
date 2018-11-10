import torch
import torch.nn as nn


class Dnn(nn.Module):
    def __init__(self, embed_mat, seq_len, class_num):
        super(Dnn, self).__init__()
        self.vocab_num, self.embed_len = embed_mat.size()
        self.seq_len = seq_len
        self.class_num = class_num
        self.embed = nn.Embedding(self.vocab_num, self.embed_len, _weight=embed_mat)
        self.la1 = nn.Sequential(nn.Linear(self.embed_len, 200),
                                 nn.ReLU())
        self.la2 = nn.Sequential(nn.Linear(200, 200),
                                 nn.ReLU())
        self.ln = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(200, self.class_num))

    def forward(self, x):
        x = self.embed(x)
        x = torch.mean(x, dim=1)
        x = self.la1(x)
        x = self.la2(x)
        return self.ln(x)


class Cnn(nn.Module):
    def __init__(self, embed_mat, seq_len, class_num):
        super(Cnn, self).__init__()
        self.vocab_num, self.embed_len = embed_mat.size()
        self.seq_len = seq_len
        self.class_num = class_num
        self.embed = nn.Embedding(self.vocab_num, self.embed_len, _weight=embed_mat)
        self.cap1 = nn.Sequential(nn.Conv1d(self.embed_len, 64, kernel_size=1, padding=0),
                                  nn.ReLU(),
                                  nn.MaxPool1d(seq_len))
        self.cap2 = nn.Sequential(nn.Conv1d(self.embed_len, 64, kernel_size=2, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(seq_len + 1))
        self.cap3 = nn.Sequential(nn.Conv1d(self.embed_len, 64, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(seq_len))
        self.la = nn.Sequential(nn.Linear(192, 200),
                                nn.ReLU())
        self.ln = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(200, self.class_num))

    def forward(self, x):
        x = self.embed(x)
        x = x.view(x.size(0), self.embed_len, -1)
        x1 = self.cap1(x)
        x2 = self.cap2(x)
        x3 = self.cap3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = x.view(x.size(0), -1)
        x = self.la(x)
        return self.ln(x)


class Rnn(nn.Module):
    def __init__(self, embed_mat, seq_len, class_num):
        super(Rnn, self).__init__()
        self.vocab_num, self.embed_len = embed_mat.size()
        self.seq_len = seq_len
        self.class_num = class_num
        self.embed = nn.Embedding(self.vocab_num, self.embed_len, _weight=embed_mat)
        self.ra = nn.LSTM(self.embed_len, 200, batch_first=True)
        self.ln = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(200, self.class_num))

    def forward(self, x):
        x = self.embed(x)
        x, state = self.ra(x)
        x = x[:, -1, :]
        return self.ln(x)
