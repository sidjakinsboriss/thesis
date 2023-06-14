import torch.nn as nn


class EmailGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers, dropout, weights):
        super(EmailGRU, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(weights, padding_idx=0)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)

        output, _ = self.gru(x)
        out = output[:, -1, :]

        return self.fc(out)


class EmailLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers, dropout, weights):
        super(EmailLSTM, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(weights, padding_idx=0)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)

        lstm_output, _ = self.lstm(x)
        lstm_output = lstm_output[:, -1, :]  # Take the last output of the LSTM sequence
        logits = self.fc(lstm_output)

        return logits
