import torch.nn as nn


class EmailRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers, dropout, weights):
        super(EmailRNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(weights, padding_idx=0)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)

        lstm_output, _ = self.lstm(x)
        lstm_output = lstm_output[:, -1, :]  # Take the last output of the LSTM sequence
        logits = self.fc(lstm_output)

        return logits
