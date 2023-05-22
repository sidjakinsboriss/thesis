import torch.nn as nn


class EmailRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EmailRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input):
        lstm_output, _ = self.lstm(input)
        lstm_output = lstm_output[:, -1, :]  # Take the last output of the LSTM sequence
        logits = self.fc(lstm_output)
        return logits
