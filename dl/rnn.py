import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(weights, padding_idx=0)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def init_hidden(self, batch_size):
        hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                  torch.zeros(self.num_layers, batch_size, self.hidden_size))
        return hidden

    def forward(self, x, lengths):
        text_emb = self.embedding(x)

        hidden = self.init_hidden(x.shape[0])

        packed_input = pack_padded_sequence(text_emb, lengths, batch_first=True)
        packed_output, (h_n, _) = self.lstm(packed_input, hidden)

        # Split layer and direction dimension (according to docs)
        h_n = h_n.view(self.num_layers, 2, 32, self.hidden_size)
        # The last hidden state (last w.r.t. number of layers)
        h_n = h_n[-1]
        # Concatenate directions
        h_fwd, h_bwd = h_n[0], h_n[1]
        h_n = torch.cat((h_fwd, h_bwd), dim=1)  # Concatenate both states

        logits = self.fc(h_n)

        return logits
