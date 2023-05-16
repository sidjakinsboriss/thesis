import torch
import torch.nn as nn


# Define the RNN model
class EmailRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(EmailRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        embed = self.embedding(input)
        output, _ = self.rnn(embed)
        output = self.fc(output[:, -1, :])
        return output


# Define the training function
def train(model, optimizer, criterion, train_loader, test_loader, num_epochs):
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        model.train()
        for input, label in train_loader:
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(output, 1)
            train_acc += (predicted == label).sum().item()
        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)
        test_loss = 0.0
        test_acc = 0.0
        model.eval()
        with torch.no_grad():
            for input, label in test_loader:
                output = model(input)
                loss = criterion(output, label)
                test_loss += loss.item()
                _, predicted = torch.max(output, 1)
                test_acc += (predicted == label).sum().item()
            test_loss /= len(test_loader)
            test_acc /= len(test_loader.dataset)
        print(
            "Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}".format(
                epoch + 1, num_epochs, train_loss, train_acc, test_loss, test_acc
            )
        )
