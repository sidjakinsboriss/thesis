import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.nn import BCEWithLogitsLoss

from dl.bert.model import BERTClass
from dl.dataset_split_handler import DatasetHandler

device = 'cpu'


class BertTraining:
    def __init__(self, model, criterion, optimizer, num_epochs):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    def train_model(self, training_loader, validation_loader):
        for epoch in range(self.num_epochs):
            train_loss = 0
            valid_loss = 0

            self.model.train()
            print('############# Epoch {}: Training Start   #############'.format(epoch))
            for data in training_loader:
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)

                self.optimizer.zero_grad()

                outputs = self.model(ids, mask)

                loss = self.criterion(outputs, targets)
                loss.backward()

                self.optimizer.step()

                train_loss += loss.item()

            self.model.eval()

            with torch.no_grad():
                best_valid_loss = float('inf')
                for data in validation_loader:
                    ids = data['ids'].to(device, dtype=torch.long)
                    mask = data['mask'].to(device, dtype=torch.long)
                    targets = data['targets'].to(device, dtype=torch.float)

                    outputs = self.model(ids, mask)

                    loss = self.criterion(outputs, targets)
                    valid_loss += loss.item()

                print('############# Epoch {}: Validation End     #############'.format(epoch))

                train_loss = train_loss / len(training_loader)
                valid_loss = valid_loss / len(validation_loader)

                print('Epoch: {} \tAvgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}'.format(
                    epoch,
                    train_loss,
                    valid_loss
                ))

                if valid_loss <= best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(self.model.state_dict(), os.path.join(os.getcwd(), '../models/bert.pt'))

    def evaluate_model(self, test_loader):
        self.model.load_state_dict(torch.load(os.path.join(os.getcwd(), '../models/bert.pt')))
        self.model.eval()

        predicted = []
        ground_truth = []

        with torch.no_grad():
            for data in test_loader:
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)

                outputs = self.model(ids, mask, token_type_ids)

                outputs = torch.sigmoid(outputs)

                threshold = 0.5  # Adjust this threshold as needed

                outputs = torch.stack([(output > threshold).int() for output in outputs])

                predicted.append(outputs.cpu().numpy())
                ground_truth.append(targets.cpu().numpy())

        predicted = np.concatenate(predicted, axis=0)
        ground_truth = np.concatenate(ground_truth, axis=0)

        print(classification_report(ground_truth, predicted))


if __name__ == '__main__':
    # Hyper-parameters
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.0001

    df = pd.read_csv(os.path.join(os.getcwd(), '../../data/dataframe_unprocessed.csv'))

    dataset_handler = DatasetHandler(df, None, False)
    dataset_handler.encode_labels()
    dataset_handler.split_dataset()

    model = BERTClass()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = BCEWithLogitsLoss()

    for train_loader, val_loader, test_loader in dataset_handler.get_data_loaders(batch_size, use_bert=True):
        training = BertTraining(model, criterion, optimizer, num_epochs)
        training.train_model(train_loader, val_loader)
        training.evaluate_model(test_loader)
