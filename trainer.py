import time

import torch
import torch.nn as nn
import torch.optim as optim

from utils import epoch_time
from model.net import TransformerLM


class Trainer:
    def __init__(self, params, train_iter=None, valid_iter=None, test_iter=None):
        self.params = params

        if params.mode == 'train':
            self.train_iter = train_iter
            self.valid_iter = valid_iter
        else:
            self.test_iter = test_iter

        self.model = TransformerLM(self.params)
        self.model.to(params.device)
        
        self.optimizer = optim.Adam(self.model.parameters())

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.criterion.to(params.device)

    def train(self):
        print(f'The model has {self.model.count_params():,} parameters')
        print(f'Original has {246900000:,} parameters')
        best_valid_loss = float('inf')

        for epoch in range(self.params.num_epoch):
            self.model.train()
            train_loss = 0
            start_time = time.time()

            for input_ids in self.train_iter:
                output = self.model(input_ids[:, :-1])
 
                preds = output.contiguous().view(-1, output.size(-1))
                # preds = [(batch size * sentence length), vocab size]
                golds = input_ids[:, 1:].contiguous().view(-1)
                # golds = [(batch size * sentence length)]
                
                loss = self.criterion(preds, golds)
                loss.backward()

                self.optimizer.step()
                
                train_loss += loss.item()

            train_loss = train_loss / len(self.train_iter)
            valid_loss = self.validate()

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), self.params.save_dir)

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

    def validate(self):
        self.model.eval()
        valid_loss = 0

        with torch.no_grad():
            for input_ids in self.valid_iter:
                output = self.model(input_ids[:, :-1])
 
                preds = output.contiguous().view(-1, output.size(-1))
                golds = input_ids[:, 1:].contiguous().view(-1)
                
                loss = self.criterion(preds, golds)
                valid_loss += loss.item()

        return valid_loss / len(self.valid_iter)

    def test(self):
        self.model.load_state_dict(torch.load(self.params.save_dir))
        self.model.eval()
        test_loss = 0

        with torch.no_grad():
            for input_ids in self.test_iter:
                output = self.model(input_ids[:, :-1])

                preds = output.contiguous().view(-1, output.size(-1))
                golds = input_ids[:, 1:].contiguous().view(-1)
                
                loss = self.criterion(preds, golds)
                test_loss += loss.item()
        
        test_loss = test_loss / len(self.test_iter)
        print(f'Test Loss: {test_loss:.3f}')
