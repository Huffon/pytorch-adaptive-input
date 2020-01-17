import torch
import torch.nn as nn
import torch.optim as optim

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

        self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(params.device)

    def train(self):
        print(f'The model has {self.model.count_params():,} parameters')
        print(f'Original has {246900000:,} parameters')
        best_valid_loss = float('inf')

        for epoch in range(self.params.num_epoch):
            self.model.train()

            for batch in self.train_iter:
                # Batch training
                outputs = self.model(batch)

    def validate(self):
        self.model.eval()
        valid_loss = 0

        with torch.no_grad():
            # Validation
            for batch in self.valid_iter:
                outputs = self.model(batch)

    def test(self):
        self.model.eval()
        test_loss = 0

        with torch.no_grad():
            # Test
            for batch in self.test_iter:
                outputs = self.model(batch)
