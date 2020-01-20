import argparse

import torch

from trainer import Trainer
from utils import build_iter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(params):
    if params.mode == 'train':
        train_iter, valid_iter = build_iter(params.mode, params.batch_size)
        train_iter = [torch.randint(50001, (4, 10)).to(params.device)]
        valid_iter = [torch.randint(50000, (4, 10)).to(params.device)]
        trainer = Trainer(params, train_iter=train_iter, valid_iter=valid_iter)
        trainer.train()
    else:
        test_iter = build_iter(params.mode, params.batch_size)
        test_iter = [torch.randint(40000, (4, 10)).to(params.device)]
        trainer = Trainer(params, test_iter=test_iter)
        trainer.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--save_dir', type=str, default='model.pt')
    
    # hyper params: training
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--num_epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--device', type=str, default=device)
    
    # hyper params: architecture
    parser.add_argument('--cut_off', type=str, default='20000,60000')
    parser.add_argument('--embed_factor', type=int, default=4)
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--ffn_dim', type=int, default=4096)
    parser.add_argument('--num_layers', type=int, default=16)
    parser.add_argument('--num_heads', type=int, default=16)

    # Add pre-built vocab size to params
    # vocab = json.load(open('vocab.json'))
    parser.add_argument('--vocab_size', type=int, default=60000)

    args = parser.parse_args()

    main(args)
