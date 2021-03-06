import torch
from torch.utils import data


def build_iter(mode: str, batch_size: int):
    """
    Build iterator corresponding to each mode
    
    Args:
        mode (str): training start time
    
    Returns:
        (DataLoader)
    """

    if mode == 'train':
        f_train = open('data/train.txt', 'r', encoding='utf-8')
        train_data = [torch.LongTensor(eval(line.replace('\n', ''))) for line in f_train.readlines()]
        train_iter = data.DataLoader(train_data, batch_size=batch_size, 
                                                 shuffle=True,
                                                 num_workers=4)
        
        f_valid = open('data/valid.txt', 'r', encoding='utf-8')
        valid_data = [torch.LongTensor(eval(line.replace('\n', ''))) for line in f_valid.readlines()]
        valid_iter = data.DataLoader(valid_data, batch_size=batch_size, 
                                                 shuffle=True,
                                                 num_workers=4)
        return train_iter, valid_iter

    else:
        f_test = open('data/test.txt', 'r', encoding='utf-8')
        test_data = [torch.LongTensor(eval(line.replace('\n', ''))) for line in f_test.readlines()]
        test_iter = data.DataLoader(test_data, batch_size=batch_size, 
                                               shuffle=True,
                                               num_workers=4)
        return test_iter


def epoch_time(start_time: float, end_time: float):
    """
    Calculate the time spent during one epoch
    
    Args:
        start_time (float): training start time
        end_time   (float): training end time
    
    Returns:
        (int, int) elapsed_mins and elapsed_sec spent during one epoch
    """

    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs


class Vocabulary:
    """Vocabulary class trained with 'training corpus'
    """
    def __init__(self, f_vocab: str, max_len: int):
        self.max_len = max_len
        self.word2idx = {'<pad>': 0,
                         '<sos>': 1,
                         '<eos>': 2,
                         '<unk>': 3}
        self.idx2word = {}
        self.f_vocab = f_vocab
        self.build_vocab()

    def build_vocab(self):
        " Build vocabulary using count based text "
        with open(self.f_vocab, 'r', encoding='utf-8') as f:
            words = f.readlines()

            for word in words:
                word = word.split('\t')[0]
                self.word2idx[word] = len(self.word2idx)

        for word, idx in self.word2idx.items():
            self.idx2word[idx] = word

    def encode(self, x):
        " Encode word sentence into index sentence "
        encoded = [self.word2idx['<sos>']] + \
                  [self.word2idx[word] if word in self.word2idx 
                                       else self.word2idx['<unk>'] for word in x]

        if len(encoded) >= self.max_len:
            encoded = encoded[:self.max_len-1]
        
        encoded += [self.word2idx['<eos>']]

        if len(encoded) < self.max_len:
            encoded = self.zero_pad(encoded)
        return encoded

    def zero_pad(self, x):
        " Add zero padding to short length sentence "
        n_pad = self.max_len - len(x)
        x.extend([0] * n_pad)
        return x

    def get_pad(self):
        " Return the index of <pad> token "
        return self.word2idx['<pad>']

    def get_sos(self):
        " Return the index of <sos> token "
        return self.word2idx['<sos>']

    def get_eos(self):
        " Return the index of <eos> token "
        return self.word2idx['<eos>']

    def get_unk(self):
        " Return the index of <unk> token "
        return self.word2idx['<unk>']