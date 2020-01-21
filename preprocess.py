import re
import string
import argparse
from typing import List
from collections import defaultdict

from utils import Vocabulary


def process_corpus(lines: List[str], vocab: Vocabulary, f_output: str):
    """
    Calculate the time spent during one epoch
    
    Args:
        lines       (list): List of string which contains line of corpus
        vocab (Vocabulary): Vocabulary class instantiated using 'vocab' file
        f_output   (float): Text file to store processed corpus
    
    Returns:
        None
    """

    with open(f_output, 'w', encoding='utf-8') as f:
        for line in lines:
            line = re.sub('[^A-Za-z0-9 ]+', '', line)
            print(vocab.encode(line), file=f)


def build_vocab(f_corpus: str, f_vocab: str, min_frequency: int, max_len: int):
    """
    Build count-based vocabulary class
    
    Args:
        f_corpus      (str): Corpus file used to extract vocabulary
        f_vocab       (str): Text file to store extracted vocabulary
        min_frequency (int): Word's minimum frequency
        max_len       (int): Maximum sentence length used to zero padding
    
    Returns:
        (Vocabulary) vocabulary class instantiated using 'vocab' file
    """

    vocab = defaultdict(int)

    corpus = open(f_corpus, 'r', encoding='utf-8')
    lines = corpus.readlines()

    for line in lines:
        line = re.sub('[^A-Za-z0-9 ]+', '', line)
        for word in line.lower().split(' '):
            vocab[word] += 1

    vocab = sorted(vocab.items(), key=(lambda x: x[1]), reverse=True)[1:]
    
    with open(f_vocab, 'w', encoding='utf-8') as f:
        for word in vocab:
            if word[1] >= min_frequency:
                print(f'{word[0]}\t{word[1]}', file=f)

    return lines, Vocabulary(f_vocab, max_len)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_frequency', type=int, default=3)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--corpus', type=str, default='data/wiki.txt')
    parser.add_argument('--output', type=str, default='data/train.txt')
    parser.add_argument('--vocab', type=str, default='vocab.txt')
    args = parser.parse_args()
    
    lines, vocab = build_vocab(args.corpus, args.vocab, args.min_frequency, args.max_len)
    process_corpus(lines, vocab, args.output)
