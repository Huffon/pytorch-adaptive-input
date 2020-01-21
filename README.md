# Adaptive Input Representations for Neural Language Modeling

(Unofficial) PyTorch implementation of **Adaptive Input Representations for Neural Language Modeling**

## Usage

- To build **vocabulary** and **processed corpus**, run following code snippet

```bash
python preprocess.py \
    --min_frequency   MIN_FREQUENCY (int)
    --max_len         MAX_LEN       (int)
    --corpus          CORPUS        (str)
    --output          OUTPUT        (str)
    --vocab           VOCAB         (str)
```

- To **train** model, run following code snippet

```bash
python main.py \
    --mode train
    --save_dir     SAVE_DIR     (str)
    --max_len      MAX_LEN      (int)
    --num_epoch    NUM_EPOCH    (int)
    --batch_size   BATCH_SIZE   (int)
    --dropout      DROPOUT    (float)
    --clip         CLIP       (float)
    --cut_off      CUT_OFF      (str)
    --embed_factor EMBED_FACTOR (int)
    --embed_dim    EMBED_DIM    (int)
    --hidden_dim   HIDDEN_DIM   (int)
    --ffn_dim      FFN_DIM      (int)
    --num_layers   NUM_LAYERS   (int)
    --num_heads    NUM_HEADS    (int)
```

<br>

## TODO

- [x] Implement **Language Modeling** logic
- [x] Implement `Vocabulary` class with **special tokens**
- [x] Implement `batchify` logic using `Vocabulary`
- [ ] Replace Softmax layer with `AdaptiveSoftmax`
- [ ] Implement Custom optimizer
- [ ] Experiment with **WikiText-103**

<br>

## References

- [Official implementation of Adaptive Input](https://github.com/pytorch/fairseq/blob/fb76dac1c4e314db75f9d7a03cb4871c532000cb/fairseq/modules/adaptive_input.py#L13)
- [Official implementation of Adaptive Softmax](https://github.com/pytorch/fairseq/blob/fb76dac1c4e314db75f9d7a03cb4871c532000cb/fairseq/modules/adaptive_softmax.py#L50)
- [Fairseq's Transformer LM](https://github.com/pytorch/fairseq/blob/master/fairseq/models/transformer_lm.py)
- [Hugging Face's Open AI Transformer LM](https://github.com/huggingface/pytorch-openai-transformer-lm)