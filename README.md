# Adaptive Input Representations for Neural Language Modeling

(Unofficial) PyTorch implementation of **Adaptive Input Representations for Neural Language Modeling**

## Usage

- To train model, run following code snippet

```python
python main.py \
    --mode train
    --max_len      MAX_LEN
    --num_epoch    NUM_EPOCH
    --batch_size   BATCH_SIZE
    --dropout      DROPOUT
    --cut_off      CUT_OFF
    --embed_factor EMBED_FACTOR
    --embed_dim    EMBED_DIM
    --hidden_dim   HIDDEN_DIM
    --ffn_dim      FFN_DIM
    --num_layers   NUM_LAYERS
    --num_heads    NUM_HEADS
```

## TODO

- [ ] Implement Language Modeling logic
- [ ] Implement batchify logic

## References

- [Official implementation of Adaptive Input](https://github.com/pytorch/fairseq/blob/fb76dac1c4e314db75f9d7a03cb4871c532000cb/fairseq/modules/adaptive_input.py#L13)
- [Fairseq's Transformer LM](https://github.com/pytorch/fairseq/blob/master/fairseq/models/transformer_lm.py)
