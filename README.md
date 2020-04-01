# Causal-Bert TF2
This is a reference Tensorflow 2.1 / Keras implementation of the "causal bert" method described in  [Using Text Embeddings for Causal Inference](arxiv.org/abs/1905.12741).
This method provides a way to estimate causal effects when either 
(1) a treatment and outcome are both influenced by confounders, and information about the confounding is contained in a 
text passage. For example, we consider estimating the effect of adding a theorem to a paper on whether or not the paper 
is accepted at a computer science conference, adjusting for the paper's abstract (topic, writing quality, etc)
(2) a treatment affecting an outcome is mediated by text. For example, we consider whether the score of a reddit post is
affected by publicly listing the gender of the author, adjusting for the text of the post

This is a reference implementation to make it easier for others to use and build on the project. The official code,
including instructions to reproduce the experiments, is available [here](https://github.com/blei-lab/causal-text-embeddings). (In Tensorflow 1.13)

All code in tf_official is taken from https://github.com/tensorflow/models/tree/master/official 
(and subject to their liscensing requirements)

# Instructions
1. Download BERT-Base, Uncased pre-trained model following instructions at https://github.com/tensorflow/models/tree/master/official/nlp/bert
Extract to ../pre-trained/uncased_L-12_H-768_A-12

2. in src/  
```
    run python -m PeerRead.model.run_causal_bert \
        --input_files=../dat/PeerRead/proc/arxiv-all.tf_record \ 
        --bert_config_file=../pre-trained/uncased_L-12_H-768_A-12/bert_config.json \ 
        --init_checkpoint=../pre-trained/uncased_L-12_H-768_A-12/bert_model.ckpt \ 
        --vocab_file=../pre-trained/uncased_L-12_H-768_A-12/vocab.txt \ 
        --seed=0 \ 
        --strategy_type=mirror \ 
        --train_batch_size=32
```

# Notes
1. This reference implementation doesn't necessarily reproduce paper results---I haven't messed around w/ weighting of unsupervised and supervised losses
2. PeerRead data from: github.com/allenai/PeerRead
3. Model performance is usually significantly improved by doing unsupervised pre-training on your dataset. 
See PeerRead/model/run_pretraining for how to do this
