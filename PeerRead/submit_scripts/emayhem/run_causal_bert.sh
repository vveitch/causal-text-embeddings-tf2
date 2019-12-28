#!/bin/bash
#SBATCH -A sml
#SBATCH -c 12
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:2
#SBATCH --mail-user=victorveitch@gmail.com
#SBATCH --mail-type=ALL

source activate ct-2

export INIT_DIR=/proj/sml_netapp/projects/victor/causal-text-tf2/out/pre-training/PeerRead_stable
export INIT_FILE=$INIT_DIR/ctl_step_100000.ckpt
export BERT_BASE_DIR=/proj/sml_netapp/projects/victor/causal-text-tf2/pre-trained/uncased_L-12_H-768_A-12
export DATA_FILE=/proj/sml_netapp/dat/undocumented/PeerRead/proc/arxiv-all.tf_record
export OUTPUT_DIR=/proj/sml_netapp/projects/victor/causal-text-tf2/out/test_cb_run
export PREDICTION_FILE=$OUTPUT_DIR/predictions.tsv

python -m PeerRead.model.run_causal_bert \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--init_checkpoint=$INIT_FILE \
--input_files=$DATA_FILE \
--model_dir=$OUTPUT_DIR/PeerRead \
--num_train_epochs=100 \
--seed=0 \
--prediction_file=$PREDICTION_FILE

# --strategy_type=mirror \