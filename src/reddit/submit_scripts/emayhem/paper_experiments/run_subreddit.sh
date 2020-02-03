#!/bin/bash
#SBATCH -A sml
#SBATCH -c 12
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL

source activate ct-2

export INIT_DIR=/proj/sml_netapp/projects/victor/causal-text-tf2/out/pre-training/reddit/pretrained
export INIT_FILE=$INIT_DIR/bert_model.ckpt-103
export BERT_BASE_DIR=/proj/sml_netapp/projects/victor/causal-text-tf2/pre-trained/uncased_L-12_H-768_A-12
#export INIT_FILE=$BERT_BASE_DIR/bert_model.ckpt
export DATA_FILE=/proj/sml_netapp/dat/undocumented/reddit/proc.tf_record
#export OUTPUT_DIR=/proj/sml_netapp/projects/victor/causal-text-tf2/out/cb_test
export PREDICTION_FILE=$OUTPUT_DIR/predictions.tsv

echo "python -m reddit.model.run_causal_bert \
  --seed=0 \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --init_checkpoint=$INIT_FILE \
  --input_files=$DATA_FILE \
  --model_dir=${OUTPUT_DIR} \
  --max_seq_length=128 \
  --train_batch_size=64 \
  --learning_rate=3e-4 \
  --num_train_epochs=10 \
  --prediction_file=$PREDICTION_FILE \
  --learning_rate=3e-5 \
  --do_masking=True \
  --num_splits=${NUM_SPLITS} \
  --test_splits=${SPLIT} \
  --dev_splits=${SPLIT} \
  --simulated=attribute \
  --beta0=${BETA0} \
  --beta1=${BETA1} \
  --gamma=${GAMMA} \
  --simulation_mode=${SIMMODE}"

python -m reddit.model.run_causal_bert \
  --mode=${MODE} \
  --seed=0 \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --init_checkpoint=$INIT_FILE \
  --input_files=$DATA_FILE \
  --model_dir=${OUTPUT_DIR} \
  --max_seq_length=128 \
  --train_batch_size=64 \
  --num_train_epochs=10 \
  --prediction_file=$PREDICTION_FILE \
  --learning_rate=3e-5 \
  --do_masking=True \
  --subreddits=${SUBREDDITS} \
  --num_splits=${NUM_SPLITS} \
  --test_splits=${SPLIT} \
  --dev_splits=${SPLIT} \
  --simulated=attribute \
  --beta0=${BETA0} \
  --beta1=${BETA1} \
  --gamma=${GAMMA} \
  --simulation_mode=${SIMMODE}