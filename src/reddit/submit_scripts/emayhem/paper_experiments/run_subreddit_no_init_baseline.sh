#!/bin/bash
#SBATCH -A sml
#SBATCH -c 12
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:2
#SBATCH --mail-user=victorveitch@gmail.com
#SBATCH --mail-type=ALL

source activate ct-2

export BERT_BASE_DIR=/proj/sml_netapp/projects/victor/causal-text-tf2/pre-trained/uncased_L-12_H-768_A-12
export DATA_FILE=/proj/sml_netapp/dat/undocumented/reddit/proc.tf_record
export PREDICTION_FILE=$OUTPUT_DIR/predictions.tsv

echo "python -m reddit.model.run_causal_bert \
  --seed=0 \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --input_files=$DATA_FILE \
  --model_dir=${OUTPUT_DIR} \
  --max_seq_length=250 \
  --train_batch_size=32 \
  --learning_rate=3e-4 \
  --num_train_epochs=10 \
  --prediction_file=$PREDICTION_FILE \
  --learning_rate=5e-4 \
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
  --seed=0 \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --input_files=$DATA_FILE \
  --model_dir=${OUTPUT_DIR} \
  --max_seq_length=250 \
  --train_batch_size=32 \
  --num_train_epochs=10 \
  --prediction_file=$PREDICTION_FILE \
  --learning_rate=5e-4 \
  --do_masking=True \
  --num_splits=${NUM_SPLITS} \
  --test_splits=${SPLIT} \
  --dev_splits=${SPLIT} \
  --simulated=attribute \
  --beta0=${BETA0} \
  --beta1=${BETA1} \
  --gamma=${GAMMA} \
  --simulation_mode=${SIMMODE}