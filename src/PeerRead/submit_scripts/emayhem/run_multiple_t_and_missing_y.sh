#!/bin/bash
#SBATCH -A sml
#SBATCH -c 12
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:2
#SBATCH --mail-user=victorveitch@gmail.com
#SBATCH --mail-type=ALL

source activate ct-2

export INIT_DIR=/proj/sml_netapp/projects/victor/causal-text-tf2/out/pre-training/PeerRead/pretrained
export INIT_FILE=$INIT_DIR/bert_model.ckpt-102
export BERT_BASE_DIR=/proj/sml_netapp/projects/victor/causal-text-tf2/pre-trained/uncased_L-12_H-768_A-12
#export INIT_FILE=$BERT_BASE_DIR/bert_model.ckpt
export DATA_FILE=/proj/sml_netapp/dat/undocumented/PeerRead/proc/arxiv-all.tf_record
export OUTPUT_DIR=/proj/sml_netapp/projects/victor/causal-text-tf2/out/multi-t-missing-y-eval-data-test
export PREDICTION_FILE=$OUTPUT_DIR/predictions.tsv

python -m PeerRead.model.run_multiple_t_and_missing_y \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--init_checkpoint=$INIT_FILE \
--input_files=$DATA_FILE \
--model_dir=$OUTPUT_DIR \
--num_train_epochs=2 \
--seed=0 \
--prediction_file=$PREDICTION_FILE \
--label_df_file ../dat/PeerRead/proc/arxiv-all-multi-treat-and-missing-outcomes.feather \
--learning_rate=5e-4
# --strategy_type=mirror \