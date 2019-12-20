python -m PeerRead.model.run_pretraining \
--input_files='/home/victor/Documents/bert2/dat/PeerRead/proc/arxiv-all.tf_record' \
--bert_config_file='/home/victor/Documents/bert2/pre-trained/uncased_L-12_H-768_A-12/bert_config.json' \
--init_checkpoint='/home/victor/Documents/bert2/pre-trained/uncased_L-12_H-768_A-12/checkpoint' \
--vocab_file='/home/victor/Documents/bert2/pre-trained/uncased_L-12_H-768_A-12/vocab.txt' \
--seed=0 \
--strategy_type=mirror

# --strategy_type=mirror \