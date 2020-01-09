python -m PeerRead.model.run_pretraining \
--input_files='../dat/PeerRead/proc/arxiv-all.tf_record' \
--bert_config_file='../pre-trained/uncased_L-12_H-768_A-12/bert_config.json' \
--init_checkpoint='../pre-trained/uncased_L-12_H-768_A-12/checkpoint' \
--vocab_file='../pre-trained/uncased_L-12_H-768_A-12/vocab.txt' \
--seed=0 \
--strategy_type=mirror

# --strategy_type=mirror \