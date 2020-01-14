#!/bin/bash

# Process all PeerRead data into tf_record format to feed into Bert

PeerDir=/home/victor/Documents/causal-spe-embeddings/dat/PeerRead/

for dataset_ in $PeerDir*/; do
    echo $dataset
#    python -m data_cleaning.process_PeerRead_abstracts \
#    --review-json-dir \
#    --parsedpdf-json-dir \
#    --out-dir \
#    --out-file \
#    --vocab_file \
#    --max_abs_len
done