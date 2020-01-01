#!/bin/bash


#SBATCH -A sml
#SBATCH -c 8
#SBATCH --mail-user=dhanya.sridhar@columbia.edu
#SBATCH --mail-type=ALL
#SBATCH --nodelist=janice,gonzo,rowlf,rizzo

# source activate py3.6

python -m PeerRead.dataset.array_from_dataset \
--mode=${SIMMODE} \
--beta0=${BETA0} \
--beta1=${BETA1} \
--gamma=${GAMMA}

# --base-output-dir=/proj/sml_netapp/dat/undocumented/PeerRead/sim/peerread_buzzytitle_based/ \
# --vocab-file=/proj/sml_netapp/dat/pre-trained_models/BERT/uncased_L-12_H-768_A-12/vocab.txt \
# --data-file=/proj/sml_netapp/dat/undocumented/PeerRead/proc/arxiv-all.tf_record \