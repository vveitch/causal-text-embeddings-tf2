#!/usr/bin/env bash
#!/bin/sh

OUTPUT_DIR_BASE=/proj/sml_netapp/projects/victor/causal-text-tf2/out/multi-t-missing-y-LR-sweep/

#rm -rf ${OUTPUT_DIR_BASE}
mkdir -p ${OUTPUT_DIR_BASE}

declare -a LRs=(5e-4 5e-3 5e-2)

for LRi in "${LRs[@]}"; do
  export LEARN_RATE=$LRi
  NAME=multi_t_lr_$LEARN_RATE
  export OUTPUT_DIR=${OUTPUT_DIR_BASE}lr${LEARN_RATE}
  sbatch --job-name=${NAME} \
    --output=${OUTPUT_DIR_BASE}/${NAME}.out \
    ./PeerRead/submit_scripts/emayhem/run_lr_example.sh
done
