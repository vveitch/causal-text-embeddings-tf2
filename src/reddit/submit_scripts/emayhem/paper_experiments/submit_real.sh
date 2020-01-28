#!/usr/bin/env bash
#!/bin/sh

# should be submitted from PeerRead dir

OUTPUT_DIR_BASE=/proj/sml_netapp/projects/victor/causal-text-tf2/out/paper/reddit/real/

#rm -rf ${OUTPUT_DIR_BASE}
mkdir -p ${OUTPUT_DIR_BASE}

declare -a TREATMENTS=(
  'gender'
)

export NUM_SPLITS=10

for TREATMENT in "${TREATMENTS[@]}"; do
  export TREATMENT=${TREATMENT}
  for SPLITi in $(seq 0 $(($NUM_SPLITS - 1))); do
    export SPLIT=${SPLITi}
    NAME=o_score_t_${TREATMENT}.split${SPLIT}.seed${SEED}
    export OUTPUT_DIR=${OUTPUT_DIR_BASE}o_accepted_t_${TREATMENT}/split${SPLIT}
    sbatch --job-name=${NAME} \
      --output=${OUTPUT_DIR_BASE}/${NAME}.out \
      ./reddit/submit_scripts/emayhem/paper_experiments/run_real_causal_bert.sh
  done
done
