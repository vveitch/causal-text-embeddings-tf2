#!/usr/bin/env bash

OUTPUT_DIR_BASE=/proj/sml_netapp/projects/victor/causal-text-tf2/out/paper/reddit/subreddit-baselines/no-init/
mkdir -p ${OUTPUT_DIR_BASE}

export NUM_SPLITS=3

#declare -a SIMMODES=('simple' 'multiplicative' 'interaction')
declare -a SIMMODES=('simple')

export BETA0=1.0
declare -a BETA1S=(10.0)
declare -a GAMMAS=(1.0)

for SIMMODEj in "${SIMMODES[@]}"; do
    export SIMMODE=${SIMMODEj}
    for BETA1j in "${BETA1S[@]}"; do
        export BETA1=${BETA1j}
        for GAMMAj in "${GAMMAS[@]}"; do
            export GAMMA=${GAMMAj}
            for SPLITi in $(seq 0 $(($NUM_SPLITS-1))); do
                export SPLIT=${SPLITi}
                export OUTPUT_DIR=${OUTPUT_DIR_BASE}mode${SIMMODE}/beta0${BETA0}.beta1${BETA1}.gamma${GAMMA}/split${SPLIT}
                NAME=mode${SIMMODE}.beta0${BETA0}.beta1${BETA1}.gamma${GAMMA}.split${SPLIT}
                sbatch --job-name=subredditsim_${NAME} \
                   --output=${OUTPUT_DIR_BASE}${NAME}.out \
                   ./reddit/submit_scripts/emayhem/paper_experiments/run_subreddit_no_masking_baseline.sh
            done
        done
    done
done