#!/usr/bin/env bash

OUTPUT_DIR_BASE=/proj/sml_netapp/projects/victor/causal-text-tf2/out/paper/PeerRead/propensity_simulation/
mkdir -p ${OUTPUT_DIR_BASE}

export NUM_SPLITS=10

export BETA0=0.25
export GAMMA=0.0
#declare -a SIMMODES=('simple' 'multiplicative' 'interaction')
declare -a SIMMODES=('simple')
declare -a BETA1S=(5.0)
declare -a EXOS=(0.0 0.2 0.4 0.6 0.8 1.0)

for SIMMODEj in "${SIMMODES[@]}"; do
    export SIMMODE=${SIMMODEj}
    for BETA1j in "${BETA1S[@]}"; do
        export BETA1=${BETA1j}
        for EXOj in "${EXOS[@]}"; do
            export EXOG=${EXOj}
            for SPLITi in $(seq 0 $(($NUM_SPLITS-1))); do
                export SPLIT=${SPLITi}
                export OUTPUT_DIR=${OUTPUT_DIR_BASE}mode${SIMMODE}/beta0${BETA0}.beta1${BETA1}.exog${EXOG}/split${SPLIT}
                NAME=mode${SIMMODE}.beta0${BETA0}.beta1${BETA1}.exog${EXOG}.split${SPLIT}
                sbatch --job-name=peerread_propsim_${NAME} \
                   --output=${OUTPUT_DIR_BASE}${NAME}.out \
                   ./PeerRead/submit_scripts/emayhem/paper_experiments/run_propensity_sim.sh
            done
        done
    done
done