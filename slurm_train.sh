#!/usr/bin/env bash

# Example of command : sh slurm_train.sh a5000 amp pspnet_r101-d8_1xb2-80k_soccernetv2-1080x1920.py --amp --resume

set -x

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-1}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:4}

mkdir -p work_dir/${JOB_NAME}_${CONFIG}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME}_${CONFIG} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    -t 9-00:00:00 \
    ${SRUN_ARGS} \
    python train.py ${CONFIG} --launcher="slurm" --work-dir work_dir/${JOB_NAME}_${CONFIG} ${PY_ARGS} > work_dir/${JOB_NAME}_${CONFIG}/train.log 2>&1 &
