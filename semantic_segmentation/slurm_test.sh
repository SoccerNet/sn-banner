#!/usr/bin/env bash

export MKL_NUM_THREADS=2
export OMP_NUM_THREADS=2

# Example of command : sh slurm_test.sh a5000 n mask2former_swin-t_1xb2-90k_soccernet.py work_dir/n_mask2former_swin-t_1xb2-90k_soccernet.py/best_mIoU_iter_7000.pth --tta

set -x

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
CHECKPOINT=$4
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-1}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}

mkdir -p work_dir/${JOB_NAME}_${CONFIG}/test

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME}_${CONFIG} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --mem=24G \
    -t 9-00:00:00 \
    ${SRUN_ARGS} \
    python test.py ${CONFIG} ${CHECKPOINT} --launcher="slurm" --work-dir work_dir/${JOB_NAME}_${CONFIG}/test --out out_${CONFIG}/ ${PY_ARGS} > work_dir/${JOB_NAME}_${CONFIG}/test.log 2>&1 &
                                             # I think it should be --lancher=slurm 
