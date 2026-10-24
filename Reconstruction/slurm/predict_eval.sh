#!/bin/bash
# Shared body of predict_a40.sbatch / predict_dgx.sbatch: predict the frozen
# cases of a split with one checkpoint of a run, then evaluate them.
# Usage (sourced from the repository root): predict_eval.sh <gpus> <run> [checkpoint] [dataset] [split] [samples]
#   checkpoint: 'latest' (default) or a step number, e.g. 60000.

GPUS="$1"
RUN="${2:?usage: <run_name> [checkpoint=latest] [dataset=resplan] [split=test] [samples=1]}"
CHECKPOINT="${3:-latest}"
DATASET="${4:-resplan}"
SPLIT="${5:-test}"
SAMPLES="${6:-1}"

source Reconstruction/slurm/env.sh
CKPT_DIR="Reconstruction/runs/${RUN}/checkpoints"
# Pin 'latest' to its step now: latest.pt is a copy of the newest step_*.pt,
# and a still-running training job may replace it during this job.
if [ "${CHECKPOINT}" = "latest" ]; then
    NEWEST="$(ls "${CKPT_DIR}"/step_*.pt | sort | tail -n 1)"
    CHECKPOINT="$(basename "${NEWEST}" .pt)"
    CHECKPOINT="${CHECKPOINT#step_}"
fi
STEP=$((10#${CHECKPOINT}))
PRED_DIR="Reconstruction/runs/${RUN}/predictions/${DATASET}_${SPLIT}/$(printf 'step_%07d' "${STEP}")_k${SAMPLES}"
echo "Run ${RUN}, checkpoint step ${STEP}, ${DATASET}/${SPLIT}, K=${SAMPLES} -> ${PRED_DIR}"

torchrun --standalone --nproc_per_node="${GPUS}" -m Reconstruction.model.predict \
    --run-name "${RUN}" --mode prod --checkpoint "${STEP}" --dataset "${DATASET}" --split "${SPLIT}" --samples "${SAMPLES}"
python -m Reconstruction.evaluation.evaluate --dataset "${DATASET}" --split "${SPLIT}" --mode prod \
    --predictions "${PRED_DIR}" --name "${RUN}_step${STEP}_k${SAMPLES}"
