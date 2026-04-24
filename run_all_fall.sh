#!/usr/bin/env bash
# Run vid2sim pipeline on every case in vid2sim_dataset_fall.
# Each case runs in its own Python process — when the process exits, the CUDA
# context is destroyed, so GPU memory is fully released between cases.
# Sequential, not parallel.

set -u  # don't set -e: one bad case shouldn't kill the whole sweep

cd "$(dirname "$0")"

DATASET_DIR="../PhysTwin/vid2sim_dataset_fall"
OUTPUT_DIR="outputs_fall"
CONFIG="config/gso.yaml"
LOG_DIR="${OUTPUT_DIR}/_logs"
mkdir -p "${LOG_DIR}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vid2sim

CASES=$(ls "${DATASET_DIR}" | sort)
TOTAL=$(echo "${CASES}" | wc -l)
i=0

for case in ${CASES}; do
    [ -d "${DATASET_DIR}/${case}" ] || continue
    i=$((i+1))
    log="${LOG_DIR}/${case}.log"
    echo "=== [${i}/${TOTAL}] ${case} $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "    log: ${log}"
    start=$(date +%s)
    python run_pipeline.py \
        --config "${CONFIG}" \
        --dataset_dir "${DATASET_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --data_name "${case}" \
        > "${log}" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - start ))
    if [ ${rc} -eq 0 ]; then
        echo "    ok (${elapsed}s)"
    else
        echo "    FAILED rc=${rc} (${elapsed}s) — see ${log}"
    fi
done

echo "All done at $(date '+%Y-%m-%d %H:%M:%S')"
