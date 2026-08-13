#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <region> <product> <shard_dir> <production_registry> <bundle_root> <host> [host ...]" >&2
  exit 2
fi
REGION="$1"
PRODUCT="$2"
SHARD_DIR="$3"
PRODUCTION_REGISTRY="$4"
BUNDLE_ROOT="$5"
shift 5
HOSTS=("$@")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC_ROOT="${HPC_ROOT:-/rsstu/users/r/rkmeente/DSFAS_GSV_NAIP/Host_NAIP_SDM}"
mkdir -p "${HPC_ROOT}/Outputs_Host_NAIP_SDMs/production_model_inference_august2026/_slurm_logs"
GPU_PARTITION="${GPU_PARTITION:-gpu}"
GPU_MODEL="${GPU_MODEL:-l40}"
GPU_GRES="${GPU_GRES:-gpu:${GPU_MODEL}:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-64G}"
WALLTIME="${WALLTIME:-24:00:00}"
MAX_SUBMIT="${MAX_SUBMIT:-999999}"
submitted=0
for host in "${HOSTS[@]}"; do
  for shard in "${SHARD_DIR}"/shard_*.txt; do
    [[ -s "${shard}" ]] || continue
    if (( submitted >= MAX_SUBMIT )); then exit 0; fi
    shard_id="$(basename "${shard}" .txt)"
    shard_id="${shard_id#shard_}"
    sbatch --partition="${GPU_PARTITION}" --gres="${GPU_GRES}" \
      --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEMORY}" --time="${WALLTIME}" \
      --export=ALL,HOST_SLUG="${host}",REGION="${REGION}",PRODUCT="${PRODUCT}",SHARD_ID="${shard_id}",TILE_LIST="${shard}",PRODUCTION_REGISTRY="${PRODUCTION_REGISTRY}",BUNDLE_ROOT="${BUNDLE_ROOT}" \
      "${SCRIPT_DIR}/submit_inference.sbatch"
    submitted=$((submitted + 1))
  done
done
echo "submitted=${submitted}"
