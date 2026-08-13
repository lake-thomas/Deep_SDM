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
SUBMISSION_LOG="${SUBMISSION_LOG:-${HPC_ROOT}/Outputs_Host_NAIP_SDMs/production_model_inference_august2026/_submission_records/${REGION}_${PRODUCT}_arrays.tsv}"

echo "Compatibility launcher: routing all shards through bounded Slurm arrays." >&2
exec bash "${SCRIPT_DIR}/submit_inference_arrays.sh" \
  "${REGION}" "${PRODUCT}" "${SHARD_DIR}" "${PRODUCTION_REGISTRY}" \
  "${BUNDLE_ROOT}" "${SUBMISSION_LOG}" "${HOSTS[@]}"
