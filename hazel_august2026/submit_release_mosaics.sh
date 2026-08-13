#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 5 ]]; then
  echo "Usage: $0 <host> <region> <tile_manifest> <bundle_root> <expected_tile_count>" >&2
  exit 2
fi
HOST_SLUG="$1"
REGION="$2"
TILE_MANIFEST="$3"
BUNDLE_ROOT="$4"
EXPECTED_INPUT_COUNT="$5"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC_ROOT="${HPC_ROOT:-/rsstu/users/r/rkmeente/DSFAS_GSV_NAIP/Host_NAIP_SDM}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HPC_ROOT}/Outputs_Host_NAIP_SDMs/production_model_inference_august2026}"
mkdir -p "${OUTPUT_ROOT}/_slurm_logs"
jobs=()
for release_product in prediction uncertainty count; do
  case "${release_product}" in
    prediction) source_product=deterministic; source_suffix=prediction ;;
    uncertainty) source_product=uncertainty; source_suffix=uncertainty ;;
    count) source_product=deterministic; source_suffix=count ;;
  esac
  source_root="${OUTPUT_ROOT}/${REGION}/${HOST_SLUG}/${source_product}"
  export_vars="ALL,HOST_SLUG=${HOST_SLUG},REGION=${REGION},RELEASE_PRODUCT=${release_product},SOURCE_SUFFIX=${source_suffix},SOURCE_OUTPUT_ROOT=${source_root},TILE_MANIFEST=${TILE_MANIFEST},BUNDLE_ROOT=${BUNDLE_ROOT}"
  job="$(sbatch --export="${export_vars}" "${SCRIPT_DIR}/mosaic_product.sbatch" | awk '{print $4}')"
  jobs+=("${job}")
done
dependency="$(IFS=:; echo "${jobs[*]}")"
sbatch --dependency="afterok:${dependency}" \
  --export=ALL,HOST_SLUG="${HOST_SLUG}",REGION="${REGION}",EXPECTED_INPUT_COUNT="${EXPECTED_INPUT_COUNT}",BUNDLE_ROOT="${BUNDLE_ROOT}" \
  "${SCRIPT_DIR}/quickviews_and_mosaic_qa.sbatch"
