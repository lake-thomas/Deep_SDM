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
ACTIVE_OUTPUT_ROOT="${ACTIVE_OUTPUT_ROOT:-/share/rkmeente/talake2/host_sdm_august2026/production_model_inference_august2026}"
DURABLE_OUTPUT_ROOT="${DURABLE_OUTPUT_ROOT:-${HPC_ROOT}/Outputs_Host_NAIP_SDMs/production_model_inference_august2026}"
MAX_CONCURRENT="${MAX_CONCURRENT:-1}"
DRY_RUN="${DRY_RUN:-1}"
SUBMISSION_LOG="${SUBMISSION_LOG:-${DURABLE_OUTPUT_ROOT}/_submission_records/${REGION}_${HOST_SLUG}_mosaic_array.tsv}"
if [[ ! -s "${TILE_MANIFEST}" || ! -s "${BUNDLE_ROOT}/COMPLETED" ]]; then
  echo "Tile manifest or frozen bundle is incomplete" >&2
  exit 3
fi
if [[ ! "${EXPECTED_INPUT_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPECTED_INPUT_COUNT must be a positive integer" >&2
  exit 3
fi
if (( MAX_CONCURRENT < 1 || MAX_CONCURRENT > 3 )); then
  echo "MAX_CONCURRENT must be between 1 and 3" >&2
  exit 3
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1" >&2
  exit 3
fi
if [[ -e "${SUBMISSION_LOG}" ]]; then
  echo "Refusing to replace submission log: ${SUBMISSION_LOG}" >&2
  exit 4
fi
mkdir -p "${DURABLE_OUTPUT_ROOT}/_slurm_logs"
mkdir -p "$(dirname "${SUBMISSION_LOG}")"
array_spec="0-2%${MAX_CONCURRENT}"
array_command=(
  sbatch --parsable --array="${array_spec}"
  --export=ALL,HOST_SLUG="${HOST_SLUG}",REGION="${REGION}",TILE_MANIFEST="${TILE_MANIFEST}",BUNDLE_ROOT="${BUNDLE_ROOT}",ACTIVE_OUTPUT_ROOT="${ACTIVE_OUTPUT_ROOT}",MOSAIC_OUTPUT_ROOT="${DURABLE_OUTPUT_ROOT}"
  "${SCRIPT_DIR}/mosaic_product.sbatch"
)
if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'DRY_RUN mosaic_array_command='
  printf '%q ' "${array_command[@]}"
  printf '\n'
  array_job=DRY_RUN
  qa_job=DRY_RUN
else
  array_job="$("${array_command[@]}")"
  if ! qa_job="$(sbatch --parsable --dependency="afterok:${array_job}" \
      --export=ALL,HOST_SLUG="${HOST_SLUG}",REGION="${REGION}",EXPECTED_INPUT_COUNT="${EXPECTED_INPUT_COUNT}",BUNDLE_ROOT="${BUNDLE_ROOT}",MOSAIC_OUTPUT_ROOT="${DURABLE_OUTPUT_ROOT}" \
      "${SCRIPT_DIR}/quickviews_and_mosaic_qa.sbatch")"; then
    qa_job=QA_SUBMISSION_FAILED
    printf 'submitted_at\thost\tregion\tmosaic_array_job_id\tqa_job_id\tarray_spec\texpected_input_count\tdry_run\n' > "${SUBMISSION_LOG}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date -Is)" "${HOST_SLUG}" "${REGION}" "${array_job}" "${qa_job}" \
      "${array_spec}" "${EXPECTED_INPUT_COUNT}" "${DRY_RUN}" >> "${SUBMISSION_LOG}"
    exit 5
  fi
fi
printf 'submitted_at\thost\tregion\tmosaic_array_job_id\tqa_job_id\tarray_spec\texpected_input_count\tdry_run\n' > "${SUBMISSION_LOG}"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$(date -Is)" "${HOST_SLUG}" "${REGION}" "${array_job}" "${qa_job}" \
  "${array_spec}" "${EXPECTED_INPUT_COUNT}" "${DRY_RUN}" >> "${SUBMISSION_LOG}"
printf 'dry_run=%s mosaic_array_job_id=%s qa_job_id=%s submission_log=%s\n' \
  "${DRY_RUN}" "${array_job}" "${qa_job}" "${SUBMISSION_LOG}"
