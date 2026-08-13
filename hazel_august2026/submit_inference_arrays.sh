#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 ]]; then
  echo "Usage: $0 <region> <product> <shard_dir> <production_registry> <bundle_root> <submission_log.tsv> <host> [host ...]" >&2
  exit 2
fi
REGION="$1"
PRODUCT="$2"
SHARD_DIR="$3"
PRODUCTION_REGISTRY="$4"
BUNDLE_ROOT="$5"
SUBMISSION_LOG="$6"
shift 6
HOSTS=("$@")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC_ROOT="${HPC_ROOT:-/rsstu/users/r/rkmeente/DSFAS_GSV_NAIP/Host_NAIP_SDM}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/share/rkmeente/talake2/host_sdm_august2026/production_model_inference_august2026}"
GPU_PARTITION="${GPU_PARTITION:-gpu}"
GPU_MODEL="${GPU_MODEL:-l40}"
GPU_GRES="${GPU_GRES:-gpu:${GPU_MODEL}:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-64G}"
WALLTIME="${WALLTIME:-24:00:00}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
DRY_RUN="${DRY_RUN:-1}"
metadata="${SHARD_DIR}/shard_metadata.json"

if [[ "${PRODUCT}" != "deterministic" && "${PRODUCT}" != "uncertainty" ]]; then
  echo "Unknown product: ${PRODUCT}" >&2
  exit 2
fi
if [[ ! -s "${metadata}" || ! -s "${SHARD_DIR}/COMPLETED" ]]; then
  echo "Shard package is incomplete: ${SHARD_DIR}" >&2
  exit 3
fi
if [[ ! -s "${PRODUCTION_REGISTRY}" || ! -s "${BUNDLE_ROOT}/COMPLETED" ]]; then
  echo "Production registry or frozen bundle is incomplete" >&2
  exit 3
fi
if [[ -e "${SUBMISSION_LOG}" ]]; then
  echo "Refusing to replace submission log: ${SUBMISSION_LOG}" >&2
  exit 4
fi
if (( MAX_CONCURRENT < 1 || MAX_CONCURRENT > 25 )); then
  echo "MAX_CONCURRENT must be between 1 and 25" >&2
  exit 5
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1" >&2
  exit 5
fi
shard_count="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["shard_count"])' "${metadata}")"
if (( shard_count < 1 || shard_count > 1001 )); then
  echo "Shard count ${shard_count} violates the bounded Hazel array design" >&2
  exit 6
fi
last_index=$((shard_count - 1))
metadata_sha256="$(sha256sum "${metadata}" | awk '{print $1}')"
mkdir -p "$(dirname "${SUBMISSION_LOG}")" "${OUTPUT_ROOT}/_slurm_logs"
mkdir -p "${HPC_ROOT}/Outputs_Host_NAIP_SDMs/production_model_inference_august2026/_slurm_logs"
printf 'submitted_at\tregion\tproduct\thost\tjob_id\tarray_spec\tshard_count\tshard_metadata\tshard_metadata_sha256\tmax_concurrent\tdry_run\n' > "${SUBMISSION_LOG}"

for host in "${HOSTS[@]}"; do
  array_spec="0-${last_index}%${MAX_CONCURRENT}"
  command=(
    sbatch --parsable --partition="${GPU_PARTITION}" --gres="${GPU_GRES}"
    --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEMORY}" --time="${WALLTIME}"
    --array="${array_spec}"
    --export=ALL,HOST_SLUG="${host}",REGION="${REGION}",PRODUCT="${PRODUCT}",SHARD_DIR="${SHARD_DIR}",PRODUCTION_REGISTRY="${PRODUCTION_REGISTRY}",BUNDLE_ROOT="${BUNDLE_ROOT}"
    "${SCRIPT_DIR}/submit_inference.sbatch"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN command='
    printf '%q ' "${command[@]}"
    printf '\n'
    job_id="DRY_RUN"
  else
    job_id="$("${command[@]}")"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Is)" "${REGION}" "${PRODUCT}" "${host}" "${job_id}" \
    "${array_spec}" "${shard_count}" "${metadata}" "${metadata_sha256}" \
    "${MAX_CONCURRENT}" "${DRY_RUN}" >> "${SUBMISSION_LOG}"
done
printf 'dry_run=%s arrays=%s submission_log=%s\n' \
  "${DRY_RUN}" "${#HOSTS[@]}" "${SUBMISSION_LOG}"
