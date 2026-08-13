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
declare -A SEEN_HOSTS=()
for host in "${HOSTS[@]}"; do
  if [[ ! "${host}" =~ ^[a-z0-9_]+$ || -n "${SEEN_HOSTS[${host}]:-}" ]]; then
    echo "Invalid or duplicate host request: ${host}" >&2
    exit 3
  fi
  SEEN_HOSTS["${host}"]=1
  registry_count="$(awk -F '\t' -v requested="${host}" '
    NR == 1 { for (i=1; i<=NF; i++) if ($i == "host") host_col=i; next }
    host_col > 0 && $host_col == requested { count++ }
    END { print count+0 }
  ' "${PRODUCTION_REGISTRY}")"
  if [[ "${registry_count}" != "1" ]]; then
    echo "Host ${host} must occur exactly once in the production registry" >&2
    exit 3
  fi
done
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
metadata_status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", ""))' "${metadata}")"
if (( shard_count < 1 || shard_count > 1001 )); then
  echo "Shard count ${shard_count} violates the bounded Hazel array design" >&2
  exit 6
fi
if [[ "${metadata_status}" != "complete" ]]; then
  echo "Shard metadata is not complete" >&2
  exit 6
fi
last_index=$((shard_count - 1))
found_count="$(find "${SHARD_DIR}" -maxdepth 1 -type f -name 'shard_*.txt' | wc -l | tr -d ' ')"
if [[ "${found_count}" != "${shard_count}" ]]; then
  echo "Shard metadata count ${shard_count} differs from files ${found_count}" >&2
  exit 6
fi
for (( index=0; index<shard_count; index++ )); do
  printf -v shard_path '%s/shard_%05d.txt' "${SHARD_DIR}" "${index}"
  if [[ ! -s "${shard_path}" ]]; then
    echo "Missing or empty sequential shard: ${shard_path}" >&2
    exit 6
  fi
done
metadata_sha256="$(sha256sum "${metadata}" | awk '{print $1}')"
mkdir -p "$(dirname "${SUBMISSION_LOG}")" "${OUTPUT_ROOT}/_slurm_logs"
mkdir -p "${HPC_ROOT}/Outputs_Host_NAIP_SDMs/production_model_inference_august2026/_slurm_logs"
printf 'submitted_at\tregion\tproduct\thost\tjob_id\tdependency_job_id\tarray_spec\tshard_count\tshard_metadata\tshard_metadata_sha256\tmax_active_tasks_within_submission\tdry_run\n' > "${SUBMISSION_LOG}"

previous_job=""
for host in "${HOSTS[@]}"; do
  array_spec="0-${last_index}%${MAX_CONCURRENT}"
  command=(
    sbatch --parsable --partition="${GPU_PARTITION}" --gres="${GPU_GRES}"
    --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEMORY}" --time="${WALLTIME}"
    --array="${array_spec}"
    --export=ALL,HOST_SLUG="${host}",REGION="${REGION}",PRODUCT="${PRODUCT}",SHARD_DIR="${SHARD_DIR}",PRODUCTION_REGISTRY="${PRODUCTION_REGISTRY}",BUNDLE_ROOT="${BUNDLE_ROOT}"
    "${SCRIPT_DIR}/submit_inference.sbatch"
  )
  if [[ -n "${previous_job}" ]]; then
    command=("${command[@]:0:2}" --dependency="afterany:${previous_job}" "${command[@]:2}")
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN command='
    printf '%q ' "${command[@]}"
    printf '\n'
    job_id="DRY_RUN"
  else
    if ! job_id="$("${command[@]}")"; then
      job_id=SUBMISSION_FAILED
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -Is)" "${REGION}" "${PRODUCT}" "${host}" "${job_id}" \
        "${previous_job}" "${array_spec}" "${shard_count}" "${metadata}" \
        "${metadata_sha256}" "${MAX_CONCURRENT}" "${DRY_RUN}" >> "${SUBMISSION_LOG}"
      exit 7
    fi
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Is)" "${REGION}" "${PRODUCT}" "${host}" "${job_id}" \
    "${previous_job}" "${array_spec}" "${shard_count}" "${metadata}" \
    "${metadata_sha256}" "${MAX_CONCURRENT}" "${DRY_RUN}" >> "${SUBMISSION_LOG}"
  previous_job="${job_id}"
done
printf 'dry_run=%s serialized_arrays=%s max_active_tasks_within_submission=%s submission_log=%s\n' \
  "${DRY_RUN}" "${#HOSTS[@]}" "${MAX_CONCURRENT}" "${SUBMISSION_LOG}"
