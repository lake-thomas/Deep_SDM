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
if [[ "${PRODUCT}" != "deterministic" && "${PRODUCT}" != "uncertainty" ]]; then
  echo "Unknown PRODUCT=${PRODUCT}" >&2
  exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/Outputs_Host_NAIP_SDMs/production_model_inference_august2026}"
PYTHON_BIN="${PYTHON_BIN:-/home/ncsucnr/miniforge3/envs/host-naip-sdm/bin/python}"
BATCH_SIZE="${BATCH_SIZE:-128}"
metadata="${SHARD_DIR}/shard_metadata.json"
for required in "${metadata}" "${SHARD_DIR}/COMPLETED" "${PRODUCTION_REGISTRY}" "${BUNDLE_ROOT}/COMPLETED"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing queue input: ${required}" >&2
    exit 3
  fi
done
"${PYTHON_BIN}" "${BUNDLE_ROOT}/validate_dgx_bundle.py" --bundle-root "${BUNDLE_ROOT}" --registry "${PRODUCTION_REGISTRY}" >/dev/null

metadata_status="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", ""))' "${metadata}")"
tile_manifest="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("source_manifest", ""))' "${metadata}")"
tile_manifest_sha256="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("source_manifest_sha256", ""))' "${metadata}")"
shard_count="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("shard_count", 0))' "${metadata}")"
if [[ "${metadata_status}" != "complete" || ! -s "${tile_manifest}" ]]; then
  echo "Shard metadata or source manifest is incomplete" >&2
  exit 4
fi
if [[ "$(sha256sum "${tile_manifest}" | awk '{print $1}')" != "${tile_manifest_sha256}" ]]; then
  echo "Source tile manifest changed after sharding" >&2
  exit 4
fi
if (( shard_count < 1 )); then
  echo "Shard package contains no work" >&2
  exit 4
fi

bundle_metadata="${BUNDLE_ROOT}/bundle_metadata.json"
base_seed="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["inference_defaults"]["base_seed"])' "${bundle_metadata}")"
mc_key="${PRODUCT}_mc_samples"
expected_mc_samples="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["inference_defaults"][sys.argv[2]])' "${bundle_metadata}" "${mc_key}")"
declare -A SEEN_HOSTS=()
for host in "${HOSTS[@]}"; do
  if [[ ! "${host}" =~ ^[a-z0-9_]+$ || -n "${SEEN_HOSTS[${host}]:-}" ]]; then
    echo "Invalid or duplicate host: ${host}" >&2
    exit 5
  fi
  SEEN_HOSTS["${host}"]=1
  registry_count="$(awk -F '\t' -v requested="${host}" '
    NR == 1 { for (i=1; i<=NF; i++) if ($i == "host") host_col=i; next }
    host_col > 0 && $host_col == requested { count++ }
    END { print count+0 }
  ' "${PRODUCTION_REGISTRY}")"
  if [[ "${registry_count}" != "1" ]]; then
    echo "Host ${host} must occur exactly once in the production registry" >&2
    exit 5
  fi
done

queue_root="${OUTPUT_ROOT}/_run_metadata/queues/${REGION}/${PRODUCT}"
mkdir -p "${queue_root}"
queue_status="${queue_root}/queue_status.json"
started_at="$(date -Is)"
printf '{"status":"running","region":"%s","product":"%s","host_count":%s,"shard_count":%s,"started_at":"%s"}\n' \
  "${REGION}" "${PRODUCT}" "${#HOSTS[@]}" "${shard_count}" "${started_at}" > "${queue_status}"

for host in "${HOSTS[@]}"; do
  for (( index=0; index<shard_count; index++ )); do
    printf -v shard_id '%05d' "${index}"
    tile_list="${SHARD_DIR}/shard_${shard_id}.txt"
    if [[ ! -s "${tile_list}" ]]; then
      echo "Missing sequential shard ${tile_list}" >&2
      exit 6
    fi
    HOST_SLUG="${host}" REGION="${REGION}" PRODUCT="${PRODUCT}" \
      SHARD_ID="${shard_id}" TILE_LIST="${tile_list}" \
      PRODUCTION_REGISTRY="${PRODUCTION_REGISTRY}" BUNDLE_ROOT="${BUNDLE_ROOT}" \
      PROJECT_ROOT="${PROJECT_ROOT}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
      PYTHON_BIN="${PYTHON_BIN}" BATCH_SIZE="${BATCH_SIZE}" \
      bash "${BUNDLE_ROOT}/run_inference_shard.sh"
  done

  qa_dir="${OUTPUT_ROOT}/_run_metadata/qa/${REGION}/${host}/${PRODUCT}/final"
  qa_summary="${qa_dir}/summary.json"
  if [[ -s "${qa_summary}" ]]; then
    "${PYTHON_BIN}" -c 'import json,pathlib,sys
d=json.load(open(sys.argv[1])); expected={"status":"complete","production_ready":True,"host":sys.argv[2],"region":sys.argv[3],"product":sys.argv[4],"tile_manifest_sha256":sys.argv[5],"base_seed":int(sys.argv[6]),"expected_mc_samples":int(sys.argv[7]),"output_root":str(pathlib.Path(sys.argv[8]).resolve())}
raise SystemExit(0 if all(d.get(k)==v for k,v in expected.items()) else 1)' \
      "${qa_summary}" "${host}" "${REGION}" "${PRODUCT}" \
      "${tile_manifest_sha256}" "${base_seed}" "${expected_mc_samples}" \
      "${OUTPUT_ROOT}/${REGION}/${host}/${PRODUCT}"
    continue
  fi
  if [[ -e "${qa_dir}" ]]; then
    echo "Refusing incomplete pre-existing QA directory: ${qa_dir}" >&2
    exit 7
  fi
  candidate_env="${queue_root}/${host}_candidate.env"
  "${PYTHON_BIN}" "${BUNDLE_ROOT}/resolve_production_candidate.py" \
    --registry "${PRODUCTION_REGISTRY}" --host "${host}" \
    --hpc-root "${PROJECT_ROOT}" > "${candidate_env}"
  source "${candidate_env}"
  qa_cmd=(
    "${PYTHON_BIN}" "${BUNDLE_ROOT}/august2026_inference_output_qa.py"
    --tile-manifest "${tile_manifest}"
    --output-root "${OUTPUT_ROOT}/${REGION}/${host}/${PRODUCT}"
    --host "${host}" --region "${REGION}" --product "${PRODUCT}"
    --expected-score-space "${SCORE_SPACE}"
    --base-seed "${base_seed}" --expected-mc-samples "${expected_mc_samples}"
    --output-dir "${qa_dir}"
  )
  if [[ -n "${CALIBRATOR_SHA256}" ]]; then
    qa_cmd+=(--expected-calibrator-sha256 "${CALIBRATOR_SHA256}")
  fi
  "${qa_cmd[@]}"
done

completed_at="$(date -Is)"
printf '{"status":"complete","region":"%s","product":"%s","host_count":%s,"shard_count":%s,"base_seed":%s,"mc_samples":%s,"tile_manifest":"%s","tile_manifest_sha256":"%s","started_at":"%s","completed_at":"%s"}\n' \
  "${REGION}" "${PRODUCT}" "${#HOSTS[@]}" "${shard_count}" "${base_seed}" \
  "${expected_mc_samples}" "${tile_manifest}" "${tile_manifest_sha256}" \
  "${started_at}" "${completed_at}" > "${queue_status}"
