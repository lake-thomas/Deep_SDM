#!/usr/bin/env bash
set -euo pipefail

HOST_SLUG="${HOST_SLUG:?Set HOST_SLUG}"
REGION="${REGION:?Set REGION}"
PRODUCT="${PRODUCT:?Set PRODUCT to deterministic or uncertainty}"
SHARD_ID="${SHARD_ID:?Set SHARD_ID}"
TILE_LIST="${TILE_LIST:?Set TILE_LIST}"
PRODUCTION_REGISTRY="${PRODUCTION_REGISTRY:?Set PRODUCTION_REGISTRY}"
BUNDLE_ROOT="${BUNDLE_ROOT:?Set BUNDLE_ROOT}"
if [[ "${PRODUCT}" != "deterministic" && "${PRODUCT}" != "uncertainty" ]]; then
  echo "Unknown PRODUCT=${PRODUCT}" >&2
  exit 2
fi
for required in "${TILE_LIST}" "${PRODUCTION_REGISTRY}" "${BUNDLE_ROOT}/COMPLETED"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing required inference input: ${required}" >&2
    exit 3
  fi
done

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/Outputs_Host_NAIP_SDMs/production_model_inference_august2026}"
PYTHON_BIN="${PYTHON_BIN:-/home/ncsucnr/miniforge3/envs/host-naip-sdm/bin/python}"
BATCH_SIZE="${BATCH_SIZE:-128}"
RUN_METADATA="${OUTPUT_ROOT}/_run_metadata"
OUT_DIR="${OUTPUT_ROOT}/${REGION}/${HOST_SLUG}/${PRODUCT}/shard_${SHARD_ID}"
STATUS_DIR="${RUN_METADATA}/status/${REGION}/${HOST_SLUG}/${PRODUCT}"
LOG_DIR="${RUN_METADATA}/logs/${REGION}/${HOST_SLUG}/${PRODUCT}"
mkdir -p "${OUT_DIR}" "${STATUS_DIR}" "${LOG_DIR}"

"${PYTHON_BIN}" "${BUNDLE_ROOT}/validate_dgx_bundle.py" --bundle-root "${BUNDLE_ROOT}" --registry "${PRODUCTION_REGISTRY}" >/dev/null
candidate_env="${STATUS_DIR}/shard_${SHARD_ID}_candidate.env"
"${PYTHON_BIN}" "${BUNDLE_ROOT}/resolve_production_candidate.py" --registry "${PRODUCTION_REGISTRY}" --host "${HOST_SLUG}" --hpc-root "${PROJECT_ROOT}" > "${candidate_env}"
source "${candidate_env}"

bundle_metadata="${BUNDLE_ROOT}/bundle_metadata.json"
registry_sha256="$(sha256sum "${PRODUCTION_REGISTRY}" | awk '{print $1}')"
bundle_metadata_sha256="$(sha256sum "${bundle_metadata}" | awk '{print $1}')"
tile_list_sha256="$(sha256sum "${TILE_LIST}" | awk '{print $1}')"
base_seed="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["inference_defaults"]["base_seed"])' "${bundle_metadata}")"
mc_samples=1
if [[ "${PRODUCT}" == "uncertainty" ]]; then
  mc_samples="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["inference_defaults"]["uncertainty_mc_samples"])' "${bundle_metadata}")"
fi
cmd=(
  "${PYTHON_BIN}" "${BUNDLE_ROOT}/code_snapshot/inference_multimodal_production.py"
  --config "${CONFIG_PATH}" --checkpoint "${CHECKPOINT_PATH}" --strict-checkpoint
  --tile-list "${TILE_LIST}" --output-dir "${OUT_DIR}"
  --worldclim-dir "${PROJECT_ROOT}/Env_Data/Worldclim"
  --ghm-raster "${PROJECT_ROOT}/Env_Data/Global_Human_Modification/gHM_WGS84.tif"
  --dem-raster "${PROJECT_ROOT}/Env_Data/Topography_3DEP/mosaic/dem_3dep_13_epsg5070_30m.tif"
  --slope-raster "${PROJECT_ROOT}/Env_Data/Topography_3DEP/mosaic/slope_degrees_3dep_13_epsg5070_30m.tif"
  --northness-raster "${PROJECT_ROOT}/Env_Data/Topography_3DEP/mosaic/northness_3dep_13_epsg5070_30m.tif"
  --eastness-raster "${PROJECT_ROOT}/Env_Data/Topography_3DEP/mosaic/eastness_3dep_13_epsg5070_30m.tif"
  --topo-normalization-stats "${PROJECT_ROOT}/Env_Data/Topography_3DEP/mosaic/topo_norm_stats_3dep_30m.json"
  --chip-size 256 --stride 64 --batch-size "${BATCH_SIZE}"
  --mc-samples "${mc_samples}" --seed "${base_seed}"
  --env-clip-conus --fast-topo-scalars --blend-window hann --blend-edge-floor 0.05 --skip-existing
)
if [[ -n "${CALIBRATOR_PATH}" ]]; then cmd+=(--platt-calibrator "${CALIBRATOR_PATH}"); fi

status_json="${STATUS_DIR}/shard_${SHARD_ID}.json"
log_file="${LOG_DIR}/shard_${SHARD_ID}.log"
started_at="$(date -Is)"
started_epoch="$(date +%s)"
printf '{"status":"running","host":"%s","region":"%s","product":"%s","shard_id":"%s","started_at":"%s"}\n' "${HOST_SLUG}" "${REGION}" "${PRODUCT}" "${SHARD_ID}" "${started_at}" > "${status_json}"
printf 'command=' > "${log_file}"; printf '%q ' "${cmd[@]}" >> "${log_file}"; printf '\n' >> "${log_file}"
set +e; "${cmd[@]}" >> "${log_file}" 2>&1; rc="$?"; set -e
summary="${OUT_DIR}/tiled_inference_summary.json"
expected_count="$(awk 'NF {count++} END {print count+0}' "${TILE_LIST}")"
if [[ "${rc}" -eq 0 ]]; then
  set +e
  "${PYTHON_BIN}" -c 'import json,sys; d=json.load(open(sys.argv[1])); n=int(sys.argv[2]); ok={"ok","skipped_existing"}; raise SystemExit(0 if len(d)==n and all(x.get("status") in ok for x in d) else 1)' "${summary}" "${expected_count}"
  summary_rc="$?"; set -e
  if [[ "${summary_rc}" -ne 0 ]]; then rc=9; fi
fi
prediction_count="$(find "${OUT_DIR}" -maxdepth 1 -name '*_prediction.tif' | wc -l | tr -d ' ')"
uncertainty_count="$(find "${OUT_DIR}" -maxdepth 1 -name '*_uncertainty.tif' | wc -l | tr -d ' ')"
count_count="$(find "${OUT_DIR}" -maxdepth 1 -name '*_count.tif' | wc -l | tr -d ' ')"
status=completed; if [[ "${rc}" -ne 0 ]]; then status=failed; fi
completed_at="$(date -Is)"
completed_epoch="$(date +%s)"
elapsed_seconds="$((completed_epoch - started_epoch))"
printf '{"status":"%s","returncode":%s,"host":"%s","region":"%s","product":"%s","shard_id":"%s","tile_list":"%s","tile_list_sha256":"%s","expected_tile_count":%s,"output_dir":"%s","prediction_count":%s,"uncertainty_count":%s,"count_count":%s,"base_seed":%s,"mc_samples":%s,"production_registry_sha256":"%s","bundle_metadata_sha256":"%s","started_at":"%s","completed_at":"%s","elapsed_seconds":%s,"log_file":"%s","experiment":"%s"}\n' "${status}" "${rc}" "${HOST_SLUG}" "${REGION}" "${PRODUCT}" "${SHARD_ID}" "${TILE_LIST}" "${tile_list_sha256}" "${expected_count}" "${OUT_DIR}" "${prediction_count}" "${uncertainty_count}" "${count_count}" "${base_seed}" "${mc_samples}" "${registry_sha256}" "${bundle_metadata_sha256}" "${started_at}" "${completed_at}" "${elapsed_seconds}" "${log_file}" "${EXPERIMENT}" > "${status_json}"
exit "${rc}"
