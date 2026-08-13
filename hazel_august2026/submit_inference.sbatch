#!/usr/bin/env bash
#SBATCH --job-name=aug26-sdm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/rsstu/users/r/rkmeente/DSFAS_GSV_NAIP/Host_NAIP_SDM/Outputs_Host_NAIP_SDMs/production_model_inference_august2026/_slurm_logs/%x-%j.out
#SBATCH --error=/rsstu/users/r/rkmeente/DSFAS_GSV_NAIP/Host_NAIP_SDM/Outputs_Host_NAIP_SDMs/production_model_inference_august2026/_slurm_logs/%x-%j.err

set -euo pipefail

HOST_SLUG="${HOST_SLUG:?Set HOST_SLUG}"
REGION="${REGION:?Set REGION}"
PRODUCT="${PRODUCT:?Set PRODUCT to deterministic or uncertainty}"
TILE_LIST="${TILE_LIST:?Set TILE_LIST}"
PRODUCTION_REGISTRY="${PRODUCTION_REGISTRY:?Set PRODUCTION_REGISTRY}"
SHARD_ID="${SHARD_ID:-manual}"
if [[ ! -s "${TILE_LIST}" ]]; then
  echo "Missing tile shard: ${TILE_LIST}" >&2
  exit 2
fi
if [[ "${PRODUCT}" != "deterministic" && "${PRODUCT}" != "uncertainty" ]]; then
  echo "Unknown PRODUCT=${PRODUCT}" >&2
  exit 2
fi

HPC_ROOT="${HPC_ROOT:-/rsstu/users/r/rkmeente/DSFAS_GSV_NAIP/Host_NAIP_SDM}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HPC_ROOT}/Outputs_Host_NAIP_SDMs/production_model_inference_august2026}"
BUNDLE_ROOT="${BUNDLE_ROOT:?Set BUNDLE_ROOT to the staged frozen bundle}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-/usr/local/usrapps/rkmeente/talake2/host-naip-sdm}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CHIP_SIZE="${CHIP_SIZE:-256}"
STRIDE="${STRIDE:-64}"
BATCH_SIZE="${BATCH_SIZE:-512}"
MC_SAMPLES=1
if [[ "${PRODUCT}" == "uncertainty" ]]; then
  MC_SAMPLES="${UNCERTAINTY_MC_SAMPLES:-10}"
fi

RUN_METADATA="${OUTPUT_ROOT}/_run_metadata"
OUT_DIR="${OUTPUT_ROOT}/${REGION}/${HOST_SLUG}/${PRODUCT}/shard_${SHARD_ID}"
STATUS_DIR="${RUN_METADATA}/status/${REGION}/${HOST_SLUG}/${PRODUCT}"
LOG_DIR="${RUN_METADATA}/logs/${REGION}/${HOST_SLUG}/${PRODUCT}"
mkdir -p "${OUT_DIR}" "${STATUS_DIR}" "${LOG_DIR}" "${OUTPUT_ROOT}/_slurm_logs"

export PROJ_LIB="${PROJ_LIB:-${CONDA_ENV_NAME}/share/proj}"
export PROJ_DATA="${PROJ_DATA:-${CONDA_ENV_NAME}/share/proj}"
export GDAL_DATA="${GDAL_DATA:-${CONDA_ENV_NAME}/share/gdal}"
export GTIFF_SRS_SOURCE="${GTIFF_SRS_SOURCE:-EPSG}"
export PROJ_NETWORK=OFF
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

candidate_env="${STATUS_DIR}/shard_${SHARD_ID}_candidate.env"
"${PYTHON_BIN}" "${BUNDLE_ROOT}/resolve_production_candidate.py" \
  --registry "${PRODUCTION_REGISTRY}" --host "${HOST_SLUG}" \
  --hpc-root "${HPC_ROOT}" > "${candidate_env}"
source "${candidate_env}"

cmd=(
  "${PYTHON_BIN}" "${BUNDLE_ROOT}/code_snapshot/inference_multimodal_production.py"
  --config "${CONFIG_PATH}"
  --checkpoint "${CHECKPOINT_PATH}"
  --strict-checkpoint
  --tile-list "${TILE_LIST}"
  --output-dir "${OUT_DIR}"
  --worldclim-dir "${HPC_ROOT}/Env_Data/Worldclim"
  --ghm-raster "${HPC_ROOT}/Env_Data/Global_Human_Modification/gHM_WGS84.tif"
  --dem-raster "${HPC_ROOT}/Env_Data/Topography_3DEP/mosaic/dem_3dep_13_epsg5070_30m.tif"
  --slope-raster "${HPC_ROOT}/Env_Data/Topography_3DEP/mosaic/slope_degrees_3dep_13_epsg5070_30m.tif"
  --northness-raster "${HPC_ROOT}/Env_Data/Topography_3DEP/mosaic/northness_3dep_13_epsg5070_30m.tif"
  --eastness-raster "${HPC_ROOT}/Env_Data/Topography_3DEP/mosaic/eastness_3dep_13_epsg5070_30m.tif"
  --topo-normalization-stats "${HPC_ROOT}/Env_Data/Topography_3DEP/mosaic/topo_norm_stats_3dep_30m.json"
  --chip-size "${CHIP_SIZE}"
  --stride "${STRIDE}"
  --batch-size "${BATCH_SIZE}"
  --mc-samples "${MC_SAMPLES}"
  --env-clip-conus
  --fast-topo-scalars
  --blend-window hann
  --blend-edge-floor 0.05
  --skip-existing
)
if [[ -n "${CALIBRATOR_PATH}" ]]; then
  cmd+=(--platt-calibrator "${CALIBRATOR_PATH}")
fi

status_json="${STATUS_DIR}/shard_${SHARD_ID}.json"
log_file="${LOG_DIR}/shard_${SHARD_ID}.log"
started_at="$(date -Is)"
printf '{"status":"running","host":"%s","region":"%s","product":"%s","shard_id":"%s","started_at":"%s"}\n' \
  "${HOST_SLUG}" "${REGION}" "${PRODUCT}" "${SHARD_ID}" "${started_at}" > "${status_json}"
printf 'command=%q ' "${cmd[@]}" > "${log_file}"
printf '\n' >> "${log_file}"
set +e
"${cmd[@]}" >> "${log_file}" 2>&1
rc="$?"
set -e
prediction_count="$(find "${OUT_DIR}" -maxdepth 1 -name '*_prediction.tif' | wc -l | tr -d ' ')"
uncertainty_count="$(find "${OUT_DIR}" -maxdepth 1 -name '*_uncertainty.tif' | wc -l | tr -d ' ')"
count_count="$(find "${OUT_DIR}" -maxdepth 1 -name '*_count.tif' | wc -l | tr -d ' ')"
completed_at="$(date -Is)"
status=completed
if [[ "${rc}" -ne 0 ]]; then status=failed; fi
printf '{"status":"%s","returncode":%s,"host":"%s","region":"%s","product":"%s","shard_id":"%s","tile_list":"%s","output_dir":"%s","prediction_count":%s,"uncertainty_count":%s,"count_count":%s,"started_at":"%s","completed_at":"%s","log_file":"%s","experiment":"%s"}\n' \
  "${status}" "${rc}" "${HOST_SLUG}" "${REGION}" "${PRODUCT}" "${SHARD_ID}" \
  "${TILE_LIST}" "${OUT_DIR}" "${prediction_count}" "${uncertainty_count}" "${count_count}" \
  "${started_at}" "${completed_at}" "${log_file}" "${EXPERIMENT}" > "${status_json}"
exit "${rc}"
