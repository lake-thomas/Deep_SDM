#!/usr/bin/env bash
#SBATCH --job-name=aug26-mosaic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=18:00:00
#SBATCH --output=/rsstu/users/r/rkmeente/DSFAS_GSV_NAIP/Host_NAIP_SDM/Outputs_Host_NAIP_SDMs/production_model_inference_august2026/_slurm_logs/%x-%A_%a.out
#SBATCH --error=/rsstu/users/r/rkmeente/DSFAS_GSV_NAIP/Host_NAIP_SDM/Outputs_Host_NAIP_SDMs/production_model_inference_august2026/_slurm_logs/%x-%A_%a.err

set -euo pipefail

HOST_SLUG="${HOST_SLUG:?Set HOST_SLUG}"
REGION="${REGION:?Set REGION}"
RELEASE_PRODUCT="${RELEASE_PRODUCT:-}"
SOURCE_SUFFIX="${SOURCE_SUFFIX:-}"
SOURCE_OUTPUT_ROOT="${SOURCE_OUTPUT_ROOT:-}"
TILE_MANIFEST="${TILE_MANIFEST:?Set TILE_MANIFEST}"
BUNDLE_ROOT="${BUNDLE_ROOT:?Set BUNDLE_ROOT}"
if [[ -z "${RELEASE_PRODUCT}" ]]; then
  case "${SLURM_ARRAY_TASK_ID:?Set RELEASE_PRODUCT or run as a three-task Slurm array}" in
    0) RELEASE_PRODUCT=prediction; SOURCE_PRODUCT=deterministic; SOURCE_SUFFIX=prediction ;;
    1) RELEASE_PRODUCT=uncertainty; SOURCE_PRODUCT=uncertainty; SOURCE_SUFFIX=uncertainty ;;
    2) RELEASE_PRODUCT=count; SOURCE_PRODUCT=deterministic; SOURCE_SUFFIX=count ;;
    *) exit 2 ;;
  esac
  ACTIVE_OUTPUT_ROOT="${ACTIVE_OUTPUT_ROOT:-/share/rkmeente/talake2/host_sdm_august2026/production_model_inference_august2026}"
  SOURCE_OUTPUT_ROOT="${ACTIVE_OUTPUT_ROOT}/${REGION}/${HOST_SLUG}/${SOURCE_PRODUCT}"
fi
if [[ "${RELEASE_PRODUCT}" != "prediction" && "${RELEASE_PRODUCT}" != "uncertainty" && "${RELEASE_PRODUCT}" != "count" ]]; then
  exit 2
fi
if [[ "${SOURCE_SUFFIX}" != "prediction" && "${SOURCE_SUFFIX}" != "uncertainty" && "${SOURCE_SUFFIX}" != "count" ]]; then
  exit 2
fi
if [[ -z "${SOURCE_OUTPUT_ROOT}" ]]; then
  exit 2
fi

HPC_ROOT="${HPC_ROOT:-/rsstu/users/r/rkmeente/DSFAS_GSV_NAIP/Host_NAIP_SDM}"
OUTPUT_ROOT="${MOSAIC_OUTPUT_ROOT:-${HPC_ROOT}/Outputs_Host_NAIP_SDMs/production_model_inference_august2026}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-/usr/local/usrapps/rkmeente/talake2/host-naip-sdm}"
TARGET_RES="${TARGET_RES:-256}"
MOSAIC_ROOT="${OUTPUT_ROOT}/mosaics/${REGION}/${HOST_SLUG}_epsg5070_zonesafe_20260813"
INPUT_DESIGN="${MOSAIC_ROOT}/work/${RELEASE_PRODUCT}_inputs"
LOG_ROOT="${MOSAIC_ROOT}/logs"
mkdir -p "${MOSAIC_ROOT}/work" "${LOG_ROOT}" "${OUTPUT_ROOT}/_slurm_logs"

export PROJ_NETWORK=OFF
export GTIFF_SRS_SOURCE="${GTIFF_SRS_SOURCE:-EPSG}"
export PROJ_LIB="${PROJ_LIB:-${CONDA_ENV_NAME}/share/proj}"
export PROJ_DATA="${PROJ_DATA:-${CONDA_ENV_NAME}/share/proj}"
export GDAL_DATA="${GDAL_DATA:-${CONDA_ENV_NAME}/share/gdal}"
export GDAL_CACHEMAX="${GDAL_CACHEMAX:-8192}"
export CPL_TMPDIR="${CPL_TMPDIR:-${MOSAIC_ROOT}/work/tmp_${RELEASE_PRODUCT}_${SLURM_JOB_ID:-manual}}"
mkdir -p "${CPL_TMPDIR}"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

python "${BUNDLE_ROOT}/august2026_prepare_mosaic_inputs.py" \
  --tile-manifest "${TILE_MANIFEST}" \
  --inference-output-root "${SOURCE_OUTPUT_ROOT}" \
  --suffix "${SOURCE_SUFFIX}" \
  --output-dir "${INPUT_DESIGN}"

metadata="${INPUT_DESIGN}/mosaic_input_metadata.json"
expected_total="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["expected_raster_count"])' "${metadata}")"
mapfile -t zone_lists < <(find "${INPUT_DESIGN}" -maxdepth 1 -name "utm*_${SOURCE_SUFFIX}_rasters.txt" | sort)
if (( ${#zone_lists[@]} == 0 )); then exit 3; fi
vrt_args=()
for zone_list in "${zone_lists[@]}"; do
  zone="$(basename "${zone_list}" | cut -d_ -f1)"
  zone_vrt="${MOSAIC_ROOT}/work/${zone}_${RELEASE_PRODUCT}.vrt"
  if [[ "${RELEASE_PRODUCT}" == "count" ]]; then
    gdalbuildvrt -overwrite -input_file_list "${zone_list}" -srcnodata 0 -vrtnodata 0 "${zone_vrt}"
  else
    gdalbuildvrt -overwrite -input_file_list "${zone_list}" -srcnodata -9999 -vrtnodata -9999 "${zone_vrt}"
  fi
  expected_zone="$(wc -l < "${zone_list}" | tr -d ' ')"
  found_zone="$(grep -c '<SourceFilename' "${zone_vrt}")"
  if [[ "${expected_zone}" != "${found_zone}" ]]; then exit 4; fi
  vrt_args+=("${zone_vrt}")
done

out_tif="${MOSAIC_ROOT}/${HOST_SLUG}_${REGION}_${RELEASE_PRODUCT}_epsg5070_${TARGET_RES}m_zonesafe.tif"
if [[ "${RELEASE_PRODUCT}" == "count" ]]; then
  resampling=max
  srcnodata=0
  dstnodata=0
else
  resampling=bilinear
  srcnodata=-9999
  dstnodata=-9999
fi
gdalwarp -overwrite -multi -wo "NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}" -wm 8192 \
  -t_srs EPSG:5070 -tr "${TARGET_RES}" "${TARGET_RES}" -tap -r "${resampling}" \
  -srcnodata "${srcnodata}" -dstnodata "${dstnodata}" \
  -co TILED=YES -co COMPRESS=DEFLATE -co PREDICTOR=2 -co BIGTIFF=YES \
  "${vrt_args[@]}" "${out_tif}"
gdalinfo -json -stats "${out_tif}" > "${out_tif%.tif}_gdalinfo.json"
printf '{"status":"complete","host":"%s","region":"%s","release_product":"%s","source_suffix":"%s","source_output_root":"%s","tile_manifest":"%s","input_count":%s,"target_resolution_m":%s,"output":"%s"}\n' \
  "${HOST_SLUG}" "${REGION}" "${RELEASE_PRODUCT}" "${SOURCE_SUFFIX}" \
  "${SOURCE_OUTPUT_ROOT}" "${TILE_MANIFEST}" "${expected_total}" "${TARGET_RES}" "${out_tif}" \
  > "${LOG_ROOT}/${RELEASE_PRODUCT}_mosaic_status.json"
