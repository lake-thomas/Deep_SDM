#!/usr/bin/env bash
#SBATCH --job-name=aug26-mapqa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/rsstu/users/r/rkmeente/DSFAS_GSV_NAIP/Host_NAIP_SDM/Outputs_Host_NAIP_SDMs/production_model_inference_august2026/_slurm_logs/%x-%j.out
#SBATCH --error=/rsstu/users/r/rkmeente/DSFAS_GSV_NAIP/Host_NAIP_SDM/Outputs_Host_NAIP_SDMs/production_model_inference_august2026/_slurm_logs/%x-%j.err

set -euo pipefail
HOST_SLUG="${HOST_SLUG:?Set HOST_SLUG}"
REGION="${REGION:?Set REGION}"
EXPECTED_INPUT_COUNT="${EXPECTED_INPUT_COUNT:?Set EXPECTED_INPUT_COUNT}"
TILE_MANIFEST="${TILE_MANIFEST:?Set TILE_MANIFEST}"
JUNE_REFERENCE_PREDICTION="${JUNE_REFERENCE_PREDICTION:?Set JUNE_REFERENCE_PREDICTION}"
BUNDLE_ROOT="${BUNDLE_ROOT:?Set BUNDLE_ROOT}"
HPC_ROOT="${HPC_ROOT:-/rsstu/users/r/rkmeente/DSFAS_GSV_NAIP/Host_NAIP_SDM}"
OUTPUT_ROOT="${MOSAIC_OUTPUT_ROOT:-${HPC_ROOT}/Outputs_Host_NAIP_SDMs/production_model_inference_august2026}"
ACTIVE_OUTPUT_ROOT="${ACTIVE_OUTPUT_ROOT:-/share/rkmeente/talake2/host_sdm_august2026/production_model_inference_august2026}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-/usr/local/usrapps/rkmeente/talake2/host-naip-sdm}"
TARGET_RES="${TARGET_RES:-256}"
MOSAIC_ROOT="${OUTPUT_ROOT}/mosaics/${REGION}/${HOST_SLUG}_epsg5070_zonesafe_20260813"
stem="${HOST_SLUG}_${REGION}"
prediction="${MOSAIC_ROOT}/${stem}_prediction_epsg5070_${TARGET_RES}m_zonesafe.tif"
uncertainty="${MOSAIC_ROOT}/${stem}_uncertainty_epsg5070_${TARGET_RES}m_zonesafe.tif"
count="${MOSAIC_ROOT}/${stem}_count_epsg5070_${TARGET_RES}m_zonesafe.tif"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u
python "${BUNDLE_ROOT}/validate_submission_gate.py" bundle \
  --bundle-root "${BUNDLE_ROOT}" >/dev/null
python "${BUNDLE_ROOT}/august2026_native_mosaic_parity.py" \
  --tile-manifest "${TILE_MANIFEST}" \
  --prediction-output-root "${ACTIVE_OUTPUT_ROOT}/${REGION}/${HOST_SLUG}/deterministic" \
  --uncertainty-output-root "${ACTIVE_OUTPUT_ROOT}/${REGION}/${HOST_SLUG}/uncertainty" \
  --prediction-mosaic "${prediction}" --uncertainty-mosaic "${uncertainty}" \
  --host "${HOST_SLUG}" --region "${REGION}" \
  --output-dir "${MOSAIC_ROOT}/logs/native_mosaic_parity"
python "${BUNDLE_ROOT}/august2026_render_fixed_quickviews.py" \
  --prediction "${prediction}" --uncertainty "${uncertainty}" --count "${count}" \
  --reference-prediction "${JUNE_REFERENCE_PREDICTION}" \
  --host "${HOST_SLUG}" --region "${REGION}" --output-dir "${MOSAIC_ROOT}/quickviews"
python "${BUNDLE_ROOT}/august2026_validate_mosaics.py" \
  --prediction "${prediction}" --uncertainty "${uncertainty}" --count "${count}" \
  --quickview-summary "${MOSAIC_ROOT}/quickviews/quickview_summary.json" \
  --parity-summary "${MOSAIC_ROOT}/logs/native_mosaic_parity/summary.json" \
  --expected-input-count "${EXPECTED_INPUT_COUNT}" --target-resolution "${TARGET_RES}" \
  --output "${MOSAIC_ROOT}/logs/automated_mosaic_validation.json"
