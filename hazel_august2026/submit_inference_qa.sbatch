#!/usr/bin/env bash
#SBATCH --job-name=aug26-tileqa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/rsstu/users/r/rkmeente/DSFAS_GSV_NAIP/Host_NAIP_SDM/Outputs_Host_NAIP_SDMs/production_model_inference_august2026/_slurm_logs/%x-%j.out
#SBATCH --error=/rsstu/users/r/rkmeente/DSFAS_GSV_NAIP/Host_NAIP_SDM/Outputs_Host_NAIP_SDMs/production_model_inference_august2026/_slurm_logs/%x-%j.err

set -euo pipefail

HOST_SLUG="${HOST_SLUG:?Set HOST_SLUG}"
REGION="${REGION:?Set REGION}"
PRODUCT="${PRODUCT:?Set PRODUCT}"
TILE_MANIFEST="${TILE_MANIFEST:?Set TILE_MANIFEST}"
PRODUCTION_REGISTRY="${PRODUCTION_REGISTRY:?Set PRODUCTION_REGISTRY}"
BUNDLE_ROOT="${BUNDLE_ROOT:?Set BUNDLE_ROOT}"
if [[ "${PRODUCT}" != "deterministic" && "${PRODUCT}" != "uncertainty" ]]; then
  exit 2
fi
HPC_ROOT="${HPC_ROOT:-/rsstu/users/r/rkmeente/DSFAS_GSV_NAIP/Host_NAIP_SDM}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HPC_ROOT}/Outputs_Host_NAIP_SDMs/production_model_inference_august2026}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-/usr/local/usrapps/rkmeente/talake2/host-naip-sdm}"
QA_ROOT="${OUTPUT_ROOT}/_run_metadata/qa/${REGION}/${HOST_SLUG}/${PRODUCT}"
SOURCE_ROOT="${OUTPUT_ROOT}/${REGION}/${HOST_SLUG}/${PRODUCT}"
candidate_env="${OUTPUT_ROOT}/_run_metadata/status/${REGION}/${HOST_SLUG}/${PRODUCT}/qa_candidate.env"

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u
python "${BUNDLE_ROOT}/resolve_production_candidate.py" \
  --registry "${PRODUCTION_REGISTRY}" --host "${HOST_SLUG}" \
  --hpc-root "${HPC_ROOT}" > "${candidate_env}"
source "${candidate_env}"

cmd=(
  python "${BUNDLE_ROOT}/august2026_inference_output_qa.py"
  --tile-manifest "${TILE_MANIFEST}"
  --output-root "${SOURCE_ROOT}"
  --host "${HOST_SLUG}"
  --region "${REGION}"
  --expected-score-space "${SCORE_SPACE}"
  --product "${PRODUCT}"
  --output-dir "${QA_ROOT}"
)
if [[ -n "${CALIBRATOR_SHA256}" ]]; then
  cmd+=(--expected-calibrator-sha256 "${CALIBRATOR_SHA256}")
fi
"${cmd[@]}"
