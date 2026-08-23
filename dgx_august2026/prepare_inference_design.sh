#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <bundle_root> <output_dir>" >&2
  exit 2
fi
BUNDLE_ROOT="$1"
OUTPUT_DIR="$2"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-${PROJECT_ROOT}/NAIP_Archive}"
PYTHON_BIN="${PYTHON_BIN:-/home/ncsucnr/miniforge3/envs/host-naip-sdm/bin/python}"
SMOKE_PER_STATE="${SMOKE_PER_STATE:-1}"
FULL_SHARD_SIZE="${FULL_SHARD_SIZE:-250}"
SMOKE_SHARD_SIZE="${SMOKE_SHARD_SIZE:-100}"

if [[ ! -s "${BUNDLE_ROOT}/COMPLETED" || ! -d "${ARCHIVE_ROOT}/states" ]]; then
  echo "Frozen DGX bundle or local NAIP archive is unavailable" >&2
  exit 3
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing pre-existing inference design root: ${OUTPUT_DIR}" >&2
  exit 4
fi
for value in "${SMOKE_PER_STATE}" "${FULL_SHARD_SIZE}" "${SMOKE_SHARD_SIZE}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Manifest and shard sizes must be positive integers" >&2
    exit 4
  fi
done
mkdir -p "${OUTPUT_DIR}"

for region in east_of_mississippi tanoak_ca_or; do
  manifest_dir="${OUTPUT_DIR}/manifests/${region}"
  "${PYTHON_BIN}" "${BUNDLE_ROOT}/build_archive_manifests.py" \
    --region "${region}" --archive-root "${ARCHIVE_ROOT}" \
    --smoke-per-state "${SMOKE_PER_STATE}" --output-dir "${manifest_dir}"
  for scope in full smoke; do
    if [[ "${scope}" == "full" ]]; then
      manifest="${manifest_dir}/${region}_all_tiles.txt"
      shard_size="${FULL_SHARD_SIZE}"
    else
      manifest="${manifest_dir}/${region}_smoke_tiles.txt"
      shard_size="${SMOKE_SHARD_SIZE}"
    fi
    "${PYTHON_BIN}" "${BUNDLE_ROOT}/make_tile_shards.py" \
      --manifest "${manifest}" --shard-size "${shard_size}" \
      --output-dir "${OUTPUT_DIR}/shards/${region}/${scope}"
  done
done

bundle_metadata_sha256="$(sha256sum "${BUNDLE_ROOT}/bundle_metadata.json" | awk '{print $1}')"
printf '{"status":"complete","archive_root":"%s","bundle_root":"%s","bundle_metadata_sha256":"%s","smoke_per_state":%s,"full_shard_size":%s,"smoke_shard_size":%s}\n' \
  "${ARCHIVE_ROOT}" "${BUNDLE_ROOT}" "${bundle_metadata_sha256}" \
  "${SMOKE_PER_STATE}" "${FULL_SHARD_SIZE}" "${SMOKE_SHARD_SIZE}" \
  > "${OUTPUT_DIR}/design_status.json"
printf 'complete\n' > "${OUTPUT_DIR}/COMPLETED"
