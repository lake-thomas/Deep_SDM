#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 9 ]]; then
  echo "Usage: $0 <host> <region> <tile_manifest> <production_registry> <bundle_root> <expected_tile_count> <june_reference_prediction> <deterministic_qa_summary> <uncertainty_qa_summary>" >&2
  exit 2
fi
HOST_SLUG="$1"
REGION="$2"
TILE_MANIFEST="$3"
PRODUCTION_REGISTRY="$4"
BUNDLE_ROOT="$5"
EXPECTED_INPUT_COUNT="$6"
JUNE_REFERENCE_PREDICTION="$7"
DETERMINISTIC_QA_SUMMARY="$8"
UNCERTAINTY_QA_SUMMARY="$9"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/rsstu_dsfas_gsv_naip/Host_NAIP_SDM}"
PYTHON_BIN="${PYTHON_BIN:-/home/ncsucnr/miniforge3/envs/host-naip-sdm/bin/python}"
bundle_metadata="${BUNDLE_ROOT}/bundle_metadata.json"
configured_output_root="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["inference_defaults"]["active_tiled_output_root"])' "${bundle_metadata}")"
ACTIVE_OUTPUT_ROOT="${ACTIVE_OUTPUT_ROOT:-${configured_output_root}}"
"${PYTHON_BIN}" -c 'import pathlib,sys; raise SystemExit(0 if pathlib.Path(sys.argv[1]).resolve()==pathlib.Path(sys.argv[2]).resolve() else 1)' \
  "${ACTIVE_OUTPUT_ROOT}" "${configured_output_root}" || {
  echo "ACTIVE_OUTPUT_ROOT differs from the frozen bundle root" >&2
  exit 3
}
MOSAIC_OUTPUT_ROOT="${MOSAIC_OUTPUT_ROOT:-${ACTIVE_OUTPUT_ROOT}}"
TARGET_RES="${TARGET_RES:-256}"
MOSAIC_ROOT="${MOSAIC_OUTPUT_ROOT}/mosaics/${REGION}/${HOST_SLUG}_epsg5070_zonesafe_august2026"
LOG_ROOT="${MOSAIC_ROOT}/logs"

for required in "${TILE_MANIFEST}" "${PRODUCTION_REGISTRY}" "${BUNDLE_ROOT}/COMPLETED" \
  "${JUNE_REFERENCE_PREDICTION}" "${DETERMINISTIC_QA_SUMMARY}" "${UNCERTAINTY_QA_SUMMARY}"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing mosaic input: ${required}" >&2
    exit 3
  fi
done
if [[ ! "${EXPECTED_INPUT_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Expected tile count must be a positive integer" >&2
  exit 3
fi
"${PYTHON_BIN}" "${BUNDLE_ROOT}/validate_dgx_bundle.py" \
  --bundle-root "${BUNDLE_ROOT}" --registry "${PRODUCTION_REGISTRY}" \
  --tile-manifest "${TILE_MANIFEST}" --host "${HOST_SLUG}" --region "${REGION}" \
  --expected-tile-count "${EXPECTED_INPUT_COUNT}" \
  --deterministic-summary "${DETERMINISTIC_QA_SUMMARY}" \
  --uncertainty-summary "${UNCERTAINTY_QA_SUMMARY}" \
  --active-output-root "${ACTIVE_OUTPUT_ROOT}" >/dev/null

mkdir -p "${MOSAIC_ROOT}/work" "${LOG_ROOT}"
export PROJ_NETWORK=OFF
export GTIFF_SRS_SOURCE="${GTIFF_SRS_SOURCE:-EPSG}"
export GDAL_CACHEMAX="${GDAL_CACHEMAX:-8192}"
export CPL_TMPDIR="${CPL_TMPDIR:-${MOSAIC_ROOT}/work/tmp}"
mkdir -p "${CPL_TMPDIR}"
tile_manifest_sha256="$(sha256sum "${TILE_MANIFEST}" | awk '{print $1}')"

for RELEASE_PRODUCT in prediction uncertainty count; do
  case "${RELEASE_PRODUCT}" in
    prediction) SOURCE_PRODUCT=deterministic; SOURCE_SUFFIX=prediction; srcnodata=-9999; dstnodata=-9999; resampling=bilinear ;;
    uncertainty) SOURCE_PRODUCT=uncertainty; SOURCE_SUFFIX=uncertainty; srcnodata=-9999; dstnodata=-9999; resampling=bilinear ;;
    count) SOURCE_PRODUCT=deterministic; SOURCE_SUFFIX=count; srcnodata=0; dstnodata=0; resampling=max ;;
  esac
  SOURCE_OUTPUT_ROOT="${ACTIVE_OUTPUT_ROOT}/${REGION}/${HOST_SLUG}/${SOURCE_PRODUCT}"
  INPUT_DESIGN="${MOSAIC_ROOT}/work/${RELEASE_PRODUCT}_inputs"
  input_metadata="${INPUT_DESIGN}/mosaic_input_metadata.json"
  if [[ -s "${input_metadata}" && -s "${INPUT_DESIGN}/COMPLETED" ]]; then
    "${PYTHON_BIN}" -c 'import json,pathlib,sys
d=json.load(open(sys.argv[1])); expected={"status":"complete","tile_manifest_sha256":sys.argv[2],"inference_output_root":str(pathlib.Path(sys.argv[3]).resolve()),"suffix":sys.argv[4],"expected_raster_count":int(sys.argv[5])}
raise SystemExit(0 if all(d.get(k)==v for k,v in expected.items()) else 1)' \
      "${input_metadata}" "${tile_manifest_sha256}" "${SOURCE_OUTPUT_ROOT}" \
      "${SOURCE_SUFFIX}" "${EXPECTED_INPUT_COUNT}"
  elif [[ -e "${INPUT_DESIGN}" ]]; then
    echo "Refusing incomplete pre-existing mosaic input design: ${INPUT_DESIGN}" >&2
    exit 4
  else
    "${PYTHON_BIN}" "${BUNDLE_ROOT}/august2026_prepare_mosaic_inputs.py" \
      --tile-manifest "${TILE_MANIFEST}" \
      --inference-output-root "${SOURCE_OUTPUT_ROOT}" \
      --suffix "${SOURCE_SUFFIX}" --output-dir "${INPUT_DESIGN}"
  fi

  out_tif="${MOSAIC_ROOT}/${HOST_SLUG}_${REGION}_${RELEASE_PRODUCT}_epsg5070_${TARGET_RES}m_zonesafe.tif"
  status_json="${LOG_ROOT}/${RELEASE_PRODUCT}_mosaic_status.json"
  if [[ -s "${status_json}" && -s "${out_tif}" ]]; then
    "${PYTHON_BIN}" -c 'import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[2]); h=hashlib.sha256();
with p.open("rb") as f:
  for chunk in iter(lambda:f.read(1048576),b""): h.update(chunk)
d=json.load(open(sys.argv[1])); expected={"status":"complete","host":sys.argv[3],"region":sys.argv[4],"release_product":sys.argv[5],"tile_manifest_sha256":sys.argv[6],"input_count":int(sys.argv[7]),"output":str(p.resolve()),"output_sha256":h.hexdigest()}
raise SystemExit(0 if all(d.get(k)==v for k,v in expected.items()) else 1)' \
      "${status_json}" "${out_tif}" "${HOST_SLUG}" "${REGION}" \
      "${RELEASE_PRODUCT}" "${tile_manifest_sha256}" "${EXPECTED_INPUT_COUNT}"
    continue
  fi
  if [[ -e "${out_tif}" || -e "${status_json}" ]]; then
    echo "Refusing incomplete or unverified mosaic output: ${out_tif}" >&2
    exit 5
  fi

  mapfile -t zone_lists < <(find "${INPUT_DESIGN}" -maxdepth 1 -name "utm*_${SOURCE_SUFFIX}_rasters.txt" | sort)
  if (( ${#zone_lists[@]} == 0 )); then
    echo "No zone manifests found for ${RELEASE_PRODUCT}" >&2
    exit 6
  fi
  vrt_args=()
  for zone_list in "${zone_lists[@]}"; do
    zone="$(basename "${zone_list}" | cut -d_ -f1)"
    zone_vrt="${MOSAIC_ROOT}/work/${zone}_${RELEASE_PRODUCT}.vrt"
    gdalbuildvrt -overwrite -input_file_list "${zone_list}" \
      -srcnodata "${srcnodata}" -vrtnodata "${dstnodata}" "${zone_vrt}"
    expected_zone="$(wc -l < "${zone_list}" | tr -d ' ')"
    found_zone="$(grep -c '<SourceFilename' "${zone_vrt}")"
    if [[ "${expected_zone}" != "${found_zone}" ]]; then
      echo "VRT source count mismatch for ${zone_vrt}" >&2
      exit 6
    fi
    vrt_args+=("${zone_vrt}")
  done
  partial_tif="${out_tif%.tif}.partial.tif"
  if [[ -e "${partial_tif}" ]]; then
    echo "Refusing stale partial mosaic: ${partial_tif}" >&2
    exit 7
  fi
  gdalwarp -multi -wo "NUM_THREADS=${GDAL_THREADS:-8}" -wm 8192 \
    -t_srs EPSG:5070 -tr "${TARGET_RES}" "${TARGET_RES}" -tap -r "${resampling}" \
    -srcnodata "${srcnodata}" -dstnodata "${dstnodata}" \
    -co TILED=YES -co COMPRESS=DEFLATE -co PREDICTOR=2 -co BIGTIFF=YES \
    "${vrt_args[@]}" "${partial_tif}"
  mv "${partial_tif}" "${out_tif}"
  gdalinfo -json -stats "${out_tif}" > "${out_tif%.tif}_gdalinfo.json"
  output_sha256="$(sha256sum "${out_tif}" | awk '{print $1}')"
  printf '{"status":"complete","host":"%s","region":"%s","release_product":"%s","source_suffix":"%s","source_output_root":"%s","tile_manifest":"%s","tile_manifest_sha256":"%s","input_count":%s,"target_resolution_m":%s,"output":"%s","output_sha256":"%s"}\n' \
    "${HOST_SLUG}" "${REGION}" "${RELEASE_PRODUCT}" "${SOURCE_SUFFIX}" \
    "${SOURCE_OUTPUT_ROOT}" "${TILE_MANIFEST}" "${tile_manifest_sha256}" \
    "${EXPECTED_INPUT_COUNT}" "${TARGET_RES}" "${out_tif}" "${output_sha256}" > "${status_json}"
done

stem="${HOST_SLUG}_${REGION}"
prediction="${MOSAIC_ROOT}/${stem}_prediction_epsg5070_${TARGET_RES}m_zonesafe.tif"
uncertainty="${MOSAIC_ROOT}/${stem}_uncertainty_epsg5070_${TARGET_RES}m_zonesafe.tif"
count="${MOSAIC_ROOT}/${stem}_count_epsg5070_${TARGET_RES}m_zonesafe.tif"
validation="${LOG_ROOT}/automated_mosaic_validation.json"
if [[ -s "${validation}" ]]; then
  "${PYTHON_BIN}" -c 'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d.get("status")=="complete" and d.get("production_ready_for_visual_review") else 1)' "${validation}"
  exit 0
fi
parity_dir="${LOG_ROOT}/native_mosaic_parity"
quickview_dir="${MOSAIC_ROOT}/quickviews"
if [[ -e "${parity_dir}" || -e "${quickview_dir}" ]]; then
  echo "Refusing partial post-mosaic QA directories" >&2
  exit 8
fi
"${PYTHON_BIN}" "${BUNDLE_ROOT}/august2026_native_mosaic_parity.py" \
  --tile-manifest "${TILE_MANIFEST}" \
  --prediction-output-root "${ACTIVE_OUTPUT_ROOT}/${REGION}/${HOST_SLUG}/deterministic" \
  --uncertainty-output-root "${ACTIVE_OUTPUT_ROOT}/${REGION}/${HOST_SLUG}/uncertainty" \
  --prediction-mosaic "${prediction}" --uncertainty-mosaic "${uncertainty}" \
  --host "${HOST_SLUG}" --region "${REGION}" --output-dir "${parity_dir}"
"${PYTHON_BIN}" "${BUNDLE_ROOT}/august2026_render_fixed_quickviews.py" \
  --prediction "${prediction}" --uncertainty "${uncertainty}" --count "${count}" \
  --reference-prediction "${JUNE_REFERENCE_PREDICTION}" \
  --host "${HOST_SLUG}" --region "${REGION}" --output-dir "${quickview_dir}"
"${PYTHON_BIN}" "${BUNDLE_ROOT}/august2026_validate_mosaics.py" \
  --prediction "${prediction}" --uncertainty "${uncertainty}" --count "${count}" \
  --quickview-summary "${quickview_dir}/quickview_summary.json" \
  --parity-summary "${parity_dir}/summary.json" \
  --expected-input-count "${EXPECTED_INPUT_COUNT}" --target-resolution "${TARGET_RES}" \
  --output "${validation}"
