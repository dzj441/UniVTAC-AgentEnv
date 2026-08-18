#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"
CONDA_BIN="${UNIVTAC_CONDA_BIN:-${PROJECT_ROOT}/miniconda3/bin/conda}"
SAM_ENV="${UNIVTAC_SAM3_ENV:-${PROJECT_ROOT}/miniconda3/envs/univtac-sam3}"
DEPTH_ENV="${UNIVTAC_UNIDEPTH_V2_ENV:-${PROJECT_ROOT}/miniconda3/envs/univtac-unidepth-v2}"
SIM_ENV="${UNIVTAC_SIM_ENV:-${PROJECT_ROOT}/miniconda3/envs/UniVTAC}"
DATA_ROOT="${UNIVTAC_SEMANTIC_DATA_ROOT:-/inspire/qb-ilm/project/semantic-visual-tokenizer/public/dzj/univtac_semantic_tools}"
TOKEN_FILE="${UNIVTAC_HF_TOKEN_FILE:-${REPO_ROOT}/hf_token}"
LOCK_FILE="${REPO_ROOT}/semantic_tools/versions.json"
MODE=all

usage() {
  echo "usage: $0 [--envs-only|--models-only]"
}

while (($#)); do
  case "$1" in
    --envs-only) MODE=envs; shift ;;
    --models-only) MODE=models; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Parent Miniconda was not found: ${CONDA_BIN}" >&2
  exit 1
fi

# Every network subprocess below inherits this proxy-free environment. This is
# intentional for the current cluster and avoids silently routing multi-GB
# checkpoints through the notebook/VPN proxy.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

read_lock() {
  "${PROJECT_ROOT}/miniconda3/bin/python" -c \
    'import json,sys; value=json.load(open(sys.argv[1])); print(value'"$2"')' \
    "${LOCK_FILE}"
}

TORCH_VERSION="$(read_lock ignored '["torch"]')"
TORCHVISION_VERSION="$(read_lock ignored '["torchvision"]')"
SAM_SOURCE="$(read_lock ignored '["sam3"]["source_url"]')"
SAM_COMMIT="$(read_lock ignored '["sam3"]["source_commit"]')"
SAM_REVISION="$(read_lock ignored '["sam3"]["model_revision"]')"
DEPTH_SOURCE="$(read_lock ignored '["unidepth_v2"]["source_url"]')"
DEPTH_COMMIT="$(read_lock ignored '["unidepth_v2"]["source_commit"]')"
DEPTH_REVISION="$(read_lock ignored '["unidepth_v2"]["model_revision"]')"

create_envs() {
  # Some cluster images omit the small GLU runtime normally supplied by an apt
  # package.  Keep this dependency inside the existing simulator Conda prefix
  # instead of changing the host OS or the shared model disk.
  if [[ -x "${SIM_ENV}/bin/python" && ! -e "${SIM_ENV}/lib/libGLU.so.1" ]]; then
    "${CONDA_BIN}" install --prefix "${SIM_ENV}" --yes libglu
  fi
  if [[ ! -x "${SAM_ENV}/bin/python" ]]; then
    "${CONDA_BIN}" create --prefix "${SAM_ENV}" --yes python=3.12 pip
  fi
  if [[ ! -x "${DEPTH_ENV}/bin/python" ]]; then
    "${CONDA_BIN}" create --prefix "${DEPTH_ENV}" --yes python=3.12 pip
  fi

  "${SAM_ENV}/bin/python" -m pip install \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    Pillow einops pycocotools 'numpy>=1.26,<2' 'timm>=1.0.17' tqdm \
    'ftfy==6.1.1' regex 'iopath>=0.1.10' typing_extensions huggingface_hub \
    psutil 'setuptools<81'
  if [[ ! -s "${SAM_ENV}/.univtac_sam3_commit" ]] || \
     [[ "$(<"${SAM_ENV}/.univtac_sam3_commit")" != "${SAM_COMMIT}" ]]; then
    "${SAM_ENV}/bin/python" -m pip install --no-deps --no-build-isolation \
      "https://codeload.github.com/facebookresearch/sam3/tar.gz/${SAM_COMMIT}"
    echo "${SAM_COMMIT}" >"${SAM_ENV}/.univtac_sam3_commit"
  fi

  "${DEPTH_ENV}/bin/python" -m pip install \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    'numpy>=2' 'Pillow>=10.2' 'einops>=0.7' 'huggingface-hub>=0.22' \
    timm scipy opencv-python matplotlib wandb 'setuptools<81'
  if [[ ! -s "${DEPTH_ENV}/.univtac_unidepth_v2_commit" ]] || \
     [[ "$(<"${DEPTH_ENV}/.univtac_unidepth_v2_commit")" != "${DEPTH_COMMIT}" ]]; then
    "${DEPTH_ENV}/bin/python" -m pip install --no-deps --no-build-isolation \
      "https://codeload.github.com/lpiccinelli-eth/UniDepth/tar.gz/${DEPTH_COMMIT}"
    echo "${DEPTH_COMMIT}" >"${DEPTH_ENV}/.univtac_unidepth_v2_commit"
  fi

  "${SAM_ENV}/bin/python" - <<'PY'
import torch, torchvision, sam3
print("SAM3 env:", torch.__version__, torchvision.__version__, sam3.__version__)
assert torch.cuda.is_available(), "SAM3 environment cannot see CUDA"
PY
  "${DEPTH_ENV}/bin/python" - <<'PY'
import torch, torchvision
from unidepth.models import UniDepthV2
print("UniDepth V2 env:", torch.__version__, torchvision.__version__, UniDepthV2.__name__)
assert torch.cuda.is_available(), "UniDepth V2 environment cannot see CUDA"
PY
}

download_models() {
  if [[ ! -s "${TOKEN_FILE}" ]]; then
    echo "Hugging Face token file is missing or empty: ${TOKEN_FILE}" >&2
    exit 1
  fi
  chmod 600 "${TOKEN_FILE}"
  mkdir -p "${DATA_ROOT}/models/sam3" \
    "${DATA_ROOT}/models/unidepth-v2-vitl14" "${DATA_ROOT}/hf"
  HF_HOME="${DATA_ROOT}/hf" "${SAM_ENV}/bin/python" - \
    "${TOKEN_FILE}" "${DATA_ROOT}" "${SAM_REVISION}" "${DEPTH_REVISION}" <<'PY'
from pathlib import Path
import sys
from huggingface_hub import hf_hub_download, snapshot_download

token = Path(sys.argv[1]).read_text(encoding="utf-8").strip()
root = Path(sys.argv[2])
sam_revision, depth_revision = sys.argv[3], sys.argv[4]
hf_hub_download(
    repo_id="facebook/sam3",
    filename="sam3.pt",
    revision=sam_revision,
    token=token,
    local_dir=root / "models" / "sam3",
)
snapshot_download(
    repo_id="lpiccinelli/unidepth-v2-vitl14",
    revision=depth_revision,
    token=token,
    local_dir=root / "models" / "unidepth-v2-vitl14",
    allow_patterns=["config.json", "model.safetensors"],
)
PY

  "${PROJECT_ROOT}/miniconda3/bin/python" - \
    "${DATA_ROOT}" "${SAM_REVISION}" "${DEPTH_REVISION}" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
sam_revision, depth_revision = sys.argv[2], sys.argv[3]
models = {
    "sam3": (root / "models" / "sam3" / "sam3.pt", sam_revision),
    "unidepth_v2": (
        root / "models" / "unidepth-v2-vitl14" / "model.safetensors",
        depth_revision,
    ),
}
payload = {
    "schema_version": "univtac.semantic_models.installation.v1",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "models": {},
}
for name, (path, revision) in models.items():
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"Downloaded model is missing or empty: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    payload["models"][name] = {
        "revision": revision,
        "artifact": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }
manifest = root / "installation_manifest.json"
temporary = manifest.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(manifest)
PY

  local link="${REPO_ROOT}/.semantic_models"
  if [[ -e "${link}" && ! -L "${link}" ]]; then
    echo "Refusing to replace non-symlink model path: ${link}" >&2
    exit 1
  fi
  if [[ -L "${link}" && "$(readlink -f "${link}")" != "$(readlink -f "${DATA_ROOT}/models")" ]]; then
    echo "Existing model symlink points elsewhere: ${link}" >&2
    exit 1
  fi
  if [[ ! -L "${link}" ]]; then
    ln -s "${DATA_ROOT}/models" "${link}"
  fi
}

case "${MODE}" in
  envs) create_envs ;;
  models) download_models ;;
  all) create_envs; download_models ;;
esac

echo "Semantic tools setup complete."
echo "SAM3 env: ${SAM_ENV}"
echo "UniDepth V2 env: ${DEPTH_ENV}"
echo "Models: ${DATA_ROOT}/models (repo link: ${REPO_ROOT}/.semantic_models)"
