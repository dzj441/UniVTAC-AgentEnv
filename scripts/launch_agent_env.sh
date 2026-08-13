#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"

PYTHON_BIN="${UNIVTAC_PYTHON:-${PROJECT_ROOT}/miniconda3/envs/UniVTAC/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "UniVTAC Python is not executable: ${PYTHON_BIN}" >&2
  echo "Set UNIVTAC_PYTHON to the configured environment's Python." >&2
  exit 1
fi

HAS_LEVEL=false
for argument in "$@"; do
  case "${argument}" in
    --level|--level=*) HAS_LEVEL=true ;;
  esac
done
if [[ "${HAS_LEVEL}" != true ]]; then
  echo "usage: $0 --level {1|2|3} [--device cuda:0] [--run-dir PATH]" >&2
  exit 2
fi

# Isaac/Kit needs a userspace NVIDIA rendering stack whose version exactly
# matches the host kernel driver.  The host's injected bundle is deliberately
# excluded: this project uses the independently prepared, reproducible bundle
# on the shared data disk.
DEFAULT_NVIDIA_RENDER_ROOT="/inspire/qb-ilm/project/semantic-visual-tokenizer/public/dzj/robomme_runtime/nvidia/570.124.06"
NVIDIA_RENDER_ROOT="${UNIVTAC_NVIDIA_RENDER_ROOT:-${DEFAULT_NVIDIA_RENDER_ROOT}}"
NVIDIA_RENDER_ROOT="${NVIDIA_RENDER_ROOT%/}"
NVIDIA_RENDER_VERSION="${UNIVTAC_NVIDIA_RENDER_VERSION:-$(basename -- "${NVIDIA_RENDER_ROOT}")}"

for required_path in \
  "${NVIDIA_RENDER_ROOT}/runtime-libs/libEGL_nvidia.so.0" \
  "${NVIDIA_RENDER_ROOT}/runtime-libs/libGLX_nvidia.so.0" \
  "${NVIDIA_RENDER_ROOT}/nvidia_icd.local.json" \
  "${NVIDIA_RENDER_ROOT}/10_nvidia.local.json"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "NVIDIA render bundle is incomplete: ${required_path}" >&2
    exit 1
  fi
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to validate the kernel/userspace driver match." >&2
  exit 1
fi
mapfile -t HOST_DRIVER_VERSIONS < <(
  nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u
)
if [[ "${#HOST_DRIVER_VERSIONS[@]}" -ne 1 ]] || \
   [[ "${HOST_DRIVER_VERSIONS[0]:-}" != "${NVIDIA_RENDER_VERSION}" ]]; then
  echo "NVIDIA driver mismatch: bundle=${NVIDIA_RENDER_VERSION}, host=${HOST_DRIVER_VERSIONS[*]:-unknown}" >&2
  echo "Set UNIVTAC_NVIDIA_RENDER_ROOT only to a bundle matching the host kernel driver." >&2
  exit 1
fi

# Preserve unrelated library locations (for example PyTorch), while removing
# host-injected NVIDIA/compat locations so they cannot silently shadow or fill
# gaps in the selected data-disk bundle.
SANITIZED_LIBRARY_PATH=""
IFS=':' read -r -a INHERITED_LIBRARY_PATHS <<< "${LD_LIBRARY_PATH:-}"
for library_path in "${INHERITED_LIBRARY_PATHS[@]}"; do
  [[ -z "${library_path}" ]] && continue
  case "${library_path}" in
    /usr/local/nvidia|/usr/local/nvidia/*|/usr/local/cuda/compat|/usr/local/cuda/compat/*)
      continue
      ;;
  esac
  if [[ -z "${SANITIZED_LIBRARY_PATH}" ]]; then
    SANITIZED_LIBRARY_PATH="${library_path}"
  else
    SANITIZED_LIBRARY_PATH="${SANITIZED_LIBRARY_PATH}:${library_path}"
  fi
done
export LD_LIBRARY_PATH="${NVIDIA_RENDER_ROOT}/runtime-libs${SANITIZED_LIBRARY_PATH:+:${SANITIZED_LIBRARY_PATH}}"
# libcuda is the kernel-facing system layer. Do not preload a second copy next
# to it; the version guard above ensures it matches this rendering stack.
unset LD_PRELOAD
export VK_ICD_FILENAMES="${NVIDIA_RENDER_ROOT}/nvidia_icd.local.json"
export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_RENDER_ROOT}/10_nvidia.local.json"
export __GLX_VENDOR_LIBRARY_NAME="nvidia"
export EGL_PLATFORM="surfaceless"

RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/univtac-agentenv-${UID}}"
mkdir -p "${RUNTIME_DIR}"
chmod 700 "${RUNTIME_DIR}"
export XDG_RUNTIME_DIR="${RUNTIME_DIR}"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${REPO_ROOT}"
fi

# The runner is entirely local and should not inherit download proxies.
unset http_proxy https_proxy ftp_proxy all_proxy no_proxy
unset HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY NO_PROXY

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" scripts/grasp_classify_agent_env.py "$@"
