#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"
PYTHON_BIN="${UNIVTAC_PYTHON:-${PROJECT_ROOT}/miniconda3/envs/UniVTAC/bin/python}"
RUN_REAL=false
LEVEL=3
DEVICE=cuda:0

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "UniVTAC Python is not executable: ${PYTHON_BIN}" >&2
  echo "Set UNIVTAC_PYTHON to the configured environment's Python." >&2
  exit 1
fi

while (($#)); do
  case "$1" in
    --real)
      RUN_REAL=true
      shift
      ;;
    --level)
      LEVEL="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    -h|--help)
      echo "usage: $0 [--real] [--level {1|2|3}] [--device cuda:0]"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m py_compile \
  agent_env/*.py \
  semantic_tools/*.py \
  scripts/grasp_classify_agent_env.py \
  scripts/embodied_agent_env.py \
  scripts/collect_data.py \
  scripts/parallel_collect_data.py \
  scripts/replay.py \
  scripts/freeze_fixed_expert.py \
  scripts/export_fixed_demo.py \
  scripts/validate_fixed_demo_assets.py \
  scripts/run_codex_agent_env.py \
  scripts/run_codex_benchmark.py \
  scripts/run_agent_viewer.py \
  scripts/manage_semantic_services.py \
  tests/agent_env/*.py
"${PYTHON_BIN}" -m pytest tests/agent_env -q
"${PYTHON_BIN}" scripts/run_codex_agent_env.py --level 1 --dry-run >/dev/null
"${PYTHON_BIN}" scripts/run_codex_agent_env.py --level 2 --dry-run >/dev/null
"${PYTHON_BIN}" scripts/run_codex_agent_env.py --level 3 --dry-run >/dev/null
"${PYTHON_BIN}" scripts/run_codex_agent_env.py --level 2 \
  --perception-profile sam3_unidepth_v2 --dry-run >/dev/null
"${PYTHON_BIN}" scripts/run_codex_benchmark.py \
  --task pull_out_key --profile 1 --dry-run >/dev/null
"${PYTHON_BIN}" scripts/run_codex_benchmark.py \
  --task put_bottle_in_shelf --profile 6 \
  --provide-bbox --provide-mask --dry-run >/dev/null
"${PYTHON_BIN}" scripts/run_codex_benchmark.py \
  --task put_bottle_in_shelf --profile 1 --pre-move --dry-run >/dev/null
"${PYTHON_BIN}" scripts/run_codex_benchmark.py \
  --task pull_out_key --profile 6 --max-output-tokens 12000 --dry-run >/dev/null
"${PYTHON_BIN}" scripts/run_codex_benchmark.py \
  --task pull_out_key --profile 6 --provide-bbox --provide-mask \
  --icl fixed_demo --dry-run >/dev/null

if [[ "${RUN_REAL}" == true ]]; then
  "${PYTHON_BIN}" tests/agent_env/real_smoke.py --level "${LEVEL}" --device "${DEVICE}"
else
  echo "Static AgentEnv acceptance passed. Add --real for a real Isaac smoke episode."
fi
