#!/usr/bin/env bash
# Real ML launcher. Shared config and existing demo outputs are left intact.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_root"
action="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

setup() {
  if [[ ! -x .venv/bin/python ]]; then python3 -m venv .venv; fi
  local torch_index="https://download.pytorch.org/whl/cpu"
  if command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
    torch_index="https://download.pytorch.org/whl/cu128"
  fi
  .venv/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url "$torch_index"
  .venv/bin/python -m pip install -c extension/requirements-real.txt -e '.[server,dev,events,environment]' 'yt-dlp[default]' 'librosa>=0.10.2'
  .venv/bin/python -m pip check
  .venv/bin/python -m soundthing.server.real prepare
  .venv/bin/python -m soundthing.server.real doctor
}

case "$action" in
  help)
    cat <<'HELP'
Usage: bash extension/real.sh COMMAND [OPTIONS]
  setup                  Install all project/model dependencies; validate local real config
  doctor                 Check ML imports, CUDA, tools, config, and AudioSet labels
  models                 Download/load the actual models without processing a video
  run VIDEO              Generate real baseline/conditioned VTTs under runs/real/
  serve [--cookies PATH] Start localhost:8000 using real models and runs/real/
  start [--cookies PATH] Install/check dependencies, load models, then serve
Shared config stays unchanged. Model weights download on first use.
HELP
    ;;
  setup) setup "$@" ;;
  start) setup; .venv/bin/python -m soundthing.server.real models; exec .venv/bin/python -m soundthing.server.real serve "$@" ;;
  doctor|models|run|serve)
    if [[ ! -x .venv/bin/python ]]; then echo "Run 'bash extension/real.sh setup' first." >&2; exit 1; fi
    exec .venv/bin/python -m soundthing.server.real "$action" "$@"
    ;;
  *) echo "Unknown command: $action. Run 'bash extension/real.sh help'." >&2; exit 2 ;;
esac
