#!/usr/bin/env bash
# Run from any working directory. No system packages or profiles are changed.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_root"
action="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$action" == "setup" ]]; then
  if [[ ! -x .venv/bin/python ]]; then python3 -m venv .venv; fi
  .venv/bin/python -m pip install -e '.[server,dev]'
  .venv/bin/python -m pip install 'yt-dlp[default]'
  exec .venv/bin/python -m soundthing.pipeline doctor "$@"
fi
if [[ "$action" == "help" ]]; then
  cat <<'HELP'
Usage: bash extension/dev.sh COMMAND [OPTIONS]
  setup                       Install Python server/test/downloader dependencies, then check tools
  doctor                      Check server, downloader, ffmpeg/ffprobe, Node, and configured cookies
  serve [--cookies PATH]      Start localhost:8000; no virtualenv activation needed
  sync [--destination PATH]   Update Windows Downloads/soundthing-extension, or a chosen folder
HELP
  exit 0
fi
if [[ ! -x .venv/bin/python ]]; then
  echo "Run 'bash extension/dev.sh setup' first." >&2
  exit 1
fi
case "$action" in
  serve|doctor) exec .venv/bin/python -m soundthing.pipeline "$action" "$@" ;;
  sync) exec .venv/bin/python -m soundthing.pipeline sync-extension "$@" ;;
  *) echo "Unknown command: $action. Run 'bash extension/dev.sh help'." >&2; exit 2 ;;
esac
