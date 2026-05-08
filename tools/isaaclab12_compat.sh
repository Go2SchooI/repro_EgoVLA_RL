#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${CONDA_DEFAULT_ENV:-}" != "vila" ]]; then
    echo "[ERROR] Please run 'conda activate vila' before using this wrapper." >&2
    exit 1
fi

export IsaacLab_PATH="${IsaacLab_PATH:-/home/jizexian/third_party/IsaacLab-v1.2.0}"
export ISAACLAB_PATH="${ISAACLAB_PATH:-${IsaacLab_PATH}}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
if [[ -z "${TERM:-}" || "${TERM}" == "dumb" ]]; then
    export TERM="xterm"
fi
export PYTHONPATH="${REPO_ROOT}/compat${PYTHONPATH:+:${PYTHONPATH}}"

exec "${IsaacLab_PATH}/isaaclab.sh" "$@"
