#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
    GUI_PYTHON="$PROJECT_DIR/.venv/bin/python"
else
    GUI_PYTHON=python3
fi
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$GUI_PYTHON" -m urdf_dh.gui "$@"
