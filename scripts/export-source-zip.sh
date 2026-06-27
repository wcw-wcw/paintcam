#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
name=${1:-paintcam-source.zip}

case "$name" in
  /*) output=$name ;;
  *) output="$root/$name" ;;
esac

cd "$root"
rm -f "$output"
zip -r "$output" . \
  -x '.git/*' 'node_modules/*' 'dist/*' 'build/*' 'src-tauri/target/*' \
     '.venv/*' '*/.venv/*' '__pycache__/*' '*/__pycache__/*' \
     'engine/models/*.task' \
     '__MACOSX/*' '*/__MACOSX/*' '.DS_Store' '*/.DS_Store' \
     "$(basename "$output")"
printf 'Created %s\n' "$output"
