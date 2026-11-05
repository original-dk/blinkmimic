#!/bin/bash
# Download the G1-retargeted LAFAN1 motions and the G1 description, then build the MJCF asset.
# LAFAN1 (Ubisoft La Forge) is CC BY-NC-ND 4.0; the Unitree G1 description is BSD-3-Clause.
set -euo pipefail
cd "$(dirname "$0")/.."
HF=https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
API=https://huggingface.co/api/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

fetch() {
    curl --fail --show-error --silent --location --retry 3 --connect-timeout 15 --max-time 300 "$@"
}

list() {
    echo "Listing $1 ..." >&2
    fetch "$API/$1" -o "$tmp/list.json" || return
    python3 - "$tmp/list.json" "$1" > "$2" <<'PY'
import json
import sys

try:
    entries = json.load(open(sys.argv[1]))
except (ValueError, OSError) as e:
    sys.exit(f"Hugging Face returned an invalid JSON listing for {sys.argv[2]}: {e}. Check network / proxy access.")
if not isinstance(entries, list) or not entries:
    sys.exit(f"Hugging Face returned no file listing for {sys.argv[2]}. Check the dataset URL and network access.")
paths = []
for entry in entries:
    if not isinstance(entry, dict) or entry.get("type") != "file":
        continue
    path = entry.get("path", "")
    prefix = sys.argv[2] + "/"
    name = path.removeprefix(prefix)
    if not path.startswith(prefix) or not name or "/" in name or name in (".", "..") or any(c.isspace() for c in name):
        sys.exit("Unexpected file path in the Hugging Face listing")
    paths.append(path)
if not paths:
    sys.exit(f"No files found in the Hugging Face listing for {sys.argv[2]}")
print("\n".join(paths))
PY
}

download() {
    echo "Downloading $1 ..." >&2
    fetch "$HF/resolve/main/$1" -o "$tmp/download" || return
    test -s "$tmp/download" || { echo "Empty download: $1" >&2; return 1; }
    mv "$tmp/download" "$2"
}

mkdir -p data/lafan1_g1 data/g1_description/meshes
list g1 "$tmp/motions"
list robot_description/g1/meshes "$tmp/meshes"
while IFS= read -r f; do download "$f" "data/lafan1_g1/$(basename "$f")"; done < "$tmp/motions"
download robot_description/g1/g1_29dof_rev_1_0.urdf data/g1_description/g1_29dof_rev_1_0.urdf
while IFS= read -r f; do download "$f" "data/g1_description/meshes/$(basename "$f")"; done < "$tmp/meshes"
echo "motions: $(ls data/lafan1_g1 | wc -l) clips"
uv run --locked python tools/urdf_to_mjcf.py
