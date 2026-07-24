#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INGOT="${INGOT:-$ROOT_DIR/../ingot}"

if [ ! -d "$INGOT/ui" ]; then
	echo "Invalid Ingot checkout: $INGOT" >&2
	exit 1
fi

mkdir -p "$SCRIPT_DIR/dist"
odin build "$SCRIPT_DIR" \
	-out:"$SCRIPT_DIR/dist/melody-trainer" \
	-o:speed \
	-collection:ingot="$INGOT"

echo "built: $SCRIPT_DIR/dist/melody-trainer"
