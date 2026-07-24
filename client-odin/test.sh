#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INGOT="${INGOT:-$ROOT_DIR/../ingot}"

odin check "$SCRIPT_DIR" \
	-collection:ingot="$INGOT" \
	-vet \
	-strict-style \
	-vet-shadowing
