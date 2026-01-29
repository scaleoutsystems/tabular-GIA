#!/usr/bin/env bash
# HOW TO USE:
# 1. Make sure you are in virtual env
# 2. chmod +x download_datasets.sh
# 3. ./download_datasets.sh data

set -euo pipefail

REPO_ID="ivopersus/LeakProTabular"
DEST_DIR="${1:-./data}"

mkdir -p "$DEST_DIR"

hf download "$REPO_ID" \
  --repo-type dataset \
  --local-dir "$DEST_DIR" \
  --max-workers 8

echo "Downloaded to: $DEST_DIR"
echo "You should now have: $DEST_DIR/binary, $DEST_DIR/multiclass, $DEST_DIR/regression"
