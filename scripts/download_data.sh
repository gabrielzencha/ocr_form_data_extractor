#!/usr/bin/env bash
# Download public form-understanding datasets into data/raw/<name>/.
# Usage: bash scripts/download_data.sh [funsd] [xfund] [naf] [bafco] [commonforms_sample]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
mkdir -p "$RAW"
want() { [ $# -eq 0 ] && return 0; for a in "${ARGS[@]}"; do [ "$a" = "$1" ] && return 0; done; return 1; }
ARGS=("$@")

if [ ${#ARGS[@]} -eq 0 ] || want funsd; then
  echo "== FUNSD"
  mkdir -p "$RAW/funsd" && cd "$RAW/funsd"
  [ -d dataset ] || { curl -sSL -o dataset.zip https://guillaumejaume.github.io/FUNSD/dataset.zip && unzip -q dataset.zip && rm dataset.zip; }
  rm -rf __MACOSX
fi

if [ ${#ARGS[@]} -eq 0 ] || want xfund; then
  echo "== XFUND"
  mkdir -p "$RAW/xfund" && cd "$RAW/xfund"
  for lang in zh ja es fr it de pt; do
    for split in train val; do
      [ -f "$lang.$split.json" ] || curl -sSL -o "$lang.$split.json" "https://github.com/doc-analysis/XFUND/releases/download/v1.0/$lang.$split.json"
      if [ ! -d "$lang.$split" ]; then
        curl -sSL -o "$lang.$split.zip" "https://github.com/doc-analysis/XFUND/releases/download/v1.0/$lang.$split.zip"
        mkdir -p "$lang.$split" && unzip -q -o "$lang.$split.zip" -d "$lang.$split" && rm "$lang.$split.zip"
      fi
    done
  done
fi

if [ ${#ARGS[@]} -eq 0 ] || want naf; then
  echo "== NAF (annotations only; images need the separate release, see README)"
  mkdir -p "$RAW/naf" && cd "$RAW/naf"
  [ -d NAF_dataset ] || git clone -q --depth 1 https://github.com/herobd/NAF_dataset.git
fi

echo "done -> $RAW"
