#!/usr/bin/env bash
# End-to-end smoke test on synthetic data (CPU, ~2-4 minutes).  Run from the repository root:
#     bash tests/smoke_test.sh
set -euo pipefail
ROOT=${1:-/tmp/sia_synth}
OUT=${2:-/tmp/sia_synth_results}
rm -rf "$ROOT" "$OUT"
python tests/make_synthetic.py "$ROOT"
python -m sia.cli all --quick --no-pretrained --device cpu \
    --oasis-root "$ROOT/oasis" --tcga-root "$ROOT/tcga" --out "$OUT"
echo
echo "produced:"
ls "$OUT" "$OUT/figures"
