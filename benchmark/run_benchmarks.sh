#!/usr/bin/env bash
# Run the full benchmark (serial sizes 10^3–10^7 and parallel throughput at N = 10^5) and
# keep the output for docs/src/performance.md.
#
# Inputs : THREADS (optional env var, default 16) — threads for the parallel mode. Hyperion
#          is shared and unscheduled, so keep this well below the 56 cores.
# Outputs: .development/benchmark/benchmark-<host>-<date>.txt (gitignored), also on stdout.
# Run    : bash benchmark/run_benchmarks.sh      (from the repository root, e.g. in tmux)
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
JULIA=/srv/home/aste0033/.juliaup/bin/julia
THREADS=${THREADS:-16}
OUTDIR="$REPO/.development/benchmark"
LOG="$OUTDIR/benchmark-$(hostname -s)-$(date +%Y-%m-%d).txt"
mkdir -p "$OUTDIR"
export JULIA_PKG_OFFLINE=true

cd "$REPO"
{
    echo "# $(date -Is) on $(hostname), git $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
    echo "## serial growth and statistics"
    "$JULIA" --project=. benchmark/benchmark.jl
    echo
    echo "## parallel throughput"
    "$JULIA" -t "$THREADS" --project=. benchmark/benchmark.jl --parallel 100000
} 2>&1 | tee "$LOG"
echo "saved to $LOG"
