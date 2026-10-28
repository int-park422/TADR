#!/usr/bin/env bash
# Full 5-seed suite: every table and figure in the paper.
#
# One process per (experiment, seed). The jobs are independent, so results land
# incrementally, a crash costs one shard rather than the run, and the queue can
# spread across every GPU. Peak VRAM per job is ~240 MB, so the limit is compute,
# not memory -- hence several concurrent jobs per GPU.
#
# The latency experiments (exp6, exp8) run afterwards, one at a time on an
# otherwise idle GPU: measured alongside other jobs they come out inflated
# several-fold.
#
#   TADR_DATA_DIR=/path/to/data ./run_all.sh [jobs_per_gpu]      default 3
#
# Re-running skips shards that already completed, so it doubles as resume.
set -u
cd "$(dirname "$0")"

PER_GPU=${1:-3}
N_GPU=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$N_GPU" -lt 1 ] && N_GPU=1
ROOT=results_final
mkdir -p "$ROOT" logs

SEEDS="0 1 2 3 4"
JOBS=$(mktemp)
# Cheapest first, so the single-entity tables land early; the pooled
# experiments (exp2-4) take one to two hours per shard.
for e in 1 5 11 10 9 7; do
    case $e in 1|5) script=tadr_experiments.py ;; *) script=rebuttal_experiments.py ;; esac
    for s in $SEEDS; do echo "$script $e $s" >> "$JOBS"; done
done
for e in 3 2 4; do for s in $SEEDS; do echo "tadr_experiments.py $e $s" >> "$JOBS"; done; done

TOTAL=$(wc -l < "$JOBS")
STREAMS=$((N_GPU * PER_GPU))
echo "$TOTAL jobs | ${N_GPU} GPU x ${PER_GPU} = ${STREAMS} concurrent streams"
echo "data <- ${TADR_DATA_DIR:-data}   results -> $ROOT/   logs -> logs/   started $(date '+%F %T')"
echo

run_job() {  # run_job <gpu> <script> <exp> <seed>
    local gpu=$1 script=$2 exp=$3 seed=$4
    local out="$ROOT/exp${exp}_seed${seed}" log="logs/exp${exp}_seed${seed}.log"
    if [ -f "$out/exp${exp}_runs.csv" ]; then
        echo "[skip] exp$exp seed$seed already complete"
        return 0
    fi
    local start=$(date +%s)
    echo "[gpu$gpu] start exp$exp seed$seed"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u "$script" --exp "$exp" --seed-list "$seed" \
        --out "$out" > "$log" 2>&1
    local rc=$? mins=$(( ($(date +%s) - start) / 60 ))
    if [ $rc -eq 0 ]; then
        echo "[gpu$gpu] done  exp$exp seed$seed  (${mins}m)"
    else
        echo "[gpu$gpu] FAIL  exp$exp seed$seed  rc=$rc  (${mins}m) -> $log"
    fi
}
export -f run_job
export ROOT

nl -ba "$JOBS" | tr -s ' ' | sed 's/^ *//; s/\t/ /' \
| xargs -P "$STREAMS" -I{} bash -c '
    read -r idx script exp seed <<< "{}"
    run_job $(( (idx - 1) % '"$N_GPU"' )) "$script" "$exp" "$seed"
'
rm -f "$JOBS"

echo
echo "--- latency experiments, one at a time ---"
for s in $SEEDS; do run_job 0 tadr_experiments.py 6 "$s"; done
# exp8 profiles hardware, not data: one run is enough.
run_job 0 rebuttal_experiments.py 8 0

echo
echo "--- merging shards and writing REPORT.md ---"
python3 finalize.py --root "$ROOT" --logs logs
echo "finished $(date '+%F %T')"
