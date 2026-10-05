#!/bin/bash
# Summarize eval results from all log files.
#
# Usage:
#   bash sdpo/slurm/summarize_eval.sh
#   bash sdpo/slurm/summarize_eval.sh /path/to/logs/dir

set -e

LOG_DIR="${1:-/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/logs/sdpo_acclength}"

echo "========================================================================"
echo "  Eval Summary"
echo "  Log dir: $LOG_DIR"
echo "========================================================================"
echo ""

# Header
printf "%-25s  %8s  %8s  %8s  %8s  %8s  %8s  %8s\n" \
    "Tag" "mt_bench" "gsm8k" "humanev" "qa" "sum" "alpaca" "avg"
printf "%-25s  %8s  %8s  %8s  %8s  %8s  %8s  %8s\n" \
    "-------------------------" "--------" "--------" "--------" "--------" "--------" "--------" "--------"

# Collect results from each eval log
for f in "$LOG_DIR"/eval_*.out; do
    [ -f "$f" ] || continue

    tag=$(grep "AccLength Eval:" "$f" 2>/dev/null | head -1 | sed 's/.*Eval: //' | xargs)
    [ -z "$tag" ] && continue

    # Skip if no results
    grep -q "Mean Alpha:" "$f" || continue

    # Get all alphas in order (mt_bench, gsm8k, humaneval, qa, sum, alpaca)
    alphas=($(grep "Mean Alpha:" "$f" | awk '{print $3}'))

    # Need at least 6 results for a complete eval
    [ ${#alphas[@]} -lt 6 ] && continue

    # Compute average
    avg=$(echo "${alphas[@]}" | tr ' ' '\n' | awk '{s+=$1} END {printf "%.3f", s/NR}')

    printf "%-25s  %8s  %8s  %8s  %8s  %8s  %8s  %8s\n" \
        "$tag" "${alphas[0]}" "${alphas[1]}" "${alphas[2]}" "${alphas[3]}" "${alphas[4]}" "${alphas[5]}" "$avg"

done | sort -t'|' -k1,1 | sort

echo ""
echo "========================================================================"

# Show failed/incomplete evals
echo ""
echo "Failed/Incomplete evals:"
for f in "$LOG_DIR"/eval_*.out; do
    [ -f "$f" ] || continue
    tag=$(grep "AccLength Eval:" "$f" 2>/dev/null | head -1 | sed 's/.*Eval: //' | xargs)
    [ -z "$tag" ] && continue
    n_alphas=$(grep -c "Mean Alpha:" "$f" 2>/dev/null)
    n_alphas=${n_alphas:-0}
    err="${f%.out}.err"
    has_error=0
    [ -f "$err" ] && has_error=$(grep -c "Traceback" "$err" 2>/dev/null) && has_error=${has_error:-0}
    if [ "$n_alphas" -lt 6 ] || [ "$has_error" -gt 0 ]; then
        jobid=$(basename "$f" | grep -oE '[0-9]+')
        printf "  %-25s  results=%d  errors=%d  job=%s\n" "$tag" "$n_alphas" "$has_error" "$jobid"
    fi
done

echo ""
echo "Benchmark order: mt_bench, gsm8k, humaneval, qa, sum, alpaca"
echo "Alpha = mean tokens accepted per verification round (higher is better)"
