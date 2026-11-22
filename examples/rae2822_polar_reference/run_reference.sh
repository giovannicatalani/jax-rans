#!/usr/bin/env bash
# Reference de performance RAE 2822 transsonique, 2.5D et 2.75D, code actuel.
# Lance sequentiellement les mesures de scripts/benchmark_polar.py.
#
# Usage (depuis n'importe ou) :
#   bash examples/rae2822_polar_reference/run_reference.sh [DOSSIER_SORTIE]
#
# Variables optionnelles :
#   PYTHON=python        interpreteur a utiliser (ex. "uv run python")
#   SKIP_SMOKE=1         saute le test rapide initial
#   ONLY="1 3"           ne lance que les etapes listees (0 a 5)
#
# Chaque etape ecrit son journal dans <sortie>/<etape>.log ; un echec n'arrete
# pas les etapes suivantes (sauf le test rapide 0, qui verifie le GPU).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT="${1:-results/bench_ref/$(date +%Y%m%d_%H%M%S)}"
PYTHON="${PYTHON:-python}"
REF=examples/rae2822_polar_reference
BENCH="scripts/benchmark_polar.py"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$OUT"

declare -A NAMES STATUS DURATION
NAMES[0]="smoke_25d"
NAMES[1]="25d_campaign"
NAMES[2]="25d_sequence"
NAMES[3]="275d_sequence"
NAMES[4]="25d_components"
NAMES[5]="275d_components"

command_for() {
    case "$1" in
        0) echo "$BENCH polar $REF/polar_25d.json --alpha 0 1 1 --out $OUT/smoke_25d" ;;
        1) echo "$BENCH polar $REF/polar_25d.json --out $OUT/25d_campaign" ;;
        2) echo "$BENCH sequence $REF/case_25d --alpha -2 4 0.5 --out $OUT/25d_sequence" ;;
        3) echo "$BENCH sequence $REF/case_275d --alpha -2 4 0.5 --out $OUT/275d_sequence" ;;
        4) echo "$BENCH components $REF/case_25d --batch 1 2 4 8 12 --trace $OUT/25d_trace --out $OUT/25d_components" ;;
        5) echo "$BENCH components $REF/case_275d --batch 1 2 4 --trace $OUT/275d_trace --out $OUT/275d_components" ;;
    esac
}

selected() {
    if [[ -n "${ONLY:-}" ]]; then [[ " $ONLY " == *" $1 "* ]]; return; fi
    [[ "$1" != 0 || -z "${SKIP_SMOKE:-}" ]]
}

{
    echo "date        : $(date)"
    echo "host        : $(hostname)"
    echo "git         : $(git rev-parse HEAD) ($(git branch --show-current))"
    echo "src modifie : $(git status --porcelain src | wc -l) fichier(s)"
    echo "python      : $($PYTHON -c 'import sys; print(sys.version.split()[0])')"
    echo "jax         : $($PYTHON -c 'import jax; print(jax.__version__, jax.devices())' 2>&1 | tail -1)"
    nvidia-smi --query-gpu=name,driver_version,memory.total,utilization.gpu,memory.used --format=csv 2>&1
} | tee "$OUT/environment.txt"

# Display agents (VDI, X server) are ignored: they do not compute.
OTHERS="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null \
          | grep -viE 'VMwareBlastServer|Xorg|gnome-shell|Xwayland' || true)"
if [[ -n "$OTHERS" ]]; then
    echo "ATTENTION : d'autres processus utilisent le GPU, les temps seront faussés :"
    echo "$OTHERS"
fi

for step in 0 1 2 3 4 5; do
    selected "$step" || continue
    name="${NAMES[$step]}"
    cmd="$(command_for "$step")"
    echo
    echo "=== [$step] $name : $(date +%H:%M:%S)"
    echo "    $PYTHON $cmd"
    begin=$(date +%s)
    $PYTHON $cmd > "$OUT/$name.log" 2>&1
    code=$?
    DURATION[$step]=$(( $(date +%s) - begin ))
    STATUS[$step]=$code
    tail -n 25 "$OUT/$name.log"
    echo "=== [$step] $name : code $code, ${DURATION[$step]} s"
    if [[ "$step" == 0 && "$code" != 0 ]] && ! grep -q "time breakdown" "$OUT/$name.log"; then
        echo "Le test rapide a échoué (GPU absent ou environnement incorrect) : arrêt. Voir $OUT/$name.log"
        exit 1
    fi
done

{
    echo
    echo "=== Récapitulatif ($(date))"
    printf "%-4s %-18s %-6s %8s\n" "step" "nom" "code" "durée s"
    for step in 0 1 2 3 4 5; do
        [[ -n "${STATUS[$step]:-}" ]] || continue
        printf "%-4s %-18s %-6s %8s\n" "$step" "${NAMES[$step]}" "${STATUS[$step]}" "${DURATION[$step]}"
    done
    echo "Résultats : $OUT (timing_breakdown.json / components.json, *.log, gpu_samples.csv, traces)"
} | tee "$OUT/summary.txt"
