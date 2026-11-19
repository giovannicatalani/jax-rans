#!/usr/bin/env bash
# Comparaison avant/apres optimisation sur une meme machine (RTX, H100...).
#
# Usage :
#   bash examples/rae2822_polar_reference/run_compare.sh REF_ROOT NEW_ROOT [SORTIE]
#     REF_ROOT : checkout du code de reference (ex. feature/continuation-batch-alpha-polar-fp32-275d)
#     NEW_ROOT : checkout de la branche perf/gpu-optimisations
#
# Variables : PYTHON (defaut python), ONLY="1 3" (etapes), QUICK=1 (polaires courtes)
#
# Le script de mesure et les cas sont pris dans NEW_ROOT et executes contre
# chaque checkout (le script s'appuie sur le src/ du dossier ou il se trouve :
# il est copie temporairement dans REF_ROOT/scripts s'il n'y existe pas).
set -uo pipefail
REF="$(cd "${1:?REF_ROOT}" && pwd)"
NEW="$(cd "${2:?NEW_ROOT}" && pwd)"
OUT="${3:-$NEW/results/bench_compare/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT" && OUT="$(cd "$OUT" && pwd)"
PYTHON="${PYTHON:-python}"
CASES="$NEW/examples/rae2822_polar_reference"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAXRANS_COMPILATION_CACHE=off           # comparaison a compilation egale
ALPHA=(-2 4 0.5); [[ -n "${QUICK:-}" ]] && ALPHA=(0 2 1)

COPIED=""
if [[ ! -f "$REF/scripts/benchmark_polar.py" ]]; then
    cp "$NEW/scripts/benchmark_polar.py" "$REF/scripts/benchmark_polar.py"; COPIED=1
fi
trap '[[ -n "$COPIED" ]] && rm -f "$REF/scripts/benchmark_polar.py"' EXIT

manifest() {   # $1 = profils (25d), $2 = fichier
    cat > "$2" <<EOF
{"batch_size": 8, "profiles": [{"name": "rae2822_25d", "case": "$CASES/case_25d",
  "alpha": [${ALPHA[0]}, ${ALPHA[1]}, ${ALPHA[2]}], "machs": [0.842, 0.8]}]}
EOF
}
manifest 25d "$OUT/polar_multi_25d.json"

run() {   # $1 etape, $2 nom, $3 racine, reste : arguments du script
    local step=$1 name=$2 root=$3; shift 3
    if [[ -n "${ONLY:-}" && " $ONLY " != *" $step "* ]]; then return; fi
    echo "=== [$step] $name ($(date +%H:%M:%S))"
    (cd "$root" && $PYTHON scripts/benchmark_polar.py "$@" --out "$OUT/$name") > "$OUT/$name.log" 2>&1
    echo "    code $?"; grep -E "ms per|ms/iteration|whole Newton|batch +[0-9]+:|total_wall|utilization" "$OUT/$name.log" | sed 's/^/    /'
}

{ echo "ref: $(git -C "$REF" rev-parse --short HEAD) $(git -C "$REF" branch --show-current)";
  echo "new: $(git -C "$NEW" rev-parse --short HEAD) $(git -C "$NEW" branch --show-current)";
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader;
  $PYTHON -c 'import jax; print("jax", jax.__version__, jax.devices())'; } | tee "$OUT/environment.txt"

# 1-2 : cout d'une iteration et passage a l'echelle avec la taille du lot
run 1 ref_components "$REF" components "$CASES/case_25d" --batch 1 2 4 8 16
run 2 new_components "$NEW" components "$CASES/case_25d" --batch 1 2 4 8 16
# 3-5 : campagne 2 Mach x polaire, ancien ordonnancement puis nouveau
run 3 ref_campaign "$REF" polar "$OUT/polar_multi_25d.json" --quiet
run 4 new_campaign_rounds "$NEW" polar "$OUT/polar_multi_25d.json" --quiet --scheduler rounds
run 5 new_campaign_slots_c4 "$NEW" polar "$OUT/polar_multi_25d.json" --quiet --chains 4
# 6 : 2.75D (n'existait pas en campagne dans la reference)
run 6 new_components_275d "$NEW" components "$CASES/case_275d" --batch 1 2 4 8
echo "Résultats : $OUT"
