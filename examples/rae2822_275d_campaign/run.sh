#!/usr/bin/env bash
# Run as a subprocess, never change the caller's shell options.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo 'Ne pas sourcer ce script. Utiliser : bash ./run.sh subsonic continuity' >&2
    return 0
fi
set -Eeuo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd -- "$repo_root"
if [[ -n "${JAXRANS_PYTHON:-}" ]]; then
    python_bin=$JAXRANS_PYTHON
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
    python_bin="$repo_root/.venv/bin/python"
else
    python_bin=python
fi
regime=${1:-subsonic}
study=${2:-continuity}
case "$regime" in subsonic|transonic) ;; *) echo 'Expected subsonic or transonic' >&2; exit 2;; esac
base="examples/rae2822_${regime}_conical_verification"
out="${JAXRANS_CAMPAIGN_OUT:-results/rae2822_275d_${regime}_mixed}"
precision_args=(--set case.general.precision=mixed)
mkdir -p "$out/logs"
log="$out/logs/$(date +%Y%m%d_%H%M%S)_${study}_$$.log"
exec > >(tee -a "$log") 2>&1
trap 'status=$?; echo "Échec (code $status). Journal : $log" >&2; exit "$status"' ERR
printf 'Dépôt : %s\nPython : %s\nJournal : %s\n' "$repo_root" "$python_bin" "$log"
# Extra CLI options allow --check or a short diagnostic run.
extra_args=("${@:3}")
run_case() {
    local name=$1
    shift
    "$python_bin" -u -m jaxrans "$base" --out "$out/$name" "${precision_args[@]}" "$@" "${extra_args[@]}"
}
case "$study" in
continuity)
    "$python_bin" -u -m jaxrans "examples/rae2822_${regime}_swept_verification" --out "$out/legacy_2p5d" "${precision_args[@]}" "${extra_args[@]}"
    for tau in 0 -0.001 -0.0001 -0.00001 0.00001 0.0001 0.001 0.01 0.03 0.1 0.2; do
        run_case "taper_$tau" --set "case.domain.sectional.taper_rate=$tau"
    done
    ;;
depth)
    for tau in 0 0.001 0.2; do
        for depth in 0.2 0.1 0.05; do
            run_case "taper_${tau}_depth_$depth" --set "case.domain.sectional.taper_rate=$tau" --set "numerical.sectional.depth=$depth"
        done
    done
    ;;
mesh)
    # Same physical boundaries; independent generated grids, not nested grids.
    for level in coarse medium fine; do
        case "$level" in
        coarse) na=96; nw=24; nn=40; spacing=4e-6;;
        medium) na=192; nw=48; nn=80; spacing=2e-6;;
        fine) na=384; nw=96; nn=160; spacing=1e-6;;
        esac
        for tau in 0 0.001 0.2; do
            run_case "taper_${tau}_mesh_$level" --set "case.domain.sectional.taper_rate=$tau" \
                --set "case.domain.cells_airfoil=$na" --set "case.domain.cells_wake=$nw" \
                --set "case.domain.cells_normal=$nn" --set "case.domain.wall_spacing=$spacing"
        done
    done
    ;;
relaxation)
    for window in '0.5 3' '1 5' '2 8'; do
        read -r start end <<< "$window"
        run_case "taper_0.2_relax_${start}_$end" --set case.domain.sectional.taper_rate=0.2 \
            --set "numerical.sectional.relaxation_start=$start" --set "numerical.sectional.relaxation_end=$end"
    done
    ;;
*) echo 'Expected continuity, depth, mesh or relaxation' >&2; exit 2;;
esac
