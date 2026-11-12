#!/usr/bin/env python3
"""Run the continuation branch's campaign runner on an ICMS campaign.

Example (use a separate checkout of the target branch):
  python scripts/icms_continuation.py examples/wing_sections --jaxrans-root ../jax-rans-polar \
      --out /tmp/wing-polar --batch-size 4 --prepare-only
  python scripts/icms_continuation.py examples/wing_sections --jaxrans-root ../jax-rans-polar \
      --out /tmp/wing-polar --batch-size 4 --resume

The input uses the ICMS branch's case.json and numerical.json schema. Results,
intermediate HUG profiles and the continuation manifest stay under --out.

Depuis la racine du dépôt, il te faut un checkout séparé de la branche cible, avec ses dépendances installées. Par exemple :

git clone --branch feature/continuation-batch-alpha-polar-fp32 \
  . ../jax-rans-polar

Le checkout cible doit aussi contenir src/jaxrans/convergence.py : la branche l’importe, mais ce fichier manque actuellement dans sa révision suivie.

Prépare ensuite une campagne ICMS dans un répertoire contenant case.json, numerical.json et ton fichier .icms. L’exemple case.json (examples/wing_sections/case.json) montre le format. Les paramètres propres à l’ICMS et à la polaire sont notamment :

"sections": {
  "file": "wing.icms",
  "count": 10,
  "units": "m",
  "header_lines": null
},
"campaign": {
  "mach": [0.3, 0.5],
  "alpha_deg": [-2.0, 0.0, 2.0, 4.0],
  "batch_size": 4
}

count est le nombre de sections, chacune devant avoir le même nombre de points. Les sections doivent être ordonnées de l’emplanture vers le saumon. Les unités possibles sont m et mm.

Pour préparer les profils et contrôler les fichiers générés sans lancer de calcul :

python scripts/icms_continuation.py examples/wing_sections \
  --jaxrans-root ../jax-rans-polar \
  --out /tmp/wing-polar \
  --prepare-only

Tu trouveras les profils normalisés dans /tmp/wing-polar/inputs/section_000/profile.hug, etc., ainsi qu’un geometry.json par section et le manifest.json de la campagne.

Pour lancer ensuite les calculs dans ce même dossier :

python scripts/icms_continuation.py examples/wing_sections \
  --jaxrans-root ../jax-rans-polar \
  --out /tmp/wing-polar \
  --batch-size 4 \
  --resume

Pour une nouvelle campagne, tu peux omettre l’étape --prepare-only et --resume. Les résultats sont placés sous /tmp/wing-polar/results/. Pour limiter le calcul à certaines sections, ajoute par exemple --sections 0 3 9 ; il faut garder la même sélection lors de la reprise.

L’interpréteur Python utilisé pour lancer le script doit avoir les dépendances de la branche cible disponibles. La préparation des profils est vérifiée ; le calcul complet reste à valider une fois le module manquant disponible.


"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def xyz(line):
    words = line.split()
    if len(words) != 3:
        return None
    try:
        return [float(w.replace('D', 'E').replace('d', 'e')) for w in words]
    except ValueError:
        return None


def read_sections(path, config):
    count, units = config['count'], config['units']
    if isinstance(count, bool) or not isinstance(count, int) or count < 2 or units not in ('m', 'mm'):
        raise ValueError('sections requires count >= 2 and units m or mm')
    tol = config.get('plane_tolerance', 1e-7)
    if not isinstance(tol, (int, float)) or isinstance(tol, bool) or not np.isfinite(tol) or tol <= 0:
        raise ValueError('plane_tolerance must be finite and positive')
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    rows = [(i + 1, line) for i, line in enumerate(lines) if line.strip()]
    header = config.get('header_lines')
    if header is None:
        start = len(rows)
        while start and xyz(rows[start - 1][1]) is not None:
            start -= 1
        rows = rows[start:]
    else:
        if isinstance(header, bool) or not isinstance(header, int) or header < 0:
            raise ValueError('header_lines must be a nonnegative integer or null')
        rows = [(i, line) for i, line in rows if i > header]
    if len(rows) < 5 * count or len(rows) % count:
        raise ValueError('ICMS coordinate rows cannot form equal profiles; check count and header_lines')
    values = []
    for i, line in rows:
        p = xyz(line)
        if p is None:
            raise ValueError(f'ICMS line {i}: expected three coordinates')
        values.append(p)
    points = np.asarray(values) * (0.001 if units == 'mm' else 1.0)
    sections = []
    n = len(points) // count
    for i in range(count):
        p = points[i*n:(i+1)*n]
        if not np.isfinite(p).all() or np.any(np.linalg.norm(np.diff(p, axis=0), axis=1) == 0):
            raise ValueError(f'section {i}: nonfinite or repeated consecutive points')
        te = (p[0] + p[-1]) / 2
        k = int(np.argmax(np.linalg.norm(p - te, axis=1)))
        le = p[k]
        chord = float(np.linalg.norm(te - le))
        if chord == 0 or k in (0, len(p)-1):
            raise ValueError(f'section {i}: degenerate chord')
        _, singular, vh = np.linalg.svd((p - p.mean(axis=0)) / chord, full_matrices=False)
        if singular[1] < 1e-10:
            raise ValueError(f'section {i}: collinear points')
        normal = vh[-1]
        error = float(np.abs((p - p.mean(axis=0)) @ normal).max() / chord)
        if error > tol:
            raise ValueError(f'section {i}: non-planar points ({error:.3e} chord)')
        ec = (te - le) / chord
        ez = np.cross(ec, normal)
        ez /= np.linalg.norm(ez)
        if ez[2] < 0:
            ez = -ez
        if ez[2] < 1e-8:
            raise ValueError(f'section {i}: thickness axis cannot be oriented upwards')
        normal = np.cross(ec, ez)
        if abs(normal[0]) > 1e-6:
            raise ValueError(f'section {i}: cut plane must contain aircraft X')
        xy = (p - le) @ np.column_stack((ec, ez)) / chord
        if np.any(np.linalg.norm(np.diff(xy, axis=0), axis=1) == 0):
            raise ValueError(f'section {i}: coincident points after flattening')
        area = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]) / 2
        if area > 0:
            xy = xy[::-1]
        sections.append(dict(index=i, xy=xy, quarter=le + 0.25 * (te-le), chord=chord,
                             le=le, te=te, basis=np.column_stack((ec, ez, normal)),
                             plane_error=error, lines=[rows[i*n][0], rows[(i+1)*n-1][0]]))
    q = np.array([s['quarter'] for s in sections])
    if np.any(np.diff(q[:, 1]) <= 0):
        raise ValueError('ICMS profiles must run root to tip with increasing quarter-chord Y')
    for i, section in enumerate(sections):
        j = max(0, i-1)
        dx, dy, dz = q[j+1] - q[j]
        section['sweep'] = float(np.degrees(np.arctan2(dx, np.hypot(dy, dz))))
        section['sweep_segment'] = [j, j+1]
    return sections


def expand(value, name):
    if isinstance(value, dict) and set(value) == {'start', 'stop', 'step'}:
        start, stop, step = (value[k] for k in ('start', 'stop', 'step'))
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x) for x in (start, stop, step)) or step <= 0 or stop < start:
            raise ValueError(f'{name}: invalid range')
        intervals = (stop-start)/step
        if intervals > 100000 or not np.isclose(intervals, round(intervals), atol=1e-9, rtol=1e-12):
            raise ValueError(f'{name}: step must divide range')
        value = np.linspace(start, stop, round(intervals)+1).tolist()
    if not isinstance(value, list) or not value or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x) for x in value) or len(set(value)) != len(value):
        raise ValueError(f'{name}: expected distinct finite numbers')
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('case', type=Path, help='ICMS campaign directory or case.json')
    ap.add_argument('--jaxrans-root', type=Path, required=True, help='checkout of feature/continuation-batch-alpha-polar-fp32')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--batch-size', type=int)
    ap.add_argument('--devices', type=int)
    ap.add_argument('--sections', type=int, nargs='+', help='zero-based section indices (default: all)')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()
    try:
        root = args.jaxrans_root.resolve()
        branch = subprocess.check_output(['git', '-C', str(root), 'branch', '--show-current'], text=True).strip()
        if branch != 'feature/continuation-batch-alpha-polar-fp32':
            raise ValueError(f'{root}: expected checkout of feature/continuation-batch-alpha-polar-fp32, got {branch!r}')
        if not args.prepare_only and not (root / 'src/jaxrans/convergence.py').is_file():
            raise ValueError(f'{root}: src/jaxrans/convergence.py is missing; the target branch imports it and cannot start')
        casepath = args.case.resolve()
        if casepath.is_dir():
            casepath /= 'case.json'
        doc = json.loads(casepath.read_text())
        numerical_path = casepath.parent / 'numerical.json'
        numerical = json.loads(numerical_path.read_text()) if numerical_path.exists() else {}
        config, schedule = doc['sections'], doc['campaign']
        if set(config) - {'file', 'count', 'units', 'header_lines', 'plane_tolerance'} or set(schedule) - {'mach', 'alpha_deg', 'batch_size'}:
            raise ValueError('unknown sections or campaign keys')
        machs, alphas = expand(schedule['mach'], 'mach'), expand(schedule['alpha_deg'], 'alpha_deg')
        if any(m <= 0 for m in machs) or any(abs(a) > 90 for a in alphas):
            raise ValueError('Mach must be positive and alpha must be in [-90, 90]')
        domain, flow = doc['domain'], doc['freestream']
        if domain.get('grid') not in ('c_grid', 'o_grid') or 'airfoil' in domain or domain.get('section', 'streamwise') != 'streamwise':
            raise ValueError('domain requires c_grid/o_grid, streamwise section and no airfoil')
        if set(flow) & {'mach', 'alpha_deg', 'sweep_deg'} or flow.get('reynolds') is None:
            raise ValueError('freestream requires reynolds; mach, alpha_deg and sweep_deg come from campaign/geometry')
        source = (casepath.parent / config['file']).resolve()
        sections = read_sections(source, config)
        selected = args.sections if args.sections is not None else list(range(len(sections)))
        if len(set(selected)) != len(selected) or any(i < 0 or i >= len(sections) for i in selected):
            raise ValueError('invalid or duplicate section indices')
        out = args.out.resolve()
        # Never silently mix inputs or overwrite an existing campaign.
        setup = {'case': doc, 'numerical': numerical, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                 'sections': selected, 'target_revision': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()}
        setup_path = out / 'icms_setup.json'
        if out.exists() and any(out.iterdir()):
            if not (args.resume and setup_path.exists() and json.loads(setup_path.read_text()) == setup):
                raise ValueError('output exists and is not a matching --resume campaign; choose a new directory')
        out.mkdir(parents=True, exist_ok=True)
        (out / 'icms_setup.json').write_text(json.dumps(setup, indent=2) + '\n')
        profiles = []
        for i in selected:
            s = sections[i]
            folder = out / 'inputs' / f'section_{i:03d}'
            folder.mkdir(parents=True, exist_ok=True)
            hug = folder / 'profile.hug'
            with hug.open('w') as f:
                f.write(f"1\n{len(s['xy'])}\n")
                np.savetxt(f, s['xy'], fmt='%.17g')
            simple = copy.deepcopy({k: v for k, v in doc.items() if k not in ('sections', 'campaign')})
            simple['general']['case_name'] = f"{simple['general']['case_name']}_section_{i:03d}"
            simple['general']['save_path'] = str(out)
            simple['domain'].update(airfoil=str(hug), section='streamwise')
            simple['freestream'].update(mach=machs[0], alpha_deg=alphas[0], sweep_deg=s['sweep'])
            (folder / 'case.json').write_text(json.dumps(simple, indent=2) + '\n')
            (folder / 'numerical.json').write_text(json.dumps(numerical, indent=2) + '\n')
            record = {'index': i, 'source_lines': s['lines'], 'leading_edge_m': s['le'].tolist(), 'trailing_edge_m': s['te'].tolist(), 'quarter_chord_m': s['quarter'].tolist(), 'chord_m': s['chord'], 'basis_columns': s['basis'].tolist(), 'plane_error_over_chord': s['plane_error'], 'sweep_deg': s['sweep'], 'sweep_segment': s['sweep_segment'], 'alpha_correction_deg': 0.0}
            (folder / 'geometry.json').write_text(json.dumps(record, indent=2) + '\n')
            profiles.append({'name': f'section_{i:03d}', 'case': str(folder), 'alphas': alphas, 'machs': machs})
        manifest = out / 'manifest.json'
        manifest.write_text(json.dumps({'profiles': profiles}, indent=2) + '\n')
        print(f'Prepared {len(profiles)} sections, {len(machs)} Mach numbers, {len(alphas)} angles: {out}', flush=True)
        if args.prepare_only:
            return 0
        cmd = [sys.executable, '-m', 'jaxrans.campaign', str(manifest), '--out', str(out / 'results')]
        batch = args.batch_size if args.batch_size is not None else schedule.get('batch_size', 8)
        cmd += ['--batch-size', str(batch)]
        if args.devices is not None: cmd += ['--devices', str(args.devices)]
        if args.resume: cmd.append('--resume')
        if args.quiet: cmd.append('--quiet')
        env = os.environ.copy()
        env['PYTHONPATH'] = str(root / 'src')
        return subprocess.call(cmd, cwd=root, env=env)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as e:
        print(f'error: {e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
