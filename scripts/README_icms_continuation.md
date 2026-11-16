# ICMS continuation campaigns

Provide a campaign directory with `case.json`, an ICMS coordinate file, and
optionally `numerical.json`. The case uses the usual case configuration with
these additional fields (adjust the values to the input file):

```json
"sections": {"file": "wing.icms", "count": 10, "units": "m", "header_lines": null},
"campaign": {"mach": [0.3, 0.5], "alpha_deg": [-2, 0, 2], "batch_size": 4}
```

Set `domain.grid` to `c_grid` or `o_grid`, and supply `freestream.reynolds`.
Do not specify `domain.airfoil`, `domain.sectional`, or freestream Mach,
angle of attack or sweep: the script supplies them. ICMS sections must have
identical point counts, run from root to tip with increasing Y, and each
profile must start and end at the trailing edge. Coordinates are in metres
or millimetres; `header_lines: null` detects the trailing coordinate block.

From the repository root, with the 2.75D branch checked out and dependencies
installed:

```bash
python scripts/icms_continuation.py /path/to/campaign \
  --jaxrans-root . --out /tmp/wing-275d --sectional-275d --prepare-only
python scripts/icms_continuation.py /path/to/campaign \
  --jaxrans-root . --out /tmp/wing-275d --sectional-275d --resume
```

The 2.75D mode uses leading-edge sweep and `taper_rate = -dc/ds`, with
`ds = hypot(dY, dZ)` measured between quarter-chord points. Each section uses
the segment to its next neighbour; the tip uses the preceding segment. Thus
positive taper means decreasing chord. Profiles are processed one at a time
in this mode because the campaign runner does not batch all sectional geometry.
The target checkout must contain `src/jaxrans/convergence.py` to run.

For 2.5D, omit `--sectional-275d` and optionally select the sweep reference:

```bash
python scripts/icms_continuation.py /path/to/campaign \
  --jaxrans-root . --out /tmp/wing-25d --sweep-position 0.25
```

The default is quarter chord; use `0` for leading edge or `1` for trailing
edge. Inspect `inputs/section_NNN/geometry.json` for calculated values and
`inputs/section_NNN/case.json` for generated settings. Resume requires the
same inputs, options, selected sections and target revision. Use `--sections`
to select zero-based indices, consistently for preparation and resume.
