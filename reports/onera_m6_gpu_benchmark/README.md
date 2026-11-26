# Newton–Krylov mixte sur ONERA M6

Mesure CUDA sur NVIDIA GeForce RTX 4070 Ti SUPER (16 Gio), JAX 0.10.2. Le cas
`examples/onera_m6_sa` utilise son maillage fourni de 16 896 cellules, SA,
Mach 0,84 et une incidence de 3,06°. Le benchmark compare le commit de référence
`062a7d7` et les sources modifiées, avec les mêmes paramètres numériques.

## Trajectoires complètes

Chaque bloc de 20 pas est synchronisé. Les temps par pas excluent la
compilation et le premier bloc. Le temps jusqu'au seuil inclut le premier
bloc et exclut la compilation, les diagnostics finaux et l'export. L'arrêt
se fait à la première fin de bloc dont le résidu de masse atteint `1e-12`.

| Mesure | Référence | Version optimisée |
|---|---:|---:|
| Temps moyen après chauffe | 11,86 ms/pas | 6,87 ms/pas |
| Temps médian après chauffe | 7,39 ms/pas | 4,32 ms/pas |
| Temps jusqu'au seuil | 17,60 s | 10,59 s |
| Pas acceptés | 1 440 | 1 480 |
| Replis FP64 | 0 | 0 |
| Résidu FP64 réévalué sur le champ final | 8,98e-13 | 8,51e-13 |
| CL | 0,221112303496 | 0,221112303543 |
| CD | 0,038967207903 | 0,038967207925 |

## Rejeu des mêmes états

Les deux versions démarrent depuis les mêmes champs, CFL et normes
sauvegardés par la référence. La mesure est la médiane de sept blocs
synchronisés de 20 pas, après un bloc de chauffe pour chaque état.

| État de référence | Référence (ms/pas) | Version optimisée (ms/pas) | Temps gagné |
|---:|---:|---:|---:|
| Pas 20 | 12,13 | 7,20 | 40,6 % |
| Pas 200 | 12,04 | 7,61 | 36,8 % |
| Pas 400 | 37,75 | 23,44 | 37,9 % |
| Pas 500 | 7,53 | 4,07 | 45,9 % |

Les fichiers [before.json](before.json) et [after.json](after.json)
contiennent les chronos individuels, le CFL, le résidu et les compteurs
par bloc. L’export Tecplot de la solution calculée contient 384 points
dans la zone ordonnée `I=16, J=24`. Les coordonnées et coefficients ont
été comparés à la sortie tabulaire, et la norme de Cf a été recalculée
à partir de ses trois composantes.

## Référence FP64 et version optimisée mixte

Une seconde comparaison force `fp64` sur le commit `062a7d7` et `mixed`
sur les sources modifiées. Le plafond est de 2 000 pas pour les deux
exécutions afin de mesurer le temps jusqu’au même seuil de `1e-12`.
Les blocs, le GPU et les règles de mesure restent les mêmes.

| Mesure | Référence FP64 | Version optimisée mixte |
|---|---:|---:|
| Temps moyen après chauffe | 12,47 ms/pas | 6,55 ms/pas |
| Temps médian après chauffe | 7,31 ms/pas | 3,95 ms/pas |
| Temps jusqu’au seuil, compilation exclue | 19,00 s | 10,08 s |
| Pas acceptés | 1 520 | 1 480 |
| Résidu FP64 réévalué sur le champ final | 8,06e-13 | 8,51e-13 |

Le temps moyen par pas baisse de 47,5 % et le temps jusqu’au seuil de 46,9 %.
Les écarts absolus sont de 2,11e-11 sur CL et de 1,81e-12 sur CD.

| État de départ identique | Référence FP64 (ms/pas) | Version optimisée mixte (ms/pas) | Temps gagné |
|---:|---:|---:|---:|
| Pas 20 | 13,58 | 6,62 | 51,3 % |
| Pas 200 | 13,81 | 6,52 | 52,7 % |
| Pas 400 | 42,32 | 20,66 | 51,2 % |
| Pas 500 | 7,71 | 4,27 | 44,6 % |

[Chronos FP64](before_fp64_to_tol.json) · [Chronos mixtes](after_mixed_to_tol.json)

## Reproduction

Depuis la racine du dépôt, avec JAX CUDA et le même environnement Python pour
les deux exécutions.

```bash
reference_dir=$(mktemp -d)
git archive 062a7d7 src | tar -x -C "$reference_dir"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1
PYTHONPATH="$reference_dir/src" .venv/bin/python scripts/benchmark_onera_m6.py \
    "$reference_dir/before.json" --save-checkpoints "$reference_dir/states.npz"
PYTHONPATH=src .venv/bin/python scripts/benchmark_onera_m6.py \
    "$reference_dir/after.json" --replay-checkpoints "$reference_dir/states.npz"
```

Pour la comparaison FP64, ajoute `--precision fp64 --max-iters 2000` à
l’exécution sur le commit de référence et `--precision mixed --max-iters 2000`
à celle sur les sources modifiées. Utilise `--replay-checkpoints` avec le
même fichier d’états pour les deux.
