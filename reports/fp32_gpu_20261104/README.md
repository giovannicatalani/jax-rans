Validation GPU FP32 du 4 novembre 2026

Audits exécutés par l’utilisateur sur sa RTX 4070 Ti SUPER, lus et comparés
aux empreintes du code dans cet espace de travail. JAX 0.10.2, Python 3.11.14.
Les 24 premiers cas de la campagne sont testés sur les maillages complets
(70 592 cellules par section), avec préallocation JAX désactivée.

| Mesure | Avant : 40 itérations au maximum | Après : 100 itérations par cas |
|---|---:|---:|
| Cas sélectionnés | 24 | 24 |
| Cas `stalled` | 23 | 0 |
| Corrections acceptées | 47 | 2 400 |
| Corrections non finies rejetées | 533 | 0 |
| Corrections rejetées pour état non physique | 0 | 0 |
| Corrections rejetées faute de descente | 0 | 0 |
| Champs finaux finis | 24/24 | 24/24 |

Après correction, chacun des cinq blocs de 20 itérations accepte ses 480
corrections (24 cas × 20), avec modification effective des champs. Les 24
cas se terminent par `max_iters`, leur budget d’audit de 100 étant atteint.
Ce statut ne signifie pas qu’ils ont convergé.

Cause corrigée : le plafonnement de `r_of` dans Spalart–Allmaras était
appliqué après une division dont la dérivée pouvait déborder en FP32.
Dans la branche plafonnée, la valeur est 10 et la dérivée doit être nulle,
mais l’ancien calcul produisait `0 * inf = NaN`. Le nouveau chemin FP32
protège le quotient avant division, sans modifier la branche active ni
le chemin FP64. Les tests analytiques couvrent les dérivées avant/arrière.

Les empreintes de tous les fichiers Python du solveur sont conservées dans
les deux fichiers `configuration.json`. Entre ces deux audits GPU, seul
`src/jaxrans/physics/spalart_allmaras.py` diffère ; les empreintes de l’audit
après correction correspondent au code actuel au moment de ce rapport.

Reproduction :

```bash
python scripts/diagnose_precision.py --precision fp32 \
  --sections 0 1 2 3 4 5 6 7 8 9 --cases 24 --iters 100 \
  --out /tmp/wing-fp32-batch24-fixed
```

Données brutes : `before_batch24/` et `after_batch24/`, chacun contenant
`solver_progress.csv`, `configuration.json` et `summary.json`.

Portée : démarrage et progression des 24 cas sur 100 itérations sur ce GPU.
La convergence finale des 90 cas, l’équivalence quantitative des coefficients
convergés FP32/FP64 et un gain de performance mesuré restent à établir.
