# Référence de performance : polaire RAE 2822 transsonique, 2.5D et 2.75D

Ce cas mesure le temps d'une polaire avec le code actuel, avant les optimisations.
Il ne sert pas à valider l'aérodynamique.

| | 2.5D (`case_25d`) | 2.75D (`case_275d`) |
|---|---|---|
| Modèle | aile en flèche infinie, `sweep_deg = 30` | stencil sectionnel conique, flèche de bord d'attaque 30°, effilement 0 |
| Section | dans le sens de l'écoulement (`streamwise`) | idem |
| Écoulement | M 0,842, Re 6,5e6, T 460 K, SA-noft2 (M normal 0,73, conditions AGARD 6) | idem |
| Maillage | grille C 256 + 2×96 × 128 (taille de production), y1 = 2e-6 | idem (le canevas a 5 plans dans l'envergure) |
| Numérique | FP32, implicite, van Albada, `max_relative_update` 0,2 | idem, profondeur sectionnelle 0,1 |
| Arrêt | forces stabilisées (`forces_tol` 1e-4), 3000 itérations au plus, blocs de 20 | idem |
| Polaire | α = −2 → 4° par pas de 0,5 (13 points), en continuation | idem |

Avec un effilement nul, les deux modèles décrivent le même écoulement : les
coefficients doivent concorder, et l'écart de temps vient uniquement de la
formulation.

## Pourquoi cette polaire

- 13 points : autant que la campagne de production (α −2 → 10 par pas de 1).
- Elle couvre la zone linéaire, l'apparition puis le renforcement du choc, et
  le décollement induit par le choc vers α ≥ 3,5° (α normal ≈ 4°). Les points
  difficiles demandent beaucoup plus d'itérations : c'est précisément ce qui
  coûte en continuation (attente du cas le plus lent, CFL repartant de 10).
- Le maillage a la taille de production, pour que l'occupation du GPU soit
  représentative.
- Attention : en continuation, la polaire s'arrête au premier point qui
  n'aboutit pas (`max_iters`). C'est le comportement normal du code ; le
  tableau final le montre.

## Chemins mesurés

- **2.5D, `polar`** : `python -m jaxrans.campaign`, inchangé. C'est le chemin
  de production (`scripts/icms_continuation.py`), instrumenté de l'extérieur.
- **2.75D, `sequence`** : avec le code actuel, `jaxrans.campaign` refuse les
  atlas sectionnels. `campaign._geometry` lit `atlas.walls.faces.a`, qui
  n'existe pas sur les `WallQuads` 3D, et lève une `AttributeError` dès la
  préparation. La continuation passe donc par `simulation.run(initial=U)`,
  point par point, avec une recompilation à chaque point (comportement actuel).
- **2.5D, `sequence`** : le même chemin que le 2.75D, pour comparer les deux
  formulations à chemin identique.

## Contrôle du GPU

Le script s'arrête si JAX ne tourne pas sur le GPU. Il vérifie aussi que
l'état reste en mémoire GPU (`state_on_gpu`), échantillonne `nvidia-smi`
toutes les 200 ms, en distinguant l'intérieur et l'extérieur des blocs de
calcul (`gpu_samples.csv`), et relève le pic de mémoire JAX. L'utilisation
affichée par `nvidia-smi` indique seulement qu'au moins un kernel tourne : elle
ne mesure pas l'occupation des multiprocesseurs. Pour cela, il faut la trace
du profileur (`components --trace`, à ouvrir dans TensorBoard ou Perfetto).
