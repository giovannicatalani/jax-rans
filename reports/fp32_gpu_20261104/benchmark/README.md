Mesure de performance FP64/FP32 sur RTX 4070 Ti SUPER, 4 novembre 2026

Mesure exécutée par l’utilisateur avec `scripts/diagnose_precision.py --compare`.
Trois incidences (-2, 0, +2 degrés) de la section 000, Mach 0,3, 70 592
cellules par cas, 100 itérations. Les deux processus utilisent les mêmes
paramètres et les mêmes empreintes du code, à la précision près.
Les 300 mises à jour de chaque précision sont acceptées, sans champ non fini.

| Mesure | FP64 | FP32 | Rapport FP64/FP32 |
|---|---:|---:|---:|
| Itération du lot, médiane hors premier bloc | 168.778 ms | 65.743 ms | 2.57× |
| Premier bloc, compilation incluse | 20.390 s | 17.071 s | — |
| Boucle de résolution, premier bloc et suivi inclus | 42.967 s | 32.157 s | 1.34× |

Les temps d’itération sont mesurés au chronomètre hôte, après synchronisation
de tous les résultats du bloc sur GPU. Quatre blocs chauds de 20 itérations
par précision fournissent la médiane. Les transferts complets des champs,
l’intégration des forces et les écritures du suivi sont hors du timer de bloc.
Ils participent toutefois au temps de la boucle de résolution.
Le prétraitement, l’initialisation des états et les archives finales ne sont
pas inclus dans cette durée de boucle.

Un avertissement de timer CUDA est apparu pendant le premier bloc FP64.
Ce bloc est exclu du calcul de la médiane des temps d’itération.
Les temps bruts par bloc sont conservés dans `fp64/timing.json` et
`fp32/timing.json`, avec les configurations, états de sortie et compteurs
d’acceptation correspondants. `comparison.json` conserve le rapport calculé.

Portée : gain mesuré à nombre d’itérations fixé sur ce lot de trois cas.
Ni le temps jusqu’à une convergence aérodynamique équivalente ni le gain
sur la campagne complète de 90 cas ne sont établis par cette mesure.
