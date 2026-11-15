## Campagne en précision mixte

Le lanceur utilise par défaut `case.general.precision=mixed` : état et résidu
FP64, Krylov et préconditionneur FP32 avec repli FP64 si nécessaire. Les
sorties sont placées dans `results/rae2822_275d_<regime>_mixed` afin de
préserver les résultats FP64 précédents. `JAXRANS_CAMPAIGN_OUT` permet de
choisir un autre dossier. Les arguments `--set` supplémentaires peuvent
surcharger la précision.

Les cas conservent `forces_tol=null` et `tol=1e-10` : avec le solveur actuel,
chaque équation doit diminuer de dix ordres depuis son propre maximum. Les
planchers absolus et le critère sur les forces ne sont donc pas utilisés.
Le statut, les résidus par équation et le repli éventuel doivent être examinés
avant toute comparaison. Les budgets restent de 12000 itérations.

# Qualification CFD RAE2822 : 2.5D vers 2.75D

Ces scénarios servent à mesurer la continuité des **solutions convergées**,
puis leurs sensibilités numériques. Ils ne constituent pas des résultats
validés. Les coordonnées RAE2822 sont déjà dans le dépôt ; aucun téléchargement
n'est nécessaire. Le code du solveur n'est pas modifié.

## Quatre cas autonomes

| Cas dans `examples/` | Mach total | Méthode |
|---|---:|---|
| `rae2822_subsonic_swept_verification` | .5 | référence 2.5D |
| `rae2822_subsonic_conical_verification` | .5 | sectionnel, effilement nul par défaut |
| `rae2822_transonic_swept_verification` | .842 | référence 2.5D |
| `rae2822_transonic_conical_verification` | .842 | sectionnel, effilement nul par défaut |

Tous utilisent le même profil **longitudinal** RAE2822, une flèche de bord
d'attaque de 30°, une incidence longitudinale de 2°, Re=6.5e6, SA-noft2,
la loi de Sutherland et le champ extérieur uniforme (`model: freestream`,
traduit en condition caractéristique par le chargeur). Le maillage C est
généré avec 192 cellules de profil demandées, 48 de sillage, 80 normales,
une première hauteur de 2e-6 et un domaine de 100 cordes.

La référence existante `rae2822_swept_sa` n'est PAS directement comparable :
elle emploie un profil normal et un champ lointain avec vortex. Le cas
transonique ici n'est pas une reproduction d'AGARD case 6. Contrôler la
présence et la position d'un choc dans la solution obtenue plutôt que les
supposer à partir du Mach total.

## Exécution

Depuis la racine, activer l'environnement Python du projet, par exemple
`source .venv/bin/activate`. Sélectionner `JAX_PLATFORMS=cpu` ou `cuda` selon
la machine. Les commandes suivantes lancent de vrais calculs, potentiellement
longs ; elles ne sont pas des tests rapides :

```bash
bash examples/rae2822_275d_campaign/run.sh subsonic continuity
bash examples/rae2822_275d_campaign/run.sh subsonic depth
bash examples/rae2822_275d_campaign/run.sh subsonic mesh
bash examples/rae2822_275d_campaign/run.sh subsonic relaxation
bash examples/rae2822_275d_campaign/run.sh transonic continuity
```

Les trois autres études acceptent également `transonic`. Chaque calcul est
archivé dans un sous-dossier distinct de `results/rae2822_275d_<regime>_mixed`.
Relancer une même étude réutilise les mêmes noms de sortie : déplacer les
résultats précédents si leur conservation est nécessaire. Le script arrête
sur erreur de commande ; **inspecter aussi la raison d'arrêt de chaque
solveur**, car terminer une commande ne prouve pas sa convergence.

Un cas isolé peut être lancé ainsi :

```bash
python -m jaxrans examples/rae2822_subsonic_conical_verification \
  --set case.domain.sectional.taper_rate=0.03 \
  --out results/rae2822_single_tau003
```

Pour comparer SA-QCR, appliquer à chaque cas comparé
`--set case.turbulence.model=sa_noft2_qcr`, avec un répertoire de sortie
séparé. Les cas restent entièrement turbulents ; aucune transition
expérimentale n'est imposée.

## Études et interprétation

- `continuity` : référence historique 2.5D et sectionnel à tau=0, puis tau
  = -1e-3, -1e-4, -1e-5, +1e-5, +1e-4, +1e-3, .01, .03, .1, .2.
  Ici tau est `-dc/ds`, pas un rapport de cordes. Chaque calcul démarre
  indépendamment du champ uniforme : cette série vérifie les solutions
  finales sans masquer un défaut par un redémarrage voisin. Ce script
  n'implémente pas le transport d'un champ pour une continuation chaude.
- `depth` : profondeurs .2, .1, .05 pour tau=0, .001 et .2.
- `mesh` : trois résolutions globales pour les mêmes tau, avec raffinement
  simultané du profil, du sillage, de la normale et de la première hauteur.
  Les grilles générées ne sont pas strictement emboîtées ; mesurer leur
  qualité et leur résolution effective. Ne pas imposer un ordre de
  Richardson à partir du seul facteur deux dans les paramètres.
- `relaxation` : fenêtres [.5,3], [1,5], [2,8] à tau=.2. C'est une sensibilité
  de fermeture, distincte de l'erreur de maillage.

Les budgets sont de 12000 itérations, tolérance résiduelle 1e-10, arrêt sur
stabilisation des forces désactivé. Ce sont des objectifs, pas une garantie.
Un cas atteignant le budget doit être marqué incomplet. Vérifier les résidus
par équation et par volume, les efforts, les états admissibles et le y+.

À tau=0, viser les tolérances du protocole sectionnel : 1e-6 absolu sur les
coefficients, 1e-5 sur Cp, 1e-6 sur Cf. Comparer aussi les champs. Pour tau
non nul, ramener vitesses et tractions dans les axes fixes de l'aile ; comparer
les distributions aux mêmes coordonnées longitudinales x/c et séparer
intrados, extrados et base. Le repère moyen varie avec tau.

Tracer CL, CD pression/frottement, CM, force latérale, Cp, Cf et crossflow
contre tau. Les différences à zéro doivent tendre vers le plancher numérique,
et les pentes `(J(tau)-J(0))/tau` doivent se stabiliser dans la zone résolue.
Ne pas imposer une évolution monotone ni une pente nulle. Une variation
inférieure aux incertitudes de convergence/maillage ne permet pas de conclure.
Au voisinage d'un choc, compléter les normes ponctuelles par sa position et
les charges intégrées.

Ces cas ne couvrent pas encore la comparaison à une aile 3D indépendante,
l'aller-retour avec redémarrage transporté, ni l'indépendance complète au
rayon extérieur et à la longueur du sillage. Ils permettent de qualifier la
continuité numérique ; la validité physique exige ces vérifications
complémentaires. Voir [l'analyse](../../ANALYSE_275D.md).

## Lancement depuis le dossier de campagne et diagnostic

Depuis ce dossier, utiliser `bash ./run.sh subsonic continuity`.
Ne pas utiliser `. ./run.sh` ni `source ./run.sh` : le script refuse désormais
ce mode sans modifier les options ni fermer le shell appelant.
La racine du dépôt est détectée automatiquement et `.venv/bin/python` est
utilisé s'il existe. `JAXRANS_PYTHON` permet de choisir un autre interpréteur.
Chaque lancement conserve stdout et stderr dans
`results/rae2822_275d_<regime>_mixed/logs/` et affiche le chemin du journal.
Les arguments supplémentaires sont transmis au solveur, par exemple :

```bash
JAX_PLATFORMS=cpu bash ./run.sh subsonic continuity --check
```

Cette commande vérifie et dessine les maillages sans lancer les itérations CFD.
