# Audit complet exactIMM — Rapport de synthèse

Audit du code (`prg/`, `tests/`, `scripts/`) et du papier (`paper/`) réalisé du 2026-06-10 au 2026-06-13
sur le commit `af19b48` (main). Méthode multi-agents : pour chaque périmètre, des agents « finders »
indépendants par dimension, puis **vérification adversariale** de chaque trouvaille par 1 à 2 agents
chargés de la *réfuter* (2 lentilles MATHS + CONTEXTE pour critical/high/medium, 1 pour low/info).
Seules les trouvailles confirmées par tous les vérificateurs sont retenues comme « confirmées ».

**289 trouvailles** instruites → **268 confirmées**, 15 réfutées, 6 incertaines.
Sévérité *effective* (après les ajustements proposés par les vérificateurs) des 268 confirmées :
**3 critical · 29 high · 72 medium · 120 low · 44 info**.

Détail par vague dans `01-code-core.md`, `02-code-periphery.md`, `03-paper-math.md`,
`04a-eq-vs-code.md`, `04b-experiments.md`, `04c-editorial.md` ; données brutes (avec le raisonnement
complet de chaque vérificateur) dans `raw/*-extracted.json` et `raw/all-findings.json`.

---

## Verdict global

**Le résultat scientifique central tient, mais le manuscrit n'est pas soumissible en l'état et le code
a deux défauts sérieux.** Aucune trouvaille n'invalide la contribution de fond (filtre exact sous (H5),
forme close de la contrainte AB) ; mais deux formules de propositions centrales du papier sont fausses,
la chaîne de preuve de la contrainte comporte plusieurs pas erronés, et le mode de filtrage **par défaut**
du code est mathématiquement biaisé.

**Code.** Le cœur exact est solide : le mode `h5_exact`, `compute_AB`/`compute_h5_residual` et le chemin
rapide Cholesky sont corrects et vérifiés numériquement (résidu AB ~3e-17 ; chemin rapide identique à scipy
à 1e-15). Les vrais problèmes sont : (i) le mode **`imm_general`** (défaut du constructeur **et** seul mode
de la CLI) implémente fidèlement l'équation de variance de paire erronée du papier → covariances de paire
souvent non-PSD, écrasées à 1e-9, et divergence du filtre exact (max|Δπ|≈0.25–0.46) même sous (H5) ;
(ii) des garde-fous de robustesse trop faibles (le « test H5 » n'est qu'une condition nécessaire ; toute la
validation de `GSSParams` saute sous `python -O` ; l'état GUI n'est pas purgé entre deux simulations) ;
(iii) une couverture de tests confinée au cas scalaire K=2, q=1, s=1, qui a laissé passer (i).

**Papier.** La contribution est juste sur le fond et le résultat final de la contrainte (A=ΔΣ_V⁻¹C,
B=ΔΣ_V⁻¹D) est correct, mais : deux formules de propositions centrales sont fausses (variance d'innovation
de paire ; noyau de transition qui omet le biais), la dérivation de la contrainte (annexes B et C) contient
plusieurs étapes fausses ou non justifiées même si sa conclusion est exacte, la section filtrage mélange
trois conventions de moments incompatibles, et plusieurs affirmations (« biais sur X invisible »,
« génériquement nécessaire ») sont à corriger. Le PDF compilé est néanmoins numériquement cohérent et la
plupart des chiffres se reproduisent exactement depuis les données committées.

---

## Thèmes majeurs (regroupant les trouvailles critical / high)

### A. Deux formules de propositions centrales du papier sont fausses — CRITICAL
- **A1 — Variance d'innovation de paire, eq. (22)/`eq:S_jk` (`03_filtering.tex:170-187`).** S^{(j,k)} mélange
  une variance *marginale* (moyennée sur le régime précédent) avec une cross-covariance *pair-conditionnelle* :
  les conditionnements sont incompatibles. La formule est imprimée comme exacte, sans dérivation, et contredit
  l'annexe B du papier lui-même. Elle est **fidèlement implémentée** dans `imm_general`
  (`gss_filter.py:793-821`, trouvaille HIGH v1) : Γ(j,k) devient indéfinie sur ~12 % des paires (pire vp ≈ −7),
  est écrasée à 1e-9, et le filtre par défaut diverge du filtre exact même sous (H5). **Correctif validé
  numériquement** : Var(Y_{n+1}|j,k) = [F_k Σ_n(j) F_kᵀ + Σ_W(k)] (blocs Y) — complément de Schur, PSD garanti ;
  avec ce remplacement `imm_general` recolle à `h5_exact` à la précision machine.
- **A2 — Noyau de transition (R,Y), Proposition 1 / `eq:mu_jk` (`02_model_h5.tex:108-130`).** L'expression omet
  le terme de biais C_k(b_{X,j} − Δ_jΣ_{V,j}⁻¹b_{Y,j}). En conséquence la Remarque « le biais sur X est
  invisible » et la phrase correspondante de l'abstract sont **fausses dès que b≠0**. Le code, lui, est correct
  (vérifié Monte-Carlo) : c'est une erreur de papier, mais elle touche une proposition centrale et une
  affirmation de l'abstract.

### B. La chaîne de preuve de la contrainte AB est cassée à plusieurs endroits — HIGH (résultat correct, preuve à réécrire)
La forme close et la **suffisance** sont correctes et vérifiées (résidu ~3e-17). Mais, tels qu'écrits :
Annexe B Step 4 contient trois identités fausses dont une dimensionnellement impossible (`B_h5_derivation.tex:138-159`) ;
Step 1 pose un « noise law » Var(Z_n|r_n=j)=P(j) non justifié par l'annexe A citée (`:22-27`) ;
Step 3 affirme à tort qu'« imposer j=k est suffisant et nécessaire » et ne prouve nulle part AB⇒(H5) pour les
paires j≠k (`:100-104`) ; l'annexe C élimine un facteur de noyau non nul (« équivalence » `eq:H5_in_X`↔`eq:H5_compact`
non établie) et suppose Z(r) indépendant de Σ(r) puis aboutit à une solution qui en dépend (`C_projections.tex:31-67`) ;
et la nécessité « génériquement nécessaire / K·s ≥ q+s » est revendiquée dans abstract/intro/conclusion sans
qu'aucun argument de comptage existe.

### C. Section filtrage : conventions de moments incohérentes — HIGH (exposition)
Trois conventions de conditionnement incompatibles pour (μ_n(k), P_n(k)) — a priori / a posteriori / prédits —
coexistent (`03_filtering.tex:35-38, 166-183, 285-302, 361-363`) ; sous la convention déclarée (posterior y_{1:n}),
Σ_YY≡0 et `eq:M_tilde` inverse une matrice nulle. De plus les macros `\Vark`/`\Covk` impriment des moments **non
centrés** mais sont employées comme covariances centrées (`macros.tex` + §2/§3/annexes B,E), rendant plusieurs
équations littéralement fausses telles qu'imprimées. Le code calcule les bonnes quantités ; l'algorithme reste
reconstructible.

### D. Garde-fous de robustesse du code — HIGH
- Le **résidu (H5) par régime n'est qu'une condition nécessaire** : un résidu nul ne garantit pas (H5) complet
  (`h5_constraint.py:39-102`) — le « test H5 » du code et du papier peut valider un modèle non conforme.
- **Toute la validation structurelle de `GSSParams` est désactivée sous `python -O`** (gardée par `__debug__`,
  `GSSParams.py:88-94`).
- **GUI** : `_SessionState` ne purge pas `filter_pis/E_xs/...` au re-Simulate/Load-CSV → matrice de confusion
  silencieusement fausse (`session_state.py:66-95`) ; et un `QThread` peut être détruit en cours d'exécution à la
  fermeture (crash possible, `main_window.py`).

### E. Couverture de tests trop étroite sur le cœur scientifique — HIGH
Filtre, simulateur et apprentissage testés **exclusivement** en K=2, q=1, s=1 (tout scalaire) ; `imm_general`
n'est jamais comparé à une référence (c'est ce trou qui a laissé passer A1) ; `h5_exact` n'est jamais exercé sur
un modèle satisfaisant réellement (H5) ; la PSD du Σ_W joint après EM/projection AB n'est jamais testée.

### F. Reproductibilité & données réelles — HIGH / MEDIUM
Labels de régime ENSO dérivés de l'**ONI centré** → look-ahead d'un mois + circularité avec l'observable Y
(biaise le test H5 et l'accuracy ; ~22 % des labels changeraient avec un proxy causal), non documenté
(`build_enso_csv.py:104-107`). Protocole EM annoncé (50 iter / 5 restarts) ≠ défauts du script (100 / 10 / N≤5000)
(`run_em.py:69-75`). `fill_placeholders.py` réécrit le `.tex` en place et détruit les `\ph{}` (non idempotent).
Artefacts `results/enso/` périmés et contradictoires (un V2 affiche mse=3.8e234) — mais sans impact sur le PDF.

### G. Hygiène dépôt / bibliographie / CI — HIGH→MEDIUM
Job CI « Security audit » (pip-audit) en échec hebdomadaire depuis ≥3 semaines (`audit.yml`, dû à `hmmlearn`
sans wheel cp314 et jamais importé). Noms d'auteurs mutilés dans la biblio (Dempster, Wu, Blom, Anderson — dus à
un `\,` dans les initiales). Dépendance à un *companion paper* non publié pour des étapes de dérivation
(auto-suffisance). Incohérence auteurs/titre CITATION.cff/README ↔ paper.tex (partiellement expliquée : le titre
de CITATION.cff est celui du companion paper).

---

## Recommandations priorisées

**P0 — à corriger avant toute diffusion du papier ou usage du filtre par défaut**
1. Corriger l'équation de variance de paire A1 dans le papier **et** le code ; rendre `imm_general` non-défaut,
   et soit le corriger (complément de Schur), soit le documenter explicitement comme approché et non exact, et
   exposer `--mode` dans la CLI.
2. Corriger `eq:mu_jk` (réintroduire le terme de biais), la Remarque « biais invisible » et l'abstract (A2).
3. Réécrire la dérivation de la contrainte AB (annexe B Steps 1/3/4, élimination annexe C) ; retirer ou
   réellement prouver la « nécessité générique ». Le résultat final ne change pas.
4. Clarifier que le « test H5 » est une condition **nécessaire** (ou implémenter la vérification complète).

**P1 — qualité / rigueur**
5. Unifier la convention de moments de la §3 et corriger les macros `\Vark`/`\Covk` (C).
6. Étendre les tests : référence pour `imm_general`, modèle (H5) réel pour `h5_exact`, cas K>2 / q>1 / s>1,
   PSD du Σ_W post-EM (E).
7. Documenter le label leakage ENSO ; aligner le protocole EM papier↔script ; rendre `fill_placeholders` idempotent (F).
8. Retirer la garde `__debug__` autour de la validation de `GSSParams` ; purger l'état GUI entre runs.

**P2 — hygiène**
9. Réparer/écarter le job pip-audit ; corriger les `\,` de la biblio ; ajouter un id préprint au companion paper.
10. Nettoyer `results/enso/` (artefacts périmés), les `test*.pdf` et figures orphelines dans `paper/`.
11. Traiter le lot de trouvailles `medium`/`low` d'exposition (RMSE non normalisé par q, LB min-p non documenté,
    BIC « oracle » vs maximisé, listes de blocs incohérentes, etc. — voir checkpoints).

---
## Tableau de bord chiffré

Sévérité **effective** (après ajustements des vérificateurs), trouvailles confirmées par vague :

| Vague | crit | high | med | low | info | confirmées | réfutées | incert. |
|-------|-----:|-----:|----:|----:|-----:|-----------:|---------:|--------:|
| 1 — Code cœur | 0 | 3 | 13 | 19 | 12 | 47 | 1 | 0 |
| 2 — Code périphérie | 0 | 11 | 27 | 36 | 11 | 85 | 4 | 0 |
| 3 — Papier maths | 3 | 13 | 22 | 29 | 9 | 76 | 2 | 2 |
| 4a — Algo↔code | 0 | 2 | 5 | 16 | 5 | 28 | 0 | 0 |
| 4b — Chiffres↔résultats | 0 | 0 | 3 | 10 | 1 | 14 | 3 | 1 |
| 4c — Éditorial/biblio | 0 | 0 | 2 | 10 | 6 | 18 | 5 | 3 |
| **Total** | **3** | **29** | **72** | **120** | **44** | **268** | **15** | **6**|

_« réfutées » = écartées par la vérification adversariale (faux positifs des finders) ; conservées dans les checkpoints détaillés. Quand la sévérité effective diffère de la sévérité d'origine, c'est que les vérificateurs ont recalibré (le plus souvent : artefact hors-papier rétrogradé)._

## Liste des trouvailles CRITICAL et HIGH confirmées (sévérité effective)

### CRITICAL (3)

- **(v3)** Eq. (22) : variance d'innovation de paire S^{(j,k)} fausse — mélange de conditionnements, contredit la Proposition 1 — `paper/sections/03_filtering.tex:170-175 (eq. (22), label eq:S_jk)`
- **(v3)** μ_jk omet le terme de biais C_k(b_{X,j} − Δ_jΣ_{V,j}⁻¹b_{Y,j}) : le noyau (R,Y) du papier est faux dès que b_X ≠ ΔΣ_V⁻¹b_Y — `paper/sections/02_model_h5.tex:109-112 (eq:mu_jk), Proposition 1`
- **(v3)** eq:(S_jk): conditionnements incohérents, présenté comme exact sans dérivation — `paper/sections/03_filtering.tex:170-187 (eq:S_jk, eq:M_tilde); cf. appendix/B_h5_derivation.tex:62-70`

### HIGH (29)

- **(v1)** imm_general : Γ(j,k) construit avec une variance marginale mélangée — fréquemment non-PSD, puis écrasé à 1e-9 — `prg/filter/gss_filter.py:793-821`
- **(v1)** Le résidu (H5) par régime n'est qu'une condition NÉCESSAIRE : résidu nul ≠ (H5) complet (équations de paires croisées non vérifiées) — `prg/utils/h5_constraint.py:39-42, 65-102`
- **(v1)** Toute la validation structurelle est désactivée sous `python -O` (gardée par `__debug__`) — `prg/classes/GSSParams.py:88-94`
- **(v2)** begin_simulation/load_external ne purgent pas filter_E_xs/Var_xs/pis/log_lik → matrice de confusion silencieusement fausse — `prg/gui/session_state.py:66-70, 90-95`
- **(v2)** Le mode imm_general n'est jamais testé contre une référence — le seul test de correction est une comparaison de MSE — `tests/test_gss_filter.py:495-507` _(rétrogradé depuis critical)_
- **(v2)** h5_exact n'est jamais testé sur un modèle qui satisfait (H5) — le filtre 'exact' tourne uniquement hors de son hypothèse — `tests/test_gss_filter.py:362, 419`
- **(v2)** Filtre, simulateur et apprentissage testés exclusivement en K=2, q=1, s=1 — tout est scalaire — `tests/test_gss_filter.py:1-507 (et test_gss_simulator.py, test_supervised.py, test_semi_supervised.py)`
- **(v2)** La PSD du Σ_W joint après EM/projection AB n'est jamais testée ; aucun round-trip apprentissage → GSSParams → filtre — `tests/test_semi_supervised.py:330-335`
- **(v2)** Planchers de covariance EM sans pression de test : famine de régime, Σ_W quasi-singulier, ridge 1e-8 testé pour la seule finitude — `prg/learning/semi_supervised.py:303-316` _(rétrogradé depuis medium)_ _(remonté depuis medium)_
- **(v2)** Les variantes V1/V2/V3 de e3_bw_em.py passent des contraintes 'b'/'a' qui sont des no-op silencieux : les 4 variantes sont devenues identiques — `scripts/e3_bw_em.py:65-70` _(rétrogradé depuis critical)_
- **(v2)** fill_placeholders réécrit le .tex en place et détruit les \ph{} : les chiffres narratifs du papier sont désormais figés et inrafraîchissables — `prg/experiments/fill_placeholders.py:297-304`
- **(v2)** Labels de régime ENSO dérivés de l'ONI : look-ahead d'un mois et circularité avec l'observable Y, biaisant le test (H5) E1 vers la non-rejection — `scripts/build_enso_csv.py:104-107`
- **(v2)** Protocole EM : défauts du script (100 runs, n_inits=10, N jusqu'à 5000) ≠ CSV commis localement (10 runs, n_inits=5, N∈{500,2000}) ≠ em_run.log (30 runs, 3 modèles) — `prg/experiments/run_em.py:69-75`
- **(v2)** Le job 'Security audit' (pip-audit) échoue à chaque exécution hebdomadaire depuis au moins 3 semaines — `.github/workflows/audit.yml:28`
- **(v3)** Eqs (21)-(23) : conditionnement \yn = y_{1:n} incohérent — Cov(Y_{n+1},Y_n|·,y_{1:n}) ≡ 0 et Σ_{YY,n}(j)^{-1} = 0^{-1} sous la boucle de la Remarque 4 — `paper/sections/03_filtering.tex:166-183 (eqs (21)-(23), labels eq:y_pred_jk, eq:S_jk, eq:M_tilde)`
- **(v3)** Preuve du retournement temporel : (H5bis) est utilisée à CHAQUE facteur rétrograde, pas seulement au dernier — l'attribution est inversée — `paper/appendix/A_time_reversal.tex:13-23 (preuve du collapse (25)-(26))`
- **(v3)** eq:H5_in_X n'est PAS équivalente à eq:H5_compact — le passage clé de la preuve de nécessité est invalide — `paper/appendix/C_projections.tex:31-39, eq:H5_in_X` _(rétrogradé depuis critical)_
- **(v3)** Élimination de Σ(r) incohérente : Z(r) « indépendant de Σ(r) » alors que la solution finale Z = Σ_V⁻¹Δᵀ dépend des blocs de Σ(r) — `paper/appendix/C_projections.tex:43-67 (eq:split_a, eq:split_b)`
- **(v3)** Step 4 contient trois identités fausses (dont une dimensionnellement impossible) — la conclusion eq:h5_compact_app reste juste — `paper/appendix/B_h5_derivation.tex:138-159 (Step 4)`
- **(v3)** Le décret « noise law » (Var(Z_n|r_n=j) = P(j)) n'est pas justifié par l'annexe A citée, et la Sec. 4 confond covariance de bruit et covariance marginale par régime — `paper/appendix/B_h5_derivation.tex:22-27 (Step 1) ; aussi 04_constraint.tex l.53-64 et 46-47 (Step 2)`
- **(v3)** « Sufficient and necessary to enforce j=k » est faux tel quel ; et nulle part le papier ne prouve AB ⇒ (H5) pour toutes les paires (j,k) — `paper/appendix/B_h5_derivation.tex:100-104 (Step 3) ; 04_constraint.tex l.81-89 (Prop. 4.1)`
- **(v3)** Remarque 2 (« Bias on X is invisible ») fausse : b_X du régime source entre dans le noyau dès que C ≠ 0 — `paper/sections/02_model_h5.tex:120-130 (Remark rem:bX_invisible)`
- **(v3)** (H3), 2e partie : conditionnement sur (x_n, y_n) manquant — telle qu'écrite, l'hypothèse contredit le modèle qu'elle définit — `paper/sections/02_model_h5.tex:21-25 (eq:H3, seconde équation)`
- **(v3)** Moments de régime (μ_n(k), P_n(k)): trois conventions de conditionnement incompatibles — `paper/sections/03_filtering.tex:35-38, 166-183, 285-302, 361-363`
- **(v3)** \Vark/\Covk impriment des moments NON centrés mais sont utilisés comme covariances centrées — `paper/macros.tex:macros:23-26; 02:152-157; 03:32,103-105; B:62-70,119-127; E:57`
- **(v3)** Step 4 de la dérivation H5 : justifications fausses et équation intermédiaire dimensionnellement invalide — `paper/appendix/B_h5_derivation.tex:136-152 (Step 4)`
- **(v3)** « Équivalence » eq:H5_in_X non établie + condition de nécessité « K·s ≥ q+s » fantôme et revendications incohérentes — `paper/appendix/C_projections.tex:C:31-39; 01:84-85; 04:67-68; abstract:33-34`
- **(v4a)** Le noyau de la Prop. 2 (eq:mu_jk) omet le terme de biais C_k(b_X,j − Δ_j Σ_V,j⁻¹ b_Y,j) ; le code h5_exact (correct) ne « matche » donc PAS μ_jk comme l'affirme la Remark h5-exact — `paper/sections/02_model_h5.tex:108-130 (et 03_filtering.tex:56-59)`
- **(v4a)** La Remark « posterior regime moments » (bouclage par les moments a posteriori) n'est pas implémentée dans le mode imm_general (mode par défaut) — `prg/filter/gss_filter.py:850-874 (vs paper/sections/03_filtering.tex:285-302)`

## Trouvailles MEDIUM confirmées (par vague, titres)

**Vague 1 — Code cœur** (13)
- Docstring de module contradictoire : annonce h5_exact comme mode par défaut alors que le défaut réel est imm_general — `prg/filter/gss_filter.py:10, 190-199 vs 172-175, 206`
- CLI : aucune option --mode/--joseph — le filtre exact h5 est inatteignable, même avec --constraint — `prg/filter/main.py:247 (et 176-181)`
- _psd_floor : plancher absolu eps=1e-9 indépendant de l'échelle — transforme une matrice indéfinie en covariance quasi-singulière — `prg/filter/gss_filter.py:1108-1119`
- Appendice B du papier : l'étape 4 contient des identités fausses et la réduction « j=k nécessaire et suffisant » est incorrecte — `paper/appendix/B_h5_derivation.tex:100-103, 139-152`
- Tolérance de symétrie ABSOLUE (1e-10) non invariante d'échelle : rejette des covariances légitimes à grande échelle, accepte des asymétries grossières à petite échelle — `prg/utils/matrix_checks.py:36, 191-203`
- x_n et y_n retournés sont des VUES de l'état interne _z_prev — une mutation par l'appelant corrompt la trajectoire — `prg/classes/GSSSimulator.py:150-158`
- FMatrix n'effectue aucun contrôle de finitude: NaN/Inf dans A/B/C/D acceptés silencieusement — `prg/classes/FMatrix.py:106-124`
- Longueur de b_list jamais validée (même avec __debug__) — `prg/classes/GSSParams.py:115-118`
- Tous les accesseurs exposent les tableaux internes par référence — mutation externe silencieuse possible après validation — `prg/classes/GSSParams.py:360-394`
- Option --output: le nom de fichier fourni est silencieusement ignoré — `prg/simulate.py:316-323`
- Mise à jour de μ_z0/Σ_z0 non conforme au M-step — la monotonie stricte d'EM annoncée n'est pas garantie — `prg/learning/semi_supervised.py:575-584`
- Effondrement de covariance insuffisamment gardé + sélection du meilleur redémarrage par logL : les runs dégénérés peuvent gagner — `prg/learning/semi_supervised.py:299-315, 326-339, 779`
- Régimes rares : résidus OLS identiquement nuls → Σ_W = 0 clampée à 1e-8·I, modèle sauvegardé pathologique avec simple warning — `prg/learning/supervised.py:228-237, 369-376`

**Vague 2 — Code périphérie** (27)
- QThread détruit pendant qu'il tourne — crash fatal Qt à la fermeture — `prg/gui/main_window.py:728-733, 1855-1858`
- Échap sur _WaitDialog contourne le Cancel : la modalité saute, double-lancement et corruption d'état possibles — `prg/gui/dialogs.py:662-697`
- Annuler un filtrage détruit la simulation : on_cancel=_on_reset efface tout l'état — `prg/gui/main_window.py:840`
- _on_regime_diag utilise self._P obsolète au lieu de la matrice P éditée/capturée — `prg/gui/main_window.py:1883-1889`
- La restauration de session ne purge ni les workers en vol ni _filter_worker.cond_moments périmé — `prg/gui/main_window.py:1356-1498, 1901-1911`
- np.load(allow_pickle=True) sur les fichiers .exactIMM : exécution de code arbitraire à l'ouverture — `prg/gui/main_window.py:1764`
- Load CSV laisse le panneau « Innovation diagnostics » et le bouton d'histogrammes dans l'état du run précédent — `prg/gui/main_window.py:1243-1251, 1744-1751`
- Le worker extrait fouille 6 attributs privés de GSSFilter, avec dégradation silencieuse — `prg/gui/workers.py:105-141`
- StochasticMatrixWidget : la matrice P uniforme par défaut est INVALIDE pour K=3 et K=6 (tolérance 1e-6 vs affichage 6 chiffres significatifs) — `prg/gui/matrix_widget.py:394, 422, 474-475`
- Randomize 🎲 et chargement de preset in-place violent silencieusement la contrainte AB active (blocs déverrouillés + pastille « ✓ » périmée) — `prg/gui/param_panel.py:475-494 (et prg/gui/main_window.py:1143-1145, 2206)`
- Les options -K/-q/-s documentées sont silencieusement écrasées par le preset par défaut — `prg/gui/main.py:63-86`
- Affichage en :.6g = troncature à 6 chiffres significatifs de tous les paramètres chargés — le filtre tourne sur des valeurs ≠ modèle — `prg/gui/matrix_widget.py:146, 422, 582`
- _fast_logpdf / _precompute_gaussian_logpdf jamais validés contre scipy ; fallback eigen mort-né — `prg/filter/gss_filter.py:1050-1105`
- test_empirical_mean_near_zero : prémisse fausse (le modèle n'est pas centré) et passage à 0.0015 de la tolérance — `tests/test_gss_simulator.py:225-236`
- FilterResult.innovation et log_lik ne sont assertés par aucun test — `prg/filter/gss_filter.py:119-146`
- test_missing_destination_regime_raises ne 'raise' rien : il asserte que le fit RÉUSSIT avec K=1 — `tests/test_supervised.py:289-298`
- Diagnostics GUI : assertions tautologiques (0 ≤ p ≤ 1) et mode h5 de _standardise_innovations non testé — `tests/test_main_window_gui.py:38-65`
- Workers jamais exécutés et main_window (2214 lignes) couvert par un seul smoke-test constructeur — `tests/test_main_window_gui.py:90-94, 128-132`
- La CLI du filtre (prg/filter/main.py) n'a aucun test, contrairement aux CLIs d'apprentissage — `prg/filter/main.py:124-190`
- Baseline Hamilton : les prédictions de X utilisent les probabilités LISSÉES (défaut statsmodels) calculées sur train+test → look-ahead, contrairement au commentaire « filtered probs » — `scripts/baselines/hamilton_msar.py:114-115`
- tab_em_basin : la légende décrit « fraction des runs atteignant le bassin avec 5 restarts » mais la valeur affichée est la fraction moyenne de restarts dans le bassin — contradiction avec tab_em_restarts dans le même papier — `prg/experiments/make_figures.py:603-667`
- RMSE : le code normalise par N·q, le papier définit sqrt((1/N)Σ‖·‖²) — les valeurs M2 (q=2) de Table 3 sont 1/√2 de la définition affichée — `prg/experiments/metrics.py:90-114`
- Le « CPU (µs/step) » publié inclut le coût du simulateur et des appends Python, pas seulement le filtre ; la légende code en dur « Apple M2 Pro » — `prg/experiments/run_simulations.py:162-173`
- Producteurs en chemins relatifs au CWD vs consommateurs ancrés au dépôt : risque de CSV orphelins et de figures générées depuis des données périmées — `prg/experiments/run_simulations.py:65`
- Le parcours d'installation documenté (README quick-start et Makefile) est cassé sur machine vierge : pytest plante au démarrage sans PyQt6 — `Makefile:35-45 (et README.md:111-137, 668-671)`
- « make check » prétend la parité CI mais omet « ruff format --check » — ça a déjà cassé la CI — `Makefile:50-59`
- Angles morts CI : les 17 tests GUI ne tournent dans aucun job, et prg/experiments/ + scripts/ n'ont aucun test — `.github/workflows/tests.yml:36-40`

**Vague 3 — Papier maths** (22)
- Argument d'exactitude de l'étape (III) erroné : la prédictive p(X_{n+1}|r_{n+1}=k, y_{1:n}) N'EST PAS « déjà une seule gaussienne » — `paper/sections/03_filtering.tex:217-228 (texte entre eqs (26) et (27))`
- Remarque 2 : le gain stationnaire K^{(k)} et S^{(k)} ne « matchent » pas Γ_{jk} et μ_{jk}(·) — mauvais objets ; la vraie forme fermée (validée par le script) n'est ni énoncée ni prouvée — `paper/sections/03_filtering.tex:48-64 (Remark 2, rem:h5exact)`
- Incohérence de bouclage : Remarque 4 (moments postérieurs) vs init §3.6 (moments a priori) vs code (moments prédits, jamais bouclés) — `paper/sections/03_filtering.tex:285-302 (Remark 4) vs 361-366 (init) vs eqs (21)-(23)`
- Macros \Covk/\Vark définies comme moments NON centrés mais utilisées comme covariances centrées — eq. (17) littéralement fausse sous la définition de la macro — `paper/macros.tex:22-26 ; 03_filtering.tex l.31-32, 103-105, 171-173, 336-337 ; 02_model_h5.tex l.152-157`
- La monotonie EM affirmée n'est pas garantie par le M-step réellement spécifié (moments initiaux, clamp SPD, resets) — `paper/appendix/D_baum_welch.tex (+ paper/sections/05_estimation.tex):App D lignes 80-112 ; 05_estimation.tex lignes 131-140`
- Le mode (ii) est appelé « Generalized EM (Wu 1983) » alors qu'un GEM garantit la monotonie — contradiction interne — `paper/sections/05_estimation.tex:lignes 135-138 (et 08_conclusion.tex l.25)`
- Sélection multi-restart sur la vraisemblance PRÉ-projection : le papier décrit fidèlement le code mais n'avertit jamais que l_hat ne correspond pas au modèle retourné — `paper/sections/05_estimation.tex:Algorithme 2, lignes 183-186 ; ligne 129`
- Prop. 1 : homogénéité non justifiée (loi initiale jamais spécifiée) et appui circulaire sur la « noise law » de l'App. B — `paper/sections/02_model_h5.tex:99-118 (Proposition 1) ; appendix/B_h5_derivation.tex l. 23-26`
- Macro \Vark = E[AAᵀ|·] (moment d'ordre 2) utilisée comme variance centrée : eq:mix_var est fausse telle qu'affichée — `paper/macros.tex:macros.tex l. 25-26 ; 02_model_h5.tex l. 152-157 (eq:mix_var)`
- Preuve A : la factorisation rétrograde n'est PAS obtenue par « chain rule + (H1) » seuls — (H5bis) est utilisé au mauvais endroit — `paper/appendix/A_time_reversal.tex:13-22 (factorisation display l.15-20 et phrase l.21-22)`
- Énoncé faux : « for any matrix K » avec membre de gauche Var[X_{n+1} | r_{n+1}=k, y_{n+1}] — `paper/appendix/E_joseph.tex:53-62 (eq:joseph_proof)`
- Symbole P surchargé avec cinq sens différents (idem R, et trois notations pour Σ_W) — `paper/sections/04_constraint.tex:04:29; 03:87,156; 05:9,25; 06:18,25; B:16-22,110`
- Le problème de filtrage est défini avec y_{1:N} (lissage) au lieu de y_{1:n} — `paper/sections/02_model_h5.tex:144`
- \yn[1:n+1] s'imprime « y_{1:n}[1:n+1] » — `paper/sections/03_filtering.tex:301`
- Résidu H5 « PM^{-1}W » : W jamais défini, et W surchargé (bruit / poids / résidu) — `paper/sections/06_experiments.tex:17-18; appendix/D_baum_welch.tex:75-82`
- b^{(k)} = coefficient de régression sur Y_n en Sec. 7, mais = vecteur de biais en Sec. 6 ; et la même quantité est appelée b^{(k)} puis B^{(k)}/B(k) — `paper/sections/07_real_data.tex:82-94; cf. 06:42,106-108`
- Intercept de eq:mean_X_linear écrit b_{X,k} (faux si b_{Y,k}≠0) alors que la remarque invoque un « intercept α » inexistant — `paper/appendix/B_h5_derivation.tex:51-58 et 165-173`
- Σ_{YY,n+1}(k,j) : ordre des arguments inversé par rapport à toutes les autres quantités de paire — `paper/appendix/B_h5_derivation.tex:62-70`
- X(r), H(r), Z(r) : trois symboles de l'appendice C qui collisionnent avec l'état X_n, l'opérateur H de Joseph et l'état augmenté Z_n — `paper/appendix/C_projections.tex:13-20, 48`
- Le test F de B=0 est présenté comme test de (H5), mais rejeter B=0 ne rejette pas (H5) — `paper/sections/07_real_data.tex:75-77, 98-100`
- Désaccords texte ↔ tables générées (valeurs et colonnes citées absentes) — `paper/sections/06_experiments.tex:06:199,216,228,280-282,350-352; 07:191`
- rem:h5exact : S^{(k)} (indexé k) annoncé « matching » Γ_{jk} (indexé paire) ; π_0 vs π_∞ ; π_0 libre ou dérivée ? — `paper/sections/03_filtering.tex:48-64; cf. 06:44, 03:335, 06:376-378`

**Vague 4a — Algo↔code** (5)
- Algorithme 1 (supervisé), étape 3 : π̂₀ par fréquences + lissage de Laplace — jamais implémentée dans fit_supervised — `prg/learning/supervised.py:424 (and 534) vs paper/sections/05_estimation.tex:43-44`
- Protocole §6.4 du papier (I_max=50, n_init=5) contredit par le script run_em.py (max_iter=100, n_inits=10) et par les défauts de la librairie — `prg/experiments/run_em.py:22-24, 70-74 vs paper/sections/06_experiments.tex:294-301, 312-313`
- §3.6 initialise avec (π_0, μ_z0(k), Σ_z0(k)) fournis par le modèle ; le code remplace ces moments par le point fixe stationnaire dans LES DEUX modes — `prg/filter/gss_filter.py:478, 486-488, 542-563, 700-728 (vs paper/sections/03_filtering.tex:330-366)`
- Docstring du module : « mode=h5_exact (default) » et exemple « (H5)-exact :: GSSFilter(params) » alors que le défaut du constructeur est imm_general — le mode divergent (acquis eq:S_jk) est sélectionné silencieusement — `prg/filter/gss_filter.py:10, 190-199 vs 202-207`
- RMSE: paper definition non normalisée par q, code divise par N·q — valeurs M2 publiées plus petites d'un facteur √2 — `paper/sections/06_experiments.tex:144-145`

**Vague 4b — Chiffres↔résultats** (3)
- Stale results/enso/e3_table.tex uses an obsolete EM-variant design (tau=A/B) absent from the paper — `results/enso/e3_table.tex:5-9`
- ENSO filter table calls the test column "joint test log-likelihood" but it is the Y-only predictive density — `prg/experiments/run_real_data.py:385 (emit_e2_tex caption) → rendered in paper/figures/generated/tab_enso_filter.tex line 2, included by paper/sections/07_real_data.tex:117`
- Committed results/enso/ contains stale, contradictory E3 artifacts produced by the S&P/VIX script, not by run_real_data.py — `results/enso/e3_summary.json + results/enso/e3_table.tex:e3_summary.json:1-86; e3_table.tex:1-15`

**Vague 4c — Éditorial/biblio** (2)
- Financial-signal (S&P500/VIX) H5-rejection claim 'p<10^-4' is asserted with no experiment, table, or appendix backing it — `paper/sections/07_real_data.tex:98-100`
- Mangled author rendering in .bbl from thin-space (\,) inside name initials — `paper/paper.bib:117`

## Méthodologie & coût

- **Organisation résiliente** : 5 vagues (cœur code, périphérie code, maths papier, cohérence papier↔code↔résultats en 3 sous-lots, synthèse), chacune checkpointée sur disque (`audit/`) et reprise après chaque coupure de session via `STATUS.md` + journaux de workflow (cache de reprise).
- **Schéma par vague** : finders par dimension (lecture intégrale des fichiers du périmètre) → vérification adversariale de chaque trouvaille (consigne explicite de *réfuter*, lecture du code/`.tex` réel, souvent reproduction numérique dans le venv du projet).
- **Synthèse** : agrégation purement locale (aucun agent), donc insensible aux limites d'usage.
- **Coût** : plusieurs centaines d'exécutions d'agents et de l'ordre de la dizaine de millions de tokens de sous-agents, étalés sur ~3 jours à cause des reprises (voir les `runId` dans `STATUS.md`).

## Fichiers
- `AUDIT_REPORT.md` (ce fichier) · `STATUS.md` (état/reprise)
- Checkpoints lisibles : `01-code-core.md`, `02-code-periphery.md`, `03-paper-math.md`, `04a-eq-vs-code.md`, `04b-experiments.md`, `04c-editorial.md`
- Données brutes : `raw/*-extracted.json` (+ votes), `raw/all-findings.json` (agrégat normalisé)

_Note : les sévérités sont indicatives et issues d'agents ; pour les points P0, le raisonnement complet des vérificateurs (souvent avec reproduction numérique) est dans les `raw/*-extracted.json` correspondants._
