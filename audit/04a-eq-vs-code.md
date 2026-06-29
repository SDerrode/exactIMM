# Vague 4a — Algorithmes du papier ↔ implémentation

Workflow `audit-4a-find` (run `wf_89cd0f89-6ee`) + vérif `verify-batch` (runs `wf_be8740bb-112`, `wf_b2d4b824-79b`). 3 finders + vérification adversariale (2 lentilles MATHS/CONTEXTE pour high/medium, 1 pour low/info). Détails : `raw/04a-extracted.json`.

**Bilan : 28 trouvailles — 28 confirmées** (0 critical, 2 high, 12 medium, 10 low, 4 info), 0 incertaines, 0 réfutées.

## Trouvailles majeures (critical + high confirmées)

- **[HIGH] Le noyau de la Prop. 2 (eq:mu_jk) omet le terme de biais C_k(b_X,j − Δ_j Σ_V,j⁻¹ b_Y,j) ; le code h5_exact (correct) ne « matche » donc PAS μ_jk comme l'affirme la Remark h5-exact** — `paper/sections/02_model_h5.tex:108-130 (et 03_filtering.tex:56-59)`
- **[HIGH] La Remark « posterior regime moments » (bouclage par les moments a posteriori) n'est pas implémentée dans le mode imm_general (mode par défaut)** — `prg/filter/gss_filter.py:850-874 (vs paper/sections/03_filtering.tex:285-302)`

## Section 5 + annexe D ↔ prg/learning/

_Comparaison point par point de la Section 5 (05_estimation.tex, Algorithmes 1 et 2) et de l'Annexe D (D_baum_welch.tex) avec prg/learning/supervised.py et prg/learning/semi_supervised.py, plus vérification croisée des hyperparamètres du protocole §6.4/§7 contre prg/experiments/run_em.py et run_real_data.py. Conformes (non rapportés) : E-step log-domain (log-MVN Cholesky eq:log_mvn, forward/backward logsumexp, terme d'émission initiale ι, formules γ/ξ), mises à jour M-step de P et π₀ (eq:M_step_P), OLS pondéré par le √w-trick et W_k=diag(γ) (eq:M_step_Theta, eq:sqrt_w), covariance pondérée /Σγ (eq:Sigma_W_weighted), moments initiaux pondérés, reset identité/unité des dynamiques dégénérées dans _weighted_fit, k-means sur les différences premières ΔZ, tri canonique par A_k[0,0] décroissant, timing de contrainte PH (défaut) vs GEM conforme à §5.2.4 et au flag --constraint-each-iter, hyperparamètres §7 = run_real_data.py. Exclus car déjà actés : eq:S_jk/imm_general, convention de moments Remark 4, projection AB post-hoc par défaut, sélection multi-restart sur vraisemblances pré-projection, π₀ EM jeté à la sauvegarde. Dix écarts rapportés : π₀ supervisé jamais estimé malgré l'étape 3 de l'Algorithme 1 ; initialisation EM ≠ « Run Algorithm 1 » ; clamp SPD de la Σ_W jointe annoncé en annexe D mais limité aux blocs dans le M-step ; non-reset des moments initiaux dégénérés + planchers incohérents ; protocole §6.4 (n_init=5, I_max=50) contredit par run_em.py (10, 100) et les défauts de l'API ; ordre E-step/test de convergence inversé avec log L périmé à max_iter ; échec silencieux de la projection AB ; option --delta-zero non documentée ; diviseurs de covariance mixtes ; gestion des régimes vides supervisés non décrite._

### ✅ [MEDIUM] Algorithme 1 (supervisé), étape 3 : π̂₀ par fréquences + lissage de Laplace — jamais implémentée dans fit_supervised

`prg/learning/supervised.py:424 (and 534) vs paper/sections/05_estimation.tex:43-44` — statut : confirmed (2 vote(s)) — catégorie : paper-claim-not-implemented

L'Algorithme 1 du papier comporte l'étape « Estimate π̂₀(k) from regime frequencies (with Laplace smoothing) » (05_estimation.tex l.43-44). Le code fit_supervised ne calcule jamais π₀ : il retourne en dur "pi0": None (supervised.py l.424), et le fichier modèle généré écrit « pi0: None → stationary distribution » (l.534). La distribution initiale utilisée en aval est donc la stationnaire de P̂, pas les fréquences empiriques lissées. Distinct de l'acquis « π₀ appris jeté dans le modèle sauvegardé » (qui concerne le π₀ EM, bien estimé puis écarté à la sauvegarde) : ici, côté supervisé, l'estimateur décrit n'existe pas du tout. Ironiquement, la formule exacte du papier (counts+1)/(N+K) est implémentée dans semi_supervised.py l.421-423, mais seulement comme graine d'initialisation EM.

**Preuve :** supervised.py l.424 : '"pi0": None,' dans le dict retourné par fit_supervised ; docstring l.310 : 'pi0 (always None → stationary distribution)'. Papier Alg.1 l.43-44 : 'Estimate π̂₀(k) from regime frequencies (with Laplace smoothing)'.

**Suggestion :** Soit implémenter l'étape 3 de l'Algorithme 1 dans fit_supervised (réutiliser la formule de semi_supervised.py l.421-423), soit retirer l'étape du pseudo-code et dire dans le papier que π₀ est prise stationnaire.

**Ajustement de sévérité (vérificateurs) :** low

### ✅ [MEDIUM] Algorithme 2 : « Run Algorithm 1 on (R⁽⁰⁾, X, Y) » — le code n'appelle pas l'Algorithme 1 mais une variante défensive différente

`prg/learning/semi_supervised.py:376-381, 409-423, 432-447, 453-461 vs paper/sections/05_estimation.tex:166-168` — statut : confirmed (2 vote(s)) — catégorie : algorithm-mismatch

L'Algorithme 2 (05_estimation.tex l.166-168) affirme que chaque redémarrage EM est initialisé en exécutant l'Algorithme 1 supervisé sur l'assignation k-means R⁽⁰⁾. Le code utilise _initialize_params_from_R (semi_supervised.py l.384-463), qui diffère de fit_supervised sur plusieurs points : (1) lignes de P sans support → remplacées par uniforme (l.413-419) alors que fit_supervised lève ValueError (supervised.py l.331-337) ; (2) OLS par régime seulement si idx.size ≥ dim_z+1, sinon repli F=I, b=0, Σ_W = cov globale + 0.1·I (l.443-447) — fit_supervised fait l'OLS dès 1 échantillon et lève ValueError si 0 ; ce repli (cov globale + 0.1·I) ne correspond pas non plus à la convention « unit noise covariance » de l'annexe D ; (3) μ_z0/Σ_z0 de repli = moments globaux (l.459-461) au lieu du zéro/identité de fit_supervised (supervised.py l.403-410) ; (4) en amont, _initialize_kmeans réassigne aléatoirement un échantillon à tout cluster k-means vide (l.376-381) — absent du papier. Notez aussi que l'Algorithme 1, tel qu'implémenté, lèverait ValueError sur les assignations k-means dégénérées, donc l'affirmation du papier n'est pas seulement imprécise : elle est inexécutable telle quelle.

**Preuve :** semi_supervised.py l.438-439 appelle _fit_regime directement (pas fit_supervised), gardé par 'if idx.size >= dim_z + 1' l.436 ; fallback l.444-447 'F = np.eye(dim_z) … SigW = Z_global_cov + 0.1 * np.eye(dim_z)'. Papier Alg.2 l.167-168 : 'Run Algorithm 1 on (R⁽⁰⁾, X, Y) with AB=false'.

**Suggestion :** Décrire dans le papier (ou en annexe D) l'initialisation réellement utilisée (seuil dim_z+1, replis, lissage des lignes de P, réassignation des clusters vides), ou faire pointer le code vers fit_supervised avec la même sémantique d'erreur.

**Ajustement de sévérité (vérificateurs) :** medium → low (idéalisation conventionnelle du pseudo-code sur les cas dégénérés d'un initialiseur EM ; le chemin générique exécute exactement les mathématiques de l'Algorithme 1 via le même code partagé ; une phrase en annexe D suffit)

### ✅ [MEDIUM] Annexe D : Σ_W « symmetrised and clamped to the nearest SPD matrix » — le M-step ne clampe que les blocs Σ_U et Σ_V, jamais la matrice jointe

`prg/learning/semi_supervised.py:313-316 et 244-247 vs paper/appendix/D_baum_welch.tex:80-89` — statut : confirmed (2 vote(s)) — catégorie : paper-claim-not-implemented

L'annexe D (l.86-89) affirme que la covariance pondérée Σ̂_W,k du M-step est « symmetrised and clamped to the nearest SPD matrix ». Dans le code, _weighted_fit symétrise bien Σ_W (l.316) mais le clamp SPD est délégué à _apply_constraints (l.246-247) qui ne clampe que les blocs diagonaux Σ_U et Σ_V séparément ; le bloc croisé Δ est conservé tel quel et la matrice jointe Σ_W = [[Σ_U, Δ],[Δᵀ, Σ_V]] reconstruite en l.572 n'est jamais projetée SPD pendant les itérations EM (le clamp complet n'existe qu'à l'initialisation, l.451). Clamper les blocs séparément ne garantit pas que la matrice jointe soit SPD (Δ peut rester trop grand) ; le code s'en remet alors au repli ridge du Cholesky dans _log_mvn_batch (l.115-118), mécanisme non décrit dans le papier. Noter que l'Algorithme 1 du papier (supervisé) dit, lui, « Clamp Σ_U, Σ_V to nearest SPD » (05_estimation.tex l.56) — c'est l'annexe D qui sur-promet par rapport au code et à l'Algorithme 1.

**Preuve :** semi_supervised.py l.315-316 : 'SigW = (residuals.T @ (w[:, None] * residuals)) / Wsum ; SigW = (SigW + SigW.T) / 2.0' — aucun _nearest_spd sur SigW ; _apply_constraints l.246-247 : 'SU = _nearest_spd(SU) ; SV = _nearest_spd(SV)'. Annexe D l.88-89 : 'which is symmetrised and clamped to the nearest SPD matrix'.

**Suggestion :** Soit clamper la Σ_W jointe dans _weighted_fit (comme à l'init, l.451), soit corriger l'annexe D : « les blocs Σ_U et Σ_V sont clampés SPD séparément ; un ridge de 10⁻⁸ est ajouté si le Cholesky de Σ_W échoue ».

**Ajustement de sévérité (vérificateurs) :** medium → low (mismatch réel mais purement documentaire : la matrice jointe est PSD par construction dans ce chemin de code, le ridge non documenté couvre les cas dégénérés, et aucun résultat du papier n'en dépend) | medium → low : écart réel mais purement documentaire ; la Σ_W estimée est PSD par construction et le clamp bloc-par-bloc préserve la PSD-ité jointe, donc impact pratique quasi nul (cas singuliers déjà couverts par le repli ridge) ; correctif = reformuler une phrase de l'annexe D.

### ✅ [MEDIUM] Protocole §6.4 du papier (I_max=50, n_init=5) contredit par le script run_em.py (max_iter=100, n_inits=10) et par les défauts de la librairie

`prg/experiments/run_em.py:22-24, 70-74 vs paper/sections/06_experiments.tex:294-301, 312-313` — statut : confirmed (2 vote(s)) — catégorie : reproducibility

Le papier (06_experiments.tex l.294-296) déclare : « Algorithm 2 is run with I_max = 50 EM iterations, convergence tolerance ε = 10⁻⁵, and n_init = 5 random restarts », et l.312-313 : « we nevertheless use n_init = 5 in all subsequent EM experiments ». Le script §6.4 (run_em.py) documente et utilise par défaut DEFAULT_N_INITS = 10 (l.72, et docstring l.24 : 'Run PH and GEM, each with n_inits=10 restarts') et DEFAULT_MAX_ITER = 100 (l.73) ; il est invoqué nu ('python -m prg.experiments.run_em') dans README.md l.625 et wiki/Paper-Reproduce.md l.37, donc avec 10 restarts et 100 itérations. Seule ε = 1e-5 concorde (l.74). Défauts secondaires divergents mais contournables par flags : DEFAULT_N_LIST = (500, 2000, 5000) et DEFAULT_N_RUNS = 100 (l.70-71) vs « N ∈ {500, 2000}, 10 MC runs » (papier l.300-301) — l'exemple d'usage l.40 montre les bons flags. Le wiki (l.40 : « × 5 restarts = 360 EM runs ») est cohérent avec le papier mais pas avec le code. À noter : les défauts de l'API fit_semi_supervised (semi_supervised.py l.699-701 : n_inits=10, max_iter=100, tol=1e-5) divergent aussi des valeurs du protocole §6. En revanche §7 (07_real_data.tex l.146-147 : n_init=5, I_max=50, ε=1e-5) correspond exactement à run_real_data.py l.242 (n_inits=5, max_iter=50).

**Preuve :** run_em.py l.72-73 : 'DEFAULT_N_INITS = 10  # EM restarts per trial' / 'DEFAULT_MAX_ITER = 100' ; docstring l.24 : '2. Run PH and GEM, each with n_inits=10 restarts.' ; papier l.295-296 : 'I_max = 50 EM iterations, convergence tolerance ε = 10⁻⁵, and n_init = 5 random restarts'.

**Suggestion :** Soit corriger le protocole §6.4 du papier (10 restarts, 100 itérations max), soit aligner DEFAULT_N_INITS/DEFAULT_MAX_ITER de run_em.py sur 5/50 et régénérer les tables ; mettre wiki/Paper-Reproduce.md en cohérence.

### ✅ [LOW] Annexe D, régimes dégénérés : le papier dit que (μ_z0, Σ_z0) sont réinitialisés ; le code les laisse à leur valeur précédente — et deux planchers différents

`prg/learning/semi_supervised.py:576-584 (et 89, 276, 550-553) vs paper/appendix/D_baum_welch.tex:105-112` — statut : confirmed (1 vote(s)) — catégorie : algorithm-mismatch

L'annexe D (l.110-112) : « The same convention applies to (μ_z0,k, Σ_z0,k) when the regime is empty over the full sequence » — c.-à-d. réinitialisation (convention identité/unité). Dans _em_run, la mise à jour de μ_z0/Σ_z0 est simplement sautée quand Σ_n γ_n(k) ≤ plancher (l.579 : 'if denom_k > _LOG_FLOOR'), conservant les valeurs de l'itération précédente — pas de réinitialisation. De plus le papier parle d'« a numerical floor » unique, alors que le code en utilise deux très différents : floor_w = 1e-12 pour (F, b, Σ_W) dans _weighted_fit (l.276) et _LOG_FLOOR = 1e-300 pour μ_z0/Σ_z0 et le dénominateur de P (l.89, l.545, l.579). Enfin, le repli des lignes de P sans support — conserver la ligne précédente de P (l.550-553) — n'est décrit nulle part dans la section « Degenerate regimes » de l'annexe.

**Preuve :** semi_supervised.py l.576-584 : la branche 'if denom_k > _LOG_FLOOR:' n'a pas de else — aucun reset ; l.89 : '_LOG_FLOOR = 1e-300' ; l.276 : 'floor_w: float = 1e-12'. Annexe D l.110-112 : 'The same convention applies to (μ_z0,k, Σ_z0,k) when the regime is empty'.

**Suggestion :** Aligner : soit réinitialiser μ_z0/Σ_z0 (zéro/identité) quand le régime est vide comme annoncé, soit reformuler l'annexe (« les moments précédents sont conservés ») ; documenter les deux planchers et le repli des lignes de P.

### ✅ [LOW] Critère d'arrêt : ℓ évalué avant le M-step (pas après comme dans l'Algorithme 2) — à max_iter, le log L rapporté/sélectionné ne correspond pas aux paramètres retournés

`prg/learning/semi_supervised.py:512-537 et 779 vs paper/sections/05_estimation.tex:127-129, 169-181` — statut : confirmed (1 vote(s)) — catégorie : algorithm-mismatch

L'Algorithme 2 du papier (l.169-181) calcule ℓ⁽ⁱ⁾ = log p(z|θ⁽ⁱ⁾) APRÈS le M-step i puis teste |ℓ⁽ⁱ⁾−ℓ⁽ⁱ⁻¹⁾| < ε. Le code évalue log_lik pendant le E-step (donc pour les paramètres du M-step précédent) et teste la convergence AVANT le M-step (l.532-537). Conséquence quand max_iter est atteint sans convergence : la boucle se termine juste après un dernier M-step dont la vraisemblance n'est jamais évaluée — les paramètres retournés sont « un M-step en avance » sur info['log_lik'] = log_lik_history[-1]. La sélection du meilleur redémarrage (fit_semi_supervised l.779) et le « Best log L » sauvegardé comparent alors des vraisemblances qui ne correspondent pas aux paramètres retournés (indépendamment de l'écart pré-projection déjà acté). Accessoirement, §5 l.128-129 dit que l'on retient « the restart with the highest converged log-likelihood » : le code retient aussi les runs non convergés (max-iter atteint) dans la compétition (l.775-780), sans filtre sur info['converged'].

**Preuve :** semi_supervised.py l.532-537 : 'if it > 0: delta = log_lik - log_lik_history[-2]; if abs(delta) < tol: converged = True; break' — placé avant le M-step ; le M-step (l.539-584) s'exécute ensuite, y compris à la toute dernière itération. Papier l.177-178 : 'ℓ⁽ⁱ⁾ ← log p(z_{1:N}|θ⁽ⁱ⁾)' après le M-step, puis test.

**Suggestion :** Évaluer la vraisemblance une dernière fois après le M-step final (ou inverser l'ordre E/M-test comme dans l'Algorithme 2), et préciser dans le papier que les runs non convergés participent à la sélection.

**Ajustement de sévérité (vérificateurs) :** none — « low » est approprié

### ✅ [LOW] Projection AB en échec silencieux : le code garde A, B non contraints avec un simple warning, alors que l'Algorithme 2 affirme appliquer la contrainte

`prg/learning/semi_supervised.py:249-259 vs paper/sections/05_estimation.tex:174-176, 184-186` — statut : confirmed (1 vote(s)) — catégorie : algorithm-mismatch

Dans _apply_constraints (semi_supervised.py l.249-259), si compute_AB lève ValueError, le code log un warning et conserve les A, B non contraints — aussi bien en mode GEM (chaque M-step) qu'en projection post-hoc finale. L'Algorithme 2 du papier (l.174-176 et l.184-186) présente l'application de la contrainte AB comme inconditionnelle ; aucun mode d'échec n'est décrit. Le modèle sauvegardé porte pourtant l'en-tête « Constraint : AB » (_generate_model_code, supervised.py l.481), si bien qu'un modèle étiqueté AB peut violer (H5) sans que rien ne l'indique dans le fichier. À noter que le chemin supervisé (_fit_regime, supervised.py l.258-261) n'a pas ce rattrapage : l'exception remonte — comportements incohérents entre les deux estimateurs.

**Preuve :** semi_supervised.py l.253-259 : 'try: A, B = compute_AB(C, D, Dt, SV) except ValueError as exc: _log.warning("AB projection failed in %s: %s — keeping unconstrained A, B.", …)'.

**Suggestion :** Faire échouer (ou re-projeter après réparation de Σ_V) plutôt que de continuer silencieusement ; au minimum, propager l'information d'échec jusqu'au fichier modèle généré, ou documenter ce repli dans le papier.

**Ajustement de sévérité (vérificateurs) :** none — low est approprié

### ✅ [INFO] Option --delta-zero (forcer Δ(k)=0 avant la projection AB) présente dans les deux estimateurs, absente de la §5 et de l'annexe D

`prg/learning/supervised.py:249-251, 603-607 (et semi_supervised.py:53, 830-834) vs paper/sections/05_estimation.tex (absent)` — statut : confirmed (1 vote(s)) — catégorie : undocumented-feature

Les deux estimateurs offrent un prétraitement --delta-zero qui annule le bloc croisé Δ(k) de Σ_W avant la projection AB (supervised.py l.249-251 dans _fit_regime ; semi_supervised.py l.244-245 dans _apply_constraints, exposé CLI l.830-834). Combiné à la projection AB, cela force A = Δ Σ_V⁻¹ C = 0 et B = 0. Ni l'Algorithme 1, ni l'Algorithme 2, ni l'annexe D ne mentionnent cette option (seule une remarque incidente en 04_constraint.tex l.125 évoque le cas Δ = 0). Un lecteur du papier ne peut pas savoir que la chaîne d'estimation publiée possède ce mode, ni qu'il transforme la projection AB en A=B=0.

**Preuve :** supervised.py l.249-251 : '# --- Step 1: enforce Δ = 0 --- if delta_zero: Dt = np.zeros((q, s))' suivi de la projection AB l.258-261 ; aucun \textsc{delta-zero} dans les algorithmes de 05_estimation.tex.

**Suggestion :** Soit documenter l'option (et sa conséquence A=B=0 sous projection AB) dans la §5, soit la retirer des CLI si elle ne sert pas les résultats du papier.

**Ajustement de sévérité (vérificateurs) :** none — 'info' est la calibration correcte (option jamais utilisée dans les expériences du papier, documentée dans le README)

### ✅ [INFO] Diviseurs incohérents : Σ_W,k en MLE (/N_k) mais Σ_z0,k en covariance non biaisée (/(n−1)) — le papier dit « sample covariance » pour les deux

`prg/learning/supervised.py:236 et 400 vs paper/sections/05_estimation.tex:22-23, 52-53, 62-63` — statut : confirmed (1 vote(s)) — catégorie : numerical

Le papier appelle Σ_W,k « the empirical residual covariance » (l.22-23) / « sample covariance of the OLS residuals » (Alg.1 l.52-53), et (μ_z0, Σ_z0) « sample mean/covariance » (Alg.1 l.62-63), sans préciser les diviseurs. Le code utilise deux conventions différentes : Σ_W,k = résidusᵀ·résidus / N_k (MLE, supervised.py l.236) mais Σ_z0,k via np.cov(..., rowvar=False) qui divise par n−1 (non biaisé, l.400 ; idem semi_supervised.py l.430, 458). Aucun des deux n'est faux isolément, mais le papier laisse croire à une convention unique, et la différence est visible sur les petits régimes. (Côté EM, l'annexe D donne explicitement les formules pondérées /Σγ — le code y est conforme.)

**Preuve :** supervised.py l.236 : 'SigW = (residuals.T @ residuals) / N_k' (commentaire docstring l.214 : 'MLE, divided by N_k') vs l.400 : 'cov_k = np.cov(Z_k, rowvar=False)' (ddof=1 par défaut).

**Suggestion :** Préciser les diviseurs dans le papier (MLE /N_k pour Σ_W, /(n−1) pour Σ_z0) ou unifier la convention dans le code.

### ✅ [INFO] Gestion des régimes vides du supervisé non décrite : le code lève ValueError (P, OLS) ou bascule sur zéro/identité (μ_z0, Σ_z0) ; l'Algorithme 1 est présenté comme total

`prg/learning/supervised.py:331-337, 355-360, 402-410 vs paper/sections/05_estimation.tex:33-67` — statut : confirmed (1 vote(s)) — catégorie : algorithm-mismatch

L'Algorithme 1 du papier ne prévoit aucun cas dégénéré : il suppose chaque régime présent comme source et destination. Le code, lui : (1) lève ValueError si un régime n'apparaît jamais comme source de transition (ligne de P inestimable, supervised.py l.331-337) ; (2) lève ValueError si un régime n'apparaît jamais comme destination (OLS impossible, l.355-360) ; (3) pour μ_z0/Σ_z0 avec moins de 2 pas de temps dans le régime, bascule silencieusement sur moyenne nulle / covariance identité (l.402-410) — repli jamais mentionné dans le papier (l'annexe D « Degenerate regimes » ne couvre que l'EM). Ces comportements expliquent aussi pourquoi l'initialisation EM ne peut pas appeler littéralement l'Algorithme 1 (cf. finding dédié).

**Preuve :** supervised.py l.357-359 : 'raise ValueError(f"Regime k={k} never appears as a transition destination — cannot estimate F(k) by OLS.")' ; l.409-410 : 'mu_z0_list.append(np.zeros((dim_z, 1))) ; Sigma_z0_list.append(np.eye(dim_z))'.

**Suggestion :** Une phrase dans §5.1 suffirait : « regimes absent as source/destination raise an error; initial moments default to (0, I) when fewer than two samples are available ».

## Remarques §3 + annexe E ↔ prg/filter/gss_filter.py

_Mission : écarts algorithme-papier ↔ code non couverts par les acquis, sur les remarques opérationnelles de la Section 3 (paper/sections/03_filtering.tex) + Annexe E (paper/appendix/E_joseph.tex) vs prg/filter/gss_filter.py (avec appuis : 02_model_h5.tex, 01_introduction.tex, 06_experiments.tex, prg/classes/GSSParams.py, tests/test_gss_filter.py). Méthode : lecture croisée + deux vérifications numériques exécutées dans le venv du dépôt (comparaison des constantes h5_exact aux formes fermées de la Prop. 2 sur le modèle jouet AB-contraint à biais non nuls ; régression Monte-Carlo sur 400k pas simulés pour trancher papier vs code). Résultats par point demandé : (1) Précalcul stationnaire — l'itération du code fige le noyau renversé à π_∞ (jamais décrit dans le papier, dont la recette via π_n est mal définie) ; stationnarité jamais vérifiée, non-convergence en simple logger.warning ; surtout, la prétention de la Remark h5-exact de retrouver μ_jk de la Prop. 2 est FAUSSE pour b_X ≠ ΔΣ_V⁻¹b_Y : le noyau du papier omet C_k(b_X,j − Δ_jΣ_V,j⁻¹b_Y,j) — le code est correct (MC-vérifié), la Prop. 2 et la Remark « Bias on X is invisible » sont erronées (Γ_jk et rem:Gamma_bias_free confirmés exacts). (2) Joseph — équivalence numérique bien testée (tests/test_gss_filter.py:363-405), mais : défaut joseph=False contre la « recommended implementation » de l'Annexe E ; flag ignoré en imm_general (seul mode où l'argument par-pas s'applique) ; PSD en pratique assurée par _psd_floor (projection spectrale 1e-9), mécanisme absent du papier ; coût « per regime per step » faux (calcul unique à la construction). (3) Complexité — aucun énoncé O(·) explicite dans le papier ; les deux modes sont O(K²) par pas ; l'avantage CPU attribué aux « gains constants » est gonflé par une asymétrie d'implémentation (Cholesky pré-factorisées + _fast_logpdf en h5_exact vs scipy logpdf refactorisé par paire/pas + eigh en imm_general). (4) Initialisation §3.6 — le code substitue (π_∞, μ_∞, Σ_∞) aux (π_0, μ_z0, Σ_z0) du modèle dans LES DEUX modes (params.pi0 jamais lu par le filtre), contredisant §3.6 et vidant de substance la phrase « le mode général reste plus fiable en transitoire » ; de plus la Remark posterior_moments (bouclage par moments a posteriori) n'est pas implémentée en imm_general (moments prédits réinjectés, docstring l'admet). (5) q≠s et biais b_k — propagation des biais (eq:mu_propagate/var_Z/cov_Z_Z, μ_Y^(j,k)) fidèlement implémentée dans les deux modes et générique en (q,s) ; seul écart : le terme de biais manquant du noyau Prop. 2 (finding 1). Bonus : contradiction interne du docstring sur le mode par défaut (annonce h5_exact, réalité imm_general — le mode au CRITICAL acté). Non re-rapportés : eq:S_jk/imm_general, conformité h5_exact↔eq:AB/H5, convention de moments, écarts côté apprentissage._

### ✅ [HIGH] Le noyau de la Prop. 2 (eq:mu_jk) omet le terme de biais C_k(b_X,j − Δ_j Σ_V,j⁻¹ b_Y,j) ; le code h5_exact (correct) ne « matche » donc PAS μ_jk comme l'affirme la Remark h5-exact

`paper/sections/02_model_h5.tex:108-130 (et 03_filtering.tex:56-59)` — statut : confirmed (2 vote(s)) — catégorie : paper-math-error / paper-code-mismatch

La Remark rem:h5exact (03_filtering.tex:56-59) affirme que les constantes du mode h5-exact « match[ent] the explicit expressions Γ_jk and μ_jk(·) of Proposition prop:markov ». Vérification numérique sur le modèle jouet AB-contraint avec b≠0 : Γ(j,k) du code (gss_filter.py:461) coïncide avec eq:Gamma_jk à 1e-17 près, et la pente avec eq:mu_jk à 1e-16 près, MAIS l'ordonnée à l'origine diverge dès que b_X,j ≠ Δ_j Σ_V,j⁻¹ b_Y,j : code (j=1,k=0)=1.6267 vs papier b_Y,k=2.0 ; (j=1,k=1)=−1.1867 vs −1.0. Une régression Monte-Carlo (400k pas simulés, GSSSimulator) donne 1.641 et −1.190 : c'est le CODE qui est exact, le papier qui est faux. Algébriquement, l'intercept exact est b_Y,k + C_k(b_X,j − Δ_j Σ_V,j⁻¹ b_Y,j) ; eq:mu_jk (02_model_h5.tex:109-112) omet le second terme. Par conséquent la Remark rem:bX_invisible (02_model_h5.tex:119-130, « Bias on X is invisible ») est fausse en général : b_X entre dans le noyau de (R,Y) via b_X − ΔΣ_V⁻¹b_Y (et son renvoi vers §3.6 — « only the stationary moments of Z need to be updated » — ne correspond à aucun énoncé de §3.6). rem:Gamma_bias_free (132-138) est en revanche confirmée numériquement. Distinct de l'acquis eq:S_jk : ici c'est la MOYENNE du noyau de la Prop. 2 et deux remarques de §2/§3 qui sont contredites par le code h5_exact, lequel implémente le bon conditionnement gaussien (gss_filter.py:458 mu_Y_jk, :460 M_t, :624 mean_jk).

**Preuve :** paper: eq:mu_jk = b_{Y,k} + (D_k + C_k Δ_j Σ_{V,j}^{-1}) y_n (02_model_h5.tex:109-112) ; rem:bX_invisible « depends on b_{Y,k} … but not on b_{X,k} » (119-130) ; rem:h5exact « matching the explicit expressions Γ_jk and μ_jk(·) » (03_filtering.tex:56-59). code: prg/filter/gss_filter.py:458 « mu_Y_jk = F[q:, :] @ self._mu_z[j] + b_Y », :624 « mean_jk = self._mu_Y_jk[j][k] + self._M_t[j][k] @ (y_prev - self._mu_Y[j]) ». Numérique : intercept code 1.6267/−1.1867 vs papier 2.0/−1.0 ; MC (n=4750 et 234915 transitions) : 1.641/−1.190 ; b_X(1)−K_1·b_Y(1) = −2 − 0.1333·(−1) = −1.8667 = écart/C_k exactement.

**Suggestion :** Corriger eq:mu_jk en μ_jk(y_n) = b_{Y,k} + C_k(b_{X,j} − Δ_j Σ_{V,j}^{-1} b_{Y,j}) + (D_k + C_k Δ_j Σ_{V,j}^{-1}) y_n, reformuler rem:bX_invisible (le biais d'état est visible via b_X − ΔΣ_V⁻¹b_Y), et préciser dans rem:h5exact que seul Γ_jk est retrouvé tel quel.

### ✅ [HIGH] La Remark « posterior regime moments » (bouclage par les moments a posteriori) n'est pas implémentée dans le mode imm_general (mode par défaut)

`prg/filter/gss_filter.py:850-874 (vs paper/sections/03_filtering.tex:285-302)` — statut : confirmed (2 vote(s)) — catégorie : algorithm-mismatch

La Remark rem:posterior_moments (03_filtering.tex:285-302) prescrit de réinjecter dans l'étape (I) du pas suivant les moments A POSTERIORI μ^post(k)=[x̂_{n+1|n+1}^{(k)}; y_{n+1}] et P^post(k), « ensuring that the prediction at time n+2 conditions on the full history ». Dans _update_step_general, les moments stockés pour le pas suivant sont les moments PRÉDITS mu_np1/P_np1 (gss_filter.py:871-874 « self._P_z = P_np1; self._mu = mu_np1 ») : le posterior Kalman calculé lignes 850-865 (E_x_r, Var_x_r) n'est jamais réinjecté et le bloc Y de μ n'est jamais collapsé sur y_{n+1}. Le docstring l'admet à demi-mot (lignes 59-63 : « µ_n(r), P_n(r) are *prior* (observation-free) quantities »). Le mode h5_exact, lui, implémente la remarque en forme fermée (Γ(j,k)=C_k P_post(j) C_kᵀ + Σ_V(k) ligne 461 et moyenne via x̂ filtré ligne 624 — équivalence vérifiée algébriquement). Lien avec l'acquis eq:S_jk : la non-réinjection des moments a posteriori est précisément ce qui empêche imm_general de produire la covariance conditionnelle correcte ; même en corrigeant eq:S_jk, imm_general ne suivrait pas la récursion de §3 sans ce bouclage. Comme imm_general est le mode par défaut du constructeur (ligne 206), l'algorithme effectivement exécuté par défaut ne correspond à aucune version de la récursion de la Section 3.

**Preuve :** paper: « These posterior values (μ^post, P^post) serve as (μ_{n+1}(k), P_{n+1}(k)) in step (I) of the next time step » (03_filtering.tex:297-301). code: gss_filter.py:871-874 stocke mu_np1/P_np1 (prédits, étape I) ; E_x_r/Var_x_r (lignes 850-865) ne sont utilisés que pour la combinaison (IV) ; docstring lignes 59-63.

**Suggestion :** Soit implémenter le bouclage de rem:posterior_moments dans imm_general (μ^post(k) = [e_x; y_new], P^post reconstruit de var_x), soit documenter dans le papier que le mode général n'effectue pas ce bouclage — mais alors la récursion §3 n'a pas d'implémentation hors h5_exact.

**Ajustement de sévérité (vérificateurs) :** Conserver « high » : le mode affecté est le défaut du constructeur et la récursion centrale de §3 n'a pas d'implémentation time-varying. Mitigation partielle (étiquetage honnête « IMM-approx » en §6 et docstring du code) qui pourrait justifier au pire « medium », mais les affirmations de §3 (propagation exacte, fiabilité du mode général en transitoire) restent contredites par le code. | high (maintenue) — medium serait défendable uniquement si l'on crédite l'étiquette « IMM-approx » de §6.2 et le docstring du code comme divulgation partielle, mais la phrase §3 sur la fiabilité en transitoire et l'affirmation §6.2 de correspondance avec les deux modes de la Sec. 3 restent contredites par le code par défaut.

### ✅ [MEDIUM] §3.6 initialise avec (π_0, μ_z0(k), Σ_z0(k)) fournis par le modèle ; le code remplace ces moments par le point fixe stationnaire dans LES DEUX modes

`prg/filter/gss_filter.py:478, 486-488, 542-563, 700-728 (vs paper/sections/03_filtering.tex:330-366)` — statut : confirmed (2 vote(s)) — catégorie : initialization-mismatch

Le papier (§3.6, 03_filtering.tex:333-366) démarre la récursion avec les moments initiaux du modèle : eq:init_post/eq:init_mean/eq:init_var utilisent μ_z0(k), Σ_z0(k), et lignes 361-363 imposent μ_1(k)=μ_z0(k), P_1(k)=Σ_z0(k)+μ_z0μ_z0ᵀ pour la première étape (I). Le code ignore systématiquement params.mu_z0/Sigma_z0/pi0 au profit des limites stationnaires : _init_step_h5 utilise _mu_X/_mu_Y/_S_YY/_K_gain/_P_post stationnaires (gss_filter.py:551-567, commentaire explicite lignes 535-539 « avoids the transient that would arise from an arbitrary user-supplied π_0 / Σ_z0 ») ; _init_step_general fait de même (709-728) et _reset_state seed μ_1/P_1 avec les moments stationnaires (486-488), π initial = π_∞ (478). Les μ_z0/Σ_z0 du modèle ne servent que de graine à l'itération de point fixe (335-336). Démonstration : sur le modèle jouet, μ_z0=0/Σ_z0=I mais le filtre s'initialise avec μ_∞=[4.31, 8.62] (k=0). Conséquence aggravante : l'affirmation du papier « The general (time-varying) mode remains more reliable in transient or non-stationary settings » (03_filtering.tex:61-63) n'a pas de contrepartie dans le code — les deux modes forcent un départ stationnaire, donc aucun mode ne peut représenter un régime initial hors-stationnaire. (Adjacent mais distinct de l'acquis « π₀ appris jeté » côté apprentissage : ici c'est le filtre qui n'utilise jamais params.pi0.)

**Preuve :** paper: « the model provides the prior π_0(k) … together with μ_z0(k) … Σ_z0(k) » (03_filtering.tex:333-338) ; « μ_1(k) = μ_z0(k) and P_1(k) = Σ_z0(k) + μ_z0 μ_z0ᵀ, which feed the first application of Step (I) » (361-364). code: gss_filter.py:478 « self._pi = self._pi_inf.copy() », 486-488, 535-539, 551-567, 709-728 ; GSSParams.pi0 jamais lu par GSSFilter.

**Suggestion :** Soit aligner le code (initialiser avec mu_z0/Sigma_z0/pi0 du modèle, au moins en imm_general), soit ajouter au papier une remarque indiquant que l'implémentation substitue les moments stationnaires à l'initialisation §3.6 — et retirer/qualifier la phrase sur la fiabilité du mode général en régime transitoire.

**Ajustement de sévérité (vérificateurs) :** none — medium confirmé

### ✅ [MEDIUM] Précalcul stationnaire : l'itération implémentée n'est pas celle prescrite par la Remark (renversement figé à π_∞), et les conditions de validité (stationnarité, convergence) ne sont pas vérifiées

`prg/filter/gss_filter.py:298-368 (vs paper/sections/03_filtering.tex:48-64, 90-94)` — statut : confirmed (2 vote(s)) — catégorie : validity-guards / paper-imprecision

(a) La Remark rem:h5exact (03_filtering.tex:52-55) dit de précalculer les limites « by iterating eq:mu_propagate–eq:var_Z to convergence » ; or ces équations dépendent des données via le noyau renversé eq:reverse_P construit sur π_n (03_filtering.tex:90-94), qui ne converge pas ponctuellement — la recette du papier est mal définie telle quelle. Le code itère une version bien définie où le noyau renversé est figé à π_∞ (gss_filter.py:321-327), jamais décrite dans le papier. (b) Conditions de validité : la Remark conditionne le mode au « time-homogeneous and stationary regime » (lignes 50-51) et l'introduction promet « an explicit characterisation of when the two modes agree » (01_introduction.tex:69-72) — jamais fournie (§3 ne donne que la phrase qualitative lignes 59-63). Côté code, seul le résidu (H5) est contrôlé avec RuntimeWarning (gss_filter.py:232-288) ; la stationnarité n'est jamais vérifiée (pas de comparaison μ_z0/Σ_z0 vs point fixe), la non-convergence du point fixe (MAX_ITER=1000, TOL=1e-12) ne déclenche qu'un logger.warning facilement invisible (363-368) — contrairement au RuntimeWarning (H5) — et aucune garde de contraction n'existe : pour des F_k instables l'itération diverge silencieusement (overflow → moments inf/NaN).

**Preuve :** paper: « One may therefore precompute these limits off-line by iterating eq:mu_propagate–eq:var_Z to convergence » (03_filtering.tex:52-55) où eq:reverse_P = π_n(j)P_jk/Σ π_n(j')P_j'k (90-94). code: gss_filter.py:323 « joint_inf = self._pi_inf[:, None] * p.P », 326-327 p_rev stationnaire ; 363-368 logger.warning (pas warnings.warn) en cas de non-convergence ; aucune vérification de stationnarité ni de rayon spectral.

**Suggestion :** Papier : définir explicitement l'itération à point fixe sur le noyau renversé stationnaire (celle du code) et énoncer ses conditions d'existence/convergence ; livrer la « characterisation of when the two modes agree » promise. Code : promouvoir la non-convergence en RuntimeWarning et vérifier la cohérence μ_z0/Σ_z0 vs point fixe quand le mode h5_exact est demandé.

**Ajustement de sévérité (vérificateurs) :** medium -> low

### ✅ [MEDIUM] Forme de Joseph : « recommended implementation » dans le papier mais joseph=False par défaut, flag ignoré en imm_general (le seul mode où elle servirait), et flooring spectral _psd_floor non documenté

`prg/filter/gss_filter.py:205, 217-223, 410-424, 862 (vs paper/appendix/E_joseph.tex:77-86 et 03_filtering.tex:256-283)` — statut : confirmed (2 vote(s)) — catégorie : paper-code-mismatch

L'Annexe E conclut « it is the recommended implementation » (E_joseph.tex:82-86) et la Remark §3 affirme « The implementation switches between the short and Joseph forms via a single boolean flag » (03_filtering.tex:280-282). En réalité : (a) le défaut est joseph=False — la forme courte (gss_filter.py:205, docstring 79-80) ; (b) le flag n'est honoré qu'en h5_exact ; en imm_general — le mode où la covariance est recalculée à chaque pas, donc le seul où l'argument de stabilité en arithmétique finie de l'Annexe E s'applique réellement — il est ignoré avec un simple logger.warning (217-223) et la mise à jour utilise la forme courte par pas (862) ; (c) la PSD y est garantie par un mécanisme absent du papier : projection spectrale _psd_floor avec plancher 1e-9 sur les valeurs propres (1108-1119), appliquée aussi PAR-DESSUS la forme de Joseph en h5_exact (418-420), rendant Joseph largement redondante dans le code. L'équivalence numérique Joseph/courte EST testée (tests/test_gss_filter.py:363-405), mais uniquement dans le cadre h5_exact à covariance constante.

**Preuve :** paper: « The Joseph form … is the recommended implementation » (E_joseph.tex:82-86) ; « switches between the short and Joseph forms via a single boolean flag » (03_filtering.tex:280-282). code: gss_filter.py:205 « joseph: bool = False », 217-223 « joseph=True has no effect in mode='imm_general' », 862 « var_x = _psd_floor(_sym(S_XX - M_r @ S_XY.T)) », 418-420 _psd_floor appliqué au résultat Joseph.

**Suggestion :** Soit faire de Joseph le défaut et l'implémenter aussi dans la mise à jour par pas d'imm_general, soit reformuler la Remark/Annexe E : préciser que la forme courte est le défaut et que la PSD est en pratique assurée par une projection spectrale (à documenter).

**Ajustement de sévérité (vérificateurs) :** medium -> low-medium | medium -> low

### ✅ [MEDIUM] Docstring du module : « mode=h5_exact (default) » et exemple « (H5)-exact :: GSSFilter(params) » alors que le défaut du constructeur est imm_general — le mode divergent (acquis eq:S_jk) est sélectionné silencieusement

`prg/filter/gss_filter.py:10, 190-199 vs 202-207` — statut : confirmed (2 vote(s)) — catégorie : documentation-bug

Le docstring du module annonce « ``mode="h5_exact"`` (default) » (gss_filter.py:10) et l'exemple intitulé « Step-by-step, (H5)-exact:: » construit GSSFilter(params) sans argument avec le commentaire « # default mode » (190-195). Or la signature du constructeur fixe mode="imm_general" (202-207), conformément au docstring de la classe (172-180). Conséquence opérationnelle : un utilisateur suivant l'exemple du module obtient, sans avertissement, le mode imm_general — celui qui implémente fidèlement l'eq:S_jk erronée (divergence confirmée, acquis CRITICAL) — en croyant exécuter le filtre exact (H5) décrit par la Section 3. Écart code-interne mais directement lié à la correspondance papier↔code : il détermine quel algorithme du papier tourne par défaut.

**Preuve :** gss_filter.py:10 « ``mode="h5_exact"`` (default) — exact IMM recursion under hypothesis (H5) » ; :190-192 « Step-by-step, (H5)-exact:: / filt = GSSFilter(params)  # default mode, short form » ; :206 « mode: str = "imm_general" ».

**Suggestion :** Aligner le docstring du module et l'exemple sur le défaut réel (ou changer le défaut en h5_exact, cohérent avec le titre du papier « exactIMM »).

**Ajustement de sévérité (vérificateurs) :** Conserver medium (pas d'ajustement). C'est une incohérence de documentation (docstring de module/exemple vs signature), pas une erreur de logique du code ni de maths du papier — ce qui plaiderait pour low. Mais la conséquence est réelle et active: l'exemple étiqueté « (H5)-exact » oriente l'utilisateur vers le mode imm_general qui implémente l'eq:S_jk confirmée erronée (divergence numérique acquise). Cela rejoint la calibration déjà appliquée dans l'audit pour les incohérences de documentation qui « misguide practitioners between the two implemented modes » (gardées medium, cf. 03-paper-math.md l.97). Medium est cohérent, en bas-milieu de fourchette. Léger bémol de scoping: la dim « filter-remarks-vs-code » suggère une correspondance papier↔code, alors qu'il s'agit d'un écart code-interne (le finding le reconnaît honnêtement) — n'affecte pas la sévérité. | Maintenir medium (haut de la fourchette). Défaut de documentation pur — le code et le docstring de classe sont cohérents, donc pas HIGH ; mais sélection silencieuse du mode divergent (eq:S_jk) en suivant l'exemple-vedette → au-dessus de low.

### ✅ [LOW] Coût annoncé de la forme de Joseph « per regime per step » erroné : dans le code elle est évaluée une seule fois à la construction

`paper/sections/03_filtering.tex:278-280 (et E_joseph.tex:85-86) vs prg/filter/gss_filter.py:410-424` — statut : confirmed (1 vote(s)) — catégorie : complexity-claim

Le papier (deux fois : 03_filtering.tex:278-280 et E_joseph.tex:85-86) chiffre le surcoût de Joseph à « two additional matrix multiplications per regime per step ». Dans le seul mode où Joseph existe (h5_exact), la covariance a posteriori est constante et la forme de Joseph est calculée UNE fois à la construction (gss_filter.py:410-424) : surcoût par pas nul. Dans le mode où un coût « par pas » aurait un sens (imm_general), Joseph n'est pas implémentée (cf. finding dédié). Le cadrage du coût ne correspond donc à aucun des deux modes.

**Preuve :** paper: « at the modest extra cost of two additional matrix multiplications per regime per step » (03_filtering.tex:278-280 ; répété E_joseph.tex:85-86). code: gss_filter.py:410-424 — boucle unique sur k dans _precompute(), aucun calcul de covariance a posteriori dans _update_step_h5 (675 : « Var_x_r.append(self._P_post[k])  # constant! »).

**Suggestion :** Reformuler : « coût unique à la pré-calcul en mode h5-exact ; par pas uniquement si les moments sont propagés en ligne ».

**Ajustement de sévérité (vérificateurs) :** Maintien à low. Reformuler le titre : il ne s'agit pas d'un coût « erroné » (l'énoncé est correct pour la récursion générale time-varying que §3 décrit), mais d'un désalignement code↔papier — Joseph n'est câblé que dans h5_exact (coût par pas nul, pré-calcul unique), pas dans imm_general (le mode qui recalcule réellement la covariance par pas). Le correctif proposé reste pertinent.

### ✅ [LOW] L'avantage de coût attribué par le papier aux « gains constants / no propagation overhead » provient en bonne partie d'une asymétrie d'implémentation (factorisations pré-calculées vs scipy par paire et par pas)

`prg/filter/gss_filter.py:447-467, 632-636, 809, 816-821, 1050-1105 (vs paper/sections/03_filtering.tex:59-61 et 06_experiments.tex:165-168)` — statut : confirmed (1 vote(s)) — catégorie : complexity-claim

Le papier ne formule aucune borne O(·) explicite (aucun O(K²) trouvé dans paper/) ; ses seules affirmations de coût sont qualitatives : h5-exact « numerically cheaper at inference time (no propagation overhead) » (03_filtering.tex:59-61) et « 1.4× faster … owing to its precomputed constant Kalman gains » (06_experiments.tex:165-168). Dans le code, les deux modes font bien une boucle O(K²) par pas sur les paires (j,k) (617-637 et 792-822), mais l'écart de vitesse mesurable vient aussi — et surtout — d'une asymétrie d'implémentation non décrite : h5_exact pré-factorise les K² covariances Γ(j,k) par Cholesky à la construction (_Gamma_prec, 447-467, 1050-1090) et n'évalue par pas qu'un produit matrice-vecteur (_fast_logpdf, 632-636, 1093-1105) ; imm_general appelle scipy.multivariate_normal.logpdf avec refactorisation complète par paire ET par pas (816-821), plus une eigh par paire via _psd_floor (809). Toute comparaison de temps entre modes (et l'attribution causale « owing to its precomputed constant Kalman gains ») confond donc gain algorithmique et artefact d'implémentation.

**Preuve :** code: gss_filter.py:1063-1065 « removes the O(N·K²) eigen-decompositions that multivariate_normal.logpdf would otherwise perform » (l'auteur du code documente lui-même l'asymétrie) ; 816-821 scipy logpdf par (j,k) par pas en imm_general. paper: « no propagation overhead » (03_filtering.tex:60), « owing to its precomputed constant Kalman gains » (06_experiments.tex:167-168) ; aucun énoncé O(K²)/mémoire dans paper/.

**Suggestion :** Si le papier conserve des chiffres de CPU par pas, factoriser aussi les covariances du mode général (elles changent par pas, mais l'appel scipy peut être remplacé par une Cholesky directe) ou signaler l'asymétrie d'implémentation dans le protocole de benchmark.

**Ajustement de sévérité (vérificateurs) :** none — la trouvaille est correctement calibrée à low (caveat de benchmark/attribution, pas une erreur de correction)

## Définitions §2 et §6 ↔ classes & modèles

_Comparaison des définitions du papier (sections 2 et 6) avec les classes et modèles du code : paper/sections/02_model_h5.tex et 06_experiments.tex vs prg/classes/GSSParams.py, prg/classes/GSSSimulator.py, prg/experiments/models_paper.py, prg/experiments/metrics.py (avec vérification des sites d'appel dans run_simulations.py, make_figures.py, gss_filter.py et de la table générée tab_filter_M2M3.tex). Vérifié sans écart : toutes les valeurs numériques de M1/M2/M3 (P de M1/M3, C, D, Σ_U, Δ, Σ_V, biais, A/B dérivés, π_∞ de M1) sont identiques papier↔code et la contrainte AB est satisfaite des deux côtés ; l'ordre régime→état de eq:dynamics (conditionnement sur r_{n+1}, tirage joint (X,Y) via chol(Σ_W)) est fidèlement implémenté par GSSSimulator ; pi0=None→stationnaire cohérent avec les π_∞ du §6 ; la formule dof_h5 est strictement identique à eq:bic_d (incl. d_H5(2,1,1)=17) ; la définition NEES (normalisation 1/(qN)) est identique. Écarts non couverts par les acquis : RMSE normalisé par q dans le code mais pas dans la définition imprimée (valeurs M2 publiées /√2), initialisation simulée N(0,I) vs marginales stationnaires du modèle/filtre, conditionnement z_n absent de la 2e condition (H3) imprimée, comparateur de benchmark décrit comme IMM de Blom 1988 mais exécuté en mode imm_general, convention min-p Ljung–Box multivariée non documentée, chronométrage CPU incluant la simulation, tolérance H5 1e-10 absolue vs 1e-8 relative, P de M2 absente du papier, BIC plug-in vs ℓ* maximisée et surcomptage des ddl de biais pour M2/M3, et impossibilité K=1 (GSSParams exige K≥2)._

### ✅ [MEDIUM] RMSE: paper definition non normalisée par q, code divise par N·q — valeurs M2 publiées plus petites d'un facteur √2

`paper/sections/06_experiments.tex:144-145` — statut : confirmed (2 vote(s)) — catégorie : metrics

Le papier définit RMSE = sqrt((1/N)·Σ‖x̂−x‖²) (06_experiments.tex l.144-145). Le code calcule sqrt((1/(N·q))·Σ‖x̂−x‖²) (prg/experiments/metrics.py l.94 et l.114, docstring explicite « normalised by the state dimension q »). Identique pour M1/M3 (q=1), mais pour M2 (q=2) les valeurs publiées 0.4595/0.4616 (figures/generated/tab_filter_M2M3.tex l.13, produites via run_simulations.py l.183 → compute_rmse) valent 1/√2 de la définition imprimée : selon la formule du papier on devrait lire ≈0.650/0.653. La comparaison ordinale H5-exact < IMM reste valide, mais les nombres ne correspondent pas à la définition affichée.

**Preuve :** Papier: « RMSE: sqrt(1/N · Σ_{n=1}^{N} ‖x̂_{n|n} − x_n‖²) » (06_experiments.tex:144-145). Code: « RMSE = sqrt( (1 / (N·q)) · Σ ... ) » et « return float(np.sqrt(np.sum((x_true - x_est) ** 2) / (N * q))) » (prg/experiments/metrics.py:94,114). Table générée: « M2 & 0.4595 & ... & 0.4616 » (paper/figures/generated/tab_filter_M2M3.tex:13).

**Suggestion :** Soit ajouter la normalisation 1/(Nq) dans la définition du papier (RMSE par dimension), soit retirer le facteur q dans compute_rmse et régénérer tab_filter_M2M3 / le texte §6.2.4.

**Ajustement de sévérité (vérificateurs) :** none — medium confirmé | none — medium confirmée

### ✅ [MEDIUM] Loi initiale de Z : papier = marginales stationnaires par régime, simulation = N(0, I) pour tous les régimes (non documenté en §6)

`paper/sections/02_model_h5.tex:84-91` — statut : confirmed (2 vote(s)) — catégorie : model-definition

Le papier présente le triplet comme « stationary in the form of its marginals » (Remark 1, 02_model_h5.tex l.84-91) et le filtre H5-exact est initialisé aux moments stationnaires (μ_∞(k), P_∞(k)) (06_experiments.tex l.127-130 ; gss_filter.py précalcule ces moments « so that the initial step can start at the stationary distribution »). Or les trois modèles de référence simulent Z_0|R_0=k ~ N(0, I_{q+s}) pour tout k (prg/experiments/models_paper.py l.99-100 M1, l.169-170 M2, l.233-234 M3 ; consommé par GSSSimulator.__next__ l.131-137). Les trajectoires démarrent donc hors stationnarité alors que le modèle du §2 et l'initialisation du filtre supposent les marginales stationnaires — le transitoire entre dans NEES, LB et RMSE. De plus §6 ne documente nulle part la loi initiale utilisée pour la simulation (μ_z0, Σ_z0). Seule la partie R_0 est cohérente : pi0=None → distribution stationnaire de P (GSSParams.py l.104-106), conforme aux π_∞ cités pour M1.

**Preuve :** Papier: « the triplet (X,R,Y) is stationary in the form of its marginals » (02_model_h5.tex:84-91) ; « the regime moments (μ_∞(k), P_∞(k)) are precomputed once by iterating ... to stationarity » (06_experiments.tex:127-130). Code: « mu_z0 = [np.zeros((dim_z, 1)) ...] ; Sig_z0 = [np.eye(dim_z) ...] » (prg/experiments/models_paper.py:99-100,169-170,233-234) ; tirage initial GSSSimulator.py:131-137.

**Suggestion :** Initialiser la simulation aux moments stationnaires par régime (mu_z0=μ_∞(k), Sigma_z0=P_∞(k)) ou documenter explicitement dans §6 le démarrage N(0,I) et son transitoire.

**Ajustement de sévérité (vérificateurs) :** medium -> low | medium -> low (closer to info/editorial): real notational omission, but NOT a substantive model-vs-generator divergence

### ✅ [MEDIUM] (H3), 2e condition : conditionnement sur (x_n, y_n) absent dans le papier — le simulateur implémente la version conditionnelle en z_n

`paper/sections/02_model_h5.tex:21-26` — statut : confirmed (2 vote(s)) — catégorie : model-definition

La seconde égalité de (H3) est imprimée « p(x_{n+1}, y_{n+1} | r_n, r_{n+1}) = p(x_{n+1}, y_{n+1} | r_{n+1}) » (02_model_h5.tex l.23-25), sans (x_n, y_n) dans les conditionnements. Telle qu'imprimée c'est une contrainte marginale sur la paire de régimes que le processus simulé NE satisfait PAS : sous eq:dynamics, p(z_{n+1}|r_n=j, r_{n+1}=k) = N(F_k μ_n(j)+b_k, F_k P_n(j)F_k^T+Σ_{W,k}) dépend de j en général (même à stationnarité, F_k μ_∞(j) varie avec j). Le code implémente la version vraisemblablement voulue, conditionnelle en z_n : GSSSimulator tire z_n ~ N(F(r_n) z_{n-1} + b(r_n), Σ_W(r_n)), c.-à-d. p(z_{n+1}|z_n, r_n, r_{n+1}) = p(z_{n+1}|z_n, r_{n+1}) (GSSSimulator.py l.139-144), cohérente avec eq:dynamics (02_model_h5.tex l.58-62). Quelle que soit la lecture, l'hypothèse (H3) imprimée et le générateur de données divergent. Distinct de l'écart eq:S_jk déjà acté (qui est en §3).

**Preuve :** Papier: « (H3) ... p(x_{n+1}, y_{n+1} | r_n, r_{n+1}) = p(x_{n+1}, y_{n+1} | r_{n+1}) » (02_model_h5.tex:23-25). Code: « z_n = F @ self._z_prev + params.b(r_n) + L @ noise » avec F = F(r_n), L = chol_W(r_n) (prg/classes/GSSSimulator.py:139-144).

**Suggestion :** Corriger (H3) en p(x_{n+1},y_{n+1} | x_n, r_n, y_n, r_{n+1}) = p(x_{n+1},y_{n+1} | x_n, y_n, r_{n+1}) (ajout de z_n des deux côtés), ou justifier la version marginale et vérifier qu'elle est imposée par construction.

**Ajustement de sévérité (vérificateurs) :** medium -> low | medium -> low

### ✅ [MEDIUM] Comparateur du benchmark : décrit comme « IMM standard de Blom 1988 avec K sous-filtres de Kalman », mais le code exécute le mode imm_general (récursion générale du papier compagnon)

`paper/sections/06_experiments.tex:131-136` — statut : confirmed (2 vote(s)) — catégorie : protocol

§6.2.1 décrit IMM-approx comme « the standard approximate IMM of [blom_interacting_1988], which collapses the per-mode mixture in the state-mixing step. Implemented with the same K Kalman sub-filters and the true B satisfying H5 » (06_experiments.tex l.131-136). Or aucune implémentation de l'IMM de Blom ni de banc de K sous-filtres de Kalman n'existe dans le dépôt : le seul filtre est prg/filter/gss_filter.py et le benchmark exécute mode="imm_general" (run_simulations.py l.64 DEFAULT_MODES=("h5_exact","imm_general"), l.374), que le docstring du code décrit comme la récursion générale du papier compagnon « CS_FinaleBis eqs (17ter), (17quater), (13')–(15), (18), (21')–(22) ... matches exactIMM ≤ v0.9.0 », avec en outre une modification assumée des poids de mélange (postérieur filtré π_n au lieu des marginales a priori du papier, gss_filter.py l.51-63). Les colonnes « IMM-approx » de tab_filter_M1/tab_filter_M2M3 sont donc produites par un algorithme différent de celui annoncé. Écart distinct du CRITICAL déjà acté sur eq:S_jk (ici c'est la description du protocole §6 qui ne correspond pas au comparateur réellement benchmarké).

**Preuve :** Papier: « IMM-approx: the standard approximate IMM of \cite{blom_interacting_1988} ... Implemented with the same K Kalman sub-filters and the true B satisfying H5 » (06_experiments.tex:131-136). Code: DEFAULT_MODES = ("h5_exact", "imm_general") (prg/experiments/run_simulations.py:64) ; « mode="imm_general" ... following the recursion of the companion paper CS_FinaleBis eqs (17ter), (17quater)... This implementation instead uses the filtered posterior π_n ... rather than the unconditional marginals of the paper » (prg/filter/gss_filter.py:44-63).

**Suggestion :** Soit implémenter réellement l'IMM de Blom (mixing + K sous-filtres de Kalman) comme comparateur, soit réécrire §6.2.1 pour décrire fidèlement le mode imm_general effectivement benchmarké (y compris sa pondération par le postérieur filtré).

**Ajustement de sévérité (vérificateurs) :** medium -> low

### ✅ [LOW] Ljung–Box multivarié : le code prend le min des p-values par composante (niveau effectif ≈0.0975 pour M2), convention absente du papier

`paper/sections/06_experiments.tex:150-152` — statut : confirmed (1 vote(s)) — catégorie : metrics

Le papier définit « LB pass (%) : fraction of runs where the Ljung–Box whiteness test (lag 20) does not reject at level 0.05 » (06_experiments.tex l.150-152) sans préciser le cas s>1. Le code applique le test composante par composante et retourne le MINIMUM des p-values (metrics.py l.209-225), puis make_figures compte (lb_pval > 0.05) (make_figures.py l.223). Pour M2 (s=2), rejeter si l'une des deux composantes rejette à 5% donne un niveau par run ≈ 1−0.95² ≈ 0.0975, pas 0.05 — ce qui explique en partie le LB% nettement plus bas de M2 (69–70%, tab_filter_M2M3.tex l.13) vs ≥90% pour M1/M3. Par ailleurs le code ne retient que la p-value au dernier lag (metrics.py l.217-219), cohérent avec « lag 20 » mais non explicité.

**Preuve :** Papier: « fraction of runs where the Ljung–Box whiteness test (lag~20) does not reject the null ... at level 0.05 » (06_experiments.tex:150-152). Code: « the test is applied to each component independently and the *minimum* p-value is returned » + min_pval = min(min_pval, pval) (prg/experiments/metrics.py:191-194,223) ; lb_pass = (g["lb_pval"] > 0.05).mean() (prg/experiments/make_figures.py:223).

**Suggestion :** Documenter la convention min-p (ou appliquer une correction de Bonferroni : seuil 0.05/s) dans le papier, et noter que le LB% de M2 n'est pas comparable à celui des modèles scalaires au même niveau nominal.

**Ajustement de sévérité (vérificateurs) :** medium -> low

### ✅ [LOW] CPU (µs/step) : le chronométrage inclut la simulation des données, pas seulement le filtre

`paper/sections/06_experiments.tex:153-154` — statut : confirmed (1 vote(s)) — catégorie : metrics

Le papier définit « CPU (µs/step): wall-clock time per time step » pour les méthodes de filtrage et en tire le facteur 1.4× (06_experiments.tex l.153-154 et l.166-168). Dans le code, t0/cpu_s englobe la boucle complète « for _,_r,x,y in sim: result = filt.step(y) » (run_simulations.py l.162-173) : chaque pas chronométré inclut le tirage aléatoire du simulateur (choice + standard_normal + produits matriciels de GSSSimulator.__next__) et les appends Python, avant division par N dans make_figures.py l.223-225. Les µs/step absolus (309/425 etc.) sont donc surestimés et le ratio H5-exact vs IMM (1.4×) est dilué par cet overhead commun aux deux modes.

**Preuve :** Papier: « CPU (µs/step): wall-clock time per time step, averaged over 100 runs » (06_experiments.tex:153-154). Code: t0 = time.perf_counter() avant la boucle simulation+filtre ; cpu_s = time.perf_counter() - t0 (prg/experiments/run_simulations.py:162-173) ; cpu_us = cpu_s / N * 1e6 (prg/experiments/make_figures.py:225).

**Suggestion :** Chronométrer uniquement filt.step(y) (accumulateur autour de l'appel) ou pré-générer les données avant le chronométrage, puis régénérer les colonnes CPU.

### ✅ [LOW] Résidu H5 des modèles : papier = norme absolue < 1e-10, code = résidu RELATIF < 1e-8

`paper/sections/06_experiments.tex:17-19` — statut : confirmed (1 vote(s)) — catégorie : model-definition

Le papier affirme que les trois modèles satisfont (H5) avec « H5 residual ‖Δ^T A^T + Σ_V B^T − PM^{-1}W‖_F < 10^{-10} » (06_experiments.tex l.17-19) — une norme de Frobenius absolue avec seuil 1e-10. Le code vérifie autre chose : un résidu relatif normalisé par max(‖Δᵀ Aᵀ + Σ_V Bᵀ‖_F, 1) avec _H5_ASSERT_TOL = 1e-8 (models_paper.py l.41 et l.44-52). En pratique le résidu est en précision machine donc les deux tiennent, mais la garantie imprimée (statistique et seuil) ne correspond pas au contrôle réellement appliqué.

**Preuve :** Papier: « H5 residual ‖Δ^T A^T + Σ_V B^T - PM^{-1}W‖_F < 10^{-10} » (06_experiments.tex:17-19). Code: _H5_ASSERT_TOL = 1e-8 ; rel = ‖res‖_F / max(‖Δᵀ Aᵀ + Σ_V Bᵀ‖_F, 1) ; assert rel < _H5_ASSERT_TOL (prg/experiments/models_paper.py:41,44-52).

**Suggestion :** Aligner le papier sur le contrôle effectif (résidu relatif, seuil 1e-8) ou resserrer l'assertion du code en absolu 1e-10.

### ✅ [LOW] M2 : la matrice de transition P n'est jamais donnée dans le papier (le code utilise celle de M1)

`paper/sections/06_experiments.tex:47-86` — statut : confirmed (1 vote(s)) — catégorie : reproducibility

Le paragraphe M2 et la Table 1 (06_experiments.tex l.47-86) listent les cinq blocs libres par régime mais ne donnent jamais P pour M2, alors que P est imprimée pour M1 (l.24-26) et M3 (l.91-95). Le code utilise P = [[0.97, 0.03], [0.02, 0.98]] — la même que M1 (models_paper.py l.137). Toutes les autres valeurs numériques de M1/M2/M3 (C, D, Σ_U, Δ, Σ_V, biais, P de M1 et M3, et les valeurs AB dérivées de M1 : A=0.10/≈0.013, B=0.35/0.08) sont identiques des deux côtés et la contrainte AB est vérifiée des deux côtés. Lacune de reproductibilité plutôt que contradiction de valeurs.

**Preuve :** Papier: le paragraphe « Model M2 » (06_experiments.tex:47-56) et tab:M2_params (l.58-86) ne contiennent aucune matrice P. Code: P = np.array([[0.97, 0.03], [0.02, 0.98]]) (prg/experiments/models_paper.py:137).

**Suggestion :** Ajouter P (et idéalement la loi initiale de Z) au paragraphe M2 du papier.

### ✅ [LOW] BIC : ℓ*_K défini comme log-vraisemblance MAXIMISÉE mais calculé avec les paramètres VRAIS ; dof compte le biais pour M2/M3 où b est fixé à 0

`paper/sections/06_experiments.tex:381-388` — statut : confirmed (1 vote(s)) — catégorie : bic

Deux écarts distincts de l'ambiguïté π₀ déjà actée. (a) Le papier définit BIC(K) = d·log N − 2·ℓ*_K « where ℓ*_K is the maximised log-likelihood under the K-regime H5-constrained model » (06_experiments.tex l.381-384), mais le BIC̄₂ = 1846.8 rapporté (l.386-388) provient de run_one_trial où compute_bic reçoit la log-vraisemblance du filtre exécuté avec les paramètres VRAIS — aucun ajustement (run_simulations.py l.160,171,187) — tout en pénalisant d_H5 = 17 paramètres jamais estimés. (b) compute_bic est appelé sans condition pour M2 et M3 ; dof_h5 compte K·(q+s) ddl de biais (metrics.py l.81-82) alors que M2 est défini avec b ≡ 0 (« not free », models_paper.py l.168 ; 06_experiments.tex l.55-56) et que M3 fixe b^(2) = 0 (models_paper.py l.230-231) : pour M2 le dof serait surcompté de 8 si ces BIC étaient publiés. La formule dof_h5 elle-même (metrics.py l.80-82) est par ailleurs strictement identique à eq:bic_d, y compris l'exemple d_H5(2,1,1)=17.

**Preuve :** Papier: « BIC(K) = d_H5(K,q,s) log N − 2 ℓ*_K, where ℓ*_K is the maximised log-likelihood » (06_experiments.tex:381-384) ; « mean BIC of 1846.8 » (l.386-388). Code: log_lik_total accumulé avec les paramètres vrais puis bic = compute_bic(log_lik_total, N, K, q, s) (prg/experiments/run_simulations.py:160-187) ; per_regime = q*s + s**2 + dim_z*(dim_z+1)//2 + dim_z (prg/experiments/metrics.py:81-82) ; M2: b_list = zeros (prg/experiments/models_paper.py:168).

**Suggestion :** Qualifier le BIC du papier de « plug-in/oracle BIC » (ou utiliser une vraisemblance maximisée par EM), et soustraire les ddl des blocs fixés à zéro quand le BIC est calculé pour M2/M3.

### ✅ [INFO] Le protocole BIC évoque des modèles K=1, mais GSSParams rejette K < 2

`paper/sections/06_experiments.tex:389-393` — statut : confirmed (1 vote(s)) — catégorie : protocol

Le papier indique qu'une sélection d'ordre complète exige d'ajuster des modèles H5-GSS avec K = 1, 3, 4 (06_experiments.tex l.389-393, différé à une étude compagnon). La classe GSSParams interdit structurellement K=1 : « K must be an integer >= 2 » (GSSParams.py l.186-187). La référence K=1 du protocole annoncé est donc irréalisable avec le code actuel.

**Preuve :** Papier: « requires fitting H5-GSS models with K = 1, 3, 4 via EM » (06_experiments.tex:389-393). Code: if not (isinstance(K, int) and K >= 2): raise ParamError(f"K must be an integer >= 2 ...") (prg/classes/GSSParams.py:186-187).

**Suggestion :** Autoriser K=1 (cas dégénéré sans chaîne) dans GSSParams ou reformuler le protocole (K ∈ {2,3,4}).

**Ajustement de sévérité (vérificateurs) :** Severity low confirmée et bien calibrée. Aucun ajustement. La discordance définition/calcul est réelle (sous-claim a, corroborée par le commentaire du code) mais sans conséquence théorique et sans nombre M2/M3 publié affecté ; impact = reproductibilité/précision d'une seule valeur dans une sous-section déjà marquée comme différée.

---
_Généré automatiquement ; raisonnements complets dans `audit/raw/04a-extracted.json`._