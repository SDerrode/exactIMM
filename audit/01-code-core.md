# Vague 1 — Audit du cœur scientifique (code)

Workflow `audit-code-core` (run `wf_a0c95e2c-68b`), terminé le 2026-06-11.
Méthode : 4 finders (filtre, H5/AB, classes, apprentissage) + vérification adversariale
(2 votes pour high/medium, 1 pour low/info). Détails complets : `raw/01-code-core-result.json`.

**Bilan : 48 trouvailles — 47 confirmées** (3 high, 13 medium, 19 low, 12 info), 1 réfutée(s).

## Synthèse des trouvailles majeures (high)

- **imm_general : Γ(j,k) construit avec une variance marginale mélangée — fréquemment non-PSD, puis écrasé à 1e-9** — `prg/filter/gss_filter.py:793-821`
- **Le résidu (H5) par régime n'est qu'une condition NÉCESSAIRE : résidu nul ≠ (H5) complet (équations de paires croisées non vérifiées)** — `prg/utils/h5_constraint.py:39-42, 65-102`
- **Toute la validation structurelle est désactivée sous `python -O` (gardée par `__debug__`)** — `prg/classes/GSSParams.py:88-94`

## Filtre (prg/filter/)

_Audit complet de prg/filter/gss_filter.py (1136 lignes) et prg/filter/main.py (319 lignes), avec vérification croisée contre prg/utils/h5_constraint.py, prg/classes/GSSParams.py, le papier (sections/02_model_h5.tex, 03_filtering.tex) et des expériences numériques dans le venv du projet (scipy 1.17.1, numpy 2.4.4). Le mode h5_exact est mathématiquement correct et conforme au papier : gains, P_post (formes courte et Joseph équivalentes), moyennes/covariances de paire (M̃ = C_k K_j + D_k, Γ = C_k P_post C_kᵀ + Σ_V), normalisations log-sum-exp avec replis sains ; le chemin rapide Cholesky est exactement équivalent à scipy (1.8e-15 end-to-end) et n'est utilisé que là où Γ est time-invariant. Le problème principal est dans le mode imm_general (défaut, et seul mode accessible via la CLI) : la covariance prédictive de paire mélange la variance marginale du mélange régime-k avec la cross-covariance pair-(j,k), produisant des Γ indéfinis (12.5% des paires sur le modèle contrast, pire vp −6.9) que _psd_floor écrase à 1e-9 en gaussiennes quasi-dégénérées — d'où des π pouvant diverger de 0.46 du filtre exact même sous (H5). S'y ajoutent des incohérences de documentation (le docstring de module annonce h5_exact comme défaut), l'absence d'option --mode dans la CLI (rendant --constraint + filtre exact impossible), et des validations d'entrée manquantes (longueur de y, NaN ; observations manquantes non gérées). Santé globale : cœur h5 solide et bien testé ; mode général numériquement fragile sur modèles à covariances contrastées._

### ✅ [HIGH] imm_general : Γ(j,k) construit avec une variance marginale mélangée — fréquemment non-PSD, puis écrasé à 1e-9

`prg/filter/gss_filter.py:793-821` — statut : confirmed (2 vote(s)) — catégorie : math

Dans _update_step_general, la covariance prédictive de paire est Γ = S_YY_np1 − M_t·Covᵀ où S_YY_np1 = Var(Y_{n+1} | r_{n+1}=k) est la variance du MÉLANGE sur r_n (incluant le terme de dispersion des moyennes inter-régimes), tandis que Cov_Ynp1_Yn = (F_k Σ_n(j))[q:,q:] est conditionnelle à la paire (j,k). Le conditionnement gaussien exige les trois moments sous le MÊME conditionnement ; la formule cohérente est Var(Y_{n+1}|j,k) = (F_k Σ_n(j) F_kᵀ + Σ_W(k))[q:,q:], qui rend Γ un complément de Schur, PSD par construction. La version implémentée (fidèle à l'éq. (S_jk) du papier, sections/03_filtering.tex) produit des Γ indéfinis : mesuré sur model_gss_K2_q1_s1_contrast, 1001/7996 paires (j,k) ont une valeur propre négative avant le floor (pire : −6.94). _psd_floor les écrase alors à 1e-9, créant une gaussienne quasi-dégénérée dont la log-vraisemblance explose (log-normalisation ≈ +10 par direction écrasée ; scipy allow_singular garde la direction 1e-9 car son cutoff ≈ 2e-10·max), ce qui distord brutalement la mise à jour de π. C'est exactement le biais systématique que le commentaire des lignes 439-443 documente et corrige... uniquement pour h5_exact. Conséquence mesurée : même sous (H5) (modèle contrast + apply_AB_constraint), imm_general diverge de h5_exact avec max|Δπ| = 0.46 et −7.3 nats de log-vraisemblance sur 1000 pas. imm_general est le mode PAR DÉFAUT, le seul accessible depuis la CLI (main.py:247) et celui utilisé par scripts/e3_bw_em.py:113.

**Preuve :** L809: Gamma = _psd_floor(_sym(S_YY_np1 - M_t @ Cov_Ynp1_Yn.T)) avec S_YY_np1 = Sig_np1[q:, q:] (mélange sur r_n, L793-794) mais Cov_Ynp1_Yn = (F @ Sig_n)[q:, q:] (conditionnel à r_n=j, L807). Expérience : 1001/7996 Γ pré-floor indéfinis (pire vp −6.94) sur model_gss_K2_q1_s1_contrast ; max|π_h5 − π_general| = 0.464 sous (H5).

**Suggestion :** Remplacer S_YY_np1 par la variance pair-conditionnelle (F_k Σ_n(j) F_kᵀ + Σ_W(rnp1))[q:, q:] dans le calcul de Γ (complément de Schur ⇒ PSD garanti, plus de dépendance au _psd_floor). Si la fidélité au papier prime, corriger aussi l'éq. (S_jk) du papier — le commentaire L439-443 du code démontre déjà que la version mélangée biaise systématiquement un régime.

### ✅ [MEDIUM] Docstring de module contradictoire : annonce h5_exact comme mode par défaut alors que le défaut réel est imm_general

`prg/filter/gss_filter.py:10, 190-199 vs 172-175, 206` — statut : confirmed (2 vote(s)) — catégorie : api

Le docstring de module (L10) dit « ``mode="h5_exact"`` (default) » et l'exemple L190-193 présente `filt = GSSFilter(params)` sous le titre « Step-by-step, (H5)-exact », alors que la signature (L206 : mode: str = "imm_general") et le docstring de classe (L172) fixent le défaut à imm_general — confirmé par tests/test_gss_filter.py::test_default_mode_is_imm_general. Un utilisateur suivant le docstring de module croit exécuter le filtre exact alors qu'il exécute le filtre approché (avec le biais du finding Γ ci-dessus).

**Preuve :** L10: ``mode="h5_exact"`` (default) — exact IMM recursion under hypothesis (H5)" ; L192: `filt = GSSFilter(params)  # default mode, short form` sous l'en-tête « Step-by-step, (H5)-exact:: » ; L206: `mode: str = "imm_general"`.

**Suggestion :** Corriger L10 en ``mode="imm_general"`` (default) et passer mode="h5_exact" explicitement dans l'exemple L192.

### ✅ [MEDIUM] CLI : aucune option --mode/--joseph — le filtre exact h5 est inatteignable, même avec --constraint

`prg/filter/main.py:247 (et 176-181)` — statut : confirmed (2 vote(s)) — catégorie : api

main() construit toujours `GSSFilter(params)` (L247), donc mode=imm_general. Le flag --constraint (L176-181) applique la contrainte AB pour rendre le modèle (H5)-compatible « before filtering », mais le filtrage qui suit reste le filtre IMM approché — la combinaison naturelle --constraint + h5_exact (la raison d'être du papier) est impossible depuis la CLI. Le GUI (prg/gui/workers.py:78) expose mode et joseph ; seule la CLI est limitée.

**Preuve :** main.py L247: `filt = GSSFilter(params)` — aucun argument mode ; _build_parser ne définit ni --mode ni --joseph alors que --constraint existe (L176).

**Suggestion :** Ajouter --mode {imm_general,h5_exact} (et éventuellement --joseph) au parser, et envisager mode='h5_exact' implicite quand --constraint est passé (ou au minimum le documenter).

### ✅ [MEDIUM] _psd_floor : plancher absolu eps=1e-9 indépendant de l'échelle — transforme une matrice indéfinie en covariance quasi-singulière

`prg/filter/gss_filter.py:1108-1119` — statut : confirmed (2 vote(s)) — catégorie : numerical-stability

Le floor des valeurs propres à eps=1e-9 absolu a deux défauts : (1) quand la matrice est structurellement indéfinie (cas Γ du mode général, vp jusqu'à −6.9), la « réparation » produit une covariance avec vp 1e-9 ⇒ précision 1e9 dans cette direction ⇒ vraisemblances explosives/écrasées au lieu d'un comportement dégradé raisonnable ; (2) pour des modèles d'échelle très petite (Σ ~ 1e-12), le floor gonfle des vp légitimes. Pour de très grandes échelles (‖M‖ ~ 1e12+), l'erreur de reconstruction eigh (~‖M‖·eps_machine) peut excéder 1e-9, donc le résultat n'est même pas garanti PD pour la factorisation Cholesky aval (testé OK à 1e12, mais sans garantie).

**Preuve :** L1119: `return vecs @ np.diag(np.maximum(vals, eps)) @ vecs.T` avec eps=1e-9 fixe ; utilisé sur Γ à L461 et L809. Mesuré : floor déclenché sur des vp de −6.94 (modèle contrast, mode général).

**Suggestion :** Utiliser un plancher relatif (p.ex. eps_rel·max(vals.max(), tiny)) et logger un avertissement quand vals.min() est significativement négatif (signe d'une formule incohérente en amont plutôt que d'un bruit d'arrondi).

### ✅ [LOW] _precompute_gaussian_logpdf : le fallback singulier n'est PAS identique à scipy (pas de test de support → jamais −inf)

`prg/filter/gss_filter.py:1081-1090` — statut : confirmed (1 vote(s)) — catégorie : numerical-stability

Le chemin Cholesky est vérifié exactement équivalent à scipy (diff 0.0 sur cas PD ; 1.8e-15 max sur les log-vraisemblances d'un run complet h5_exact de 300 pas). Mais le fallback pseudo-inverse diverge de scipy.stats.multivariate_normal.logpdf(allow_singular=True) sur trois points : (1) pas de vérification d'appartenance au support — pour Γ singulière, un y hors du sous-espace porteur reçoit une log-densité FINIE (vérifié : scipy 1.17.1 → −inf, fast → −1.044) car la composante hors-support est silencieusement projetée ; (2) le cutoff utilise vals.max() (signé) au lieu de max(abs(vals)) chez scipy ; (3) scipy lève ValueError pour une matrice indéfinie, le fallback tronque silencieusement les vp négatives. En pratique ce chemin est quasi inatteignable (Γ est PSD-floored à 1e-9 ⇒ Cholesky réussit), mais le docstring « mathematically identical to scipy » est inexact, et aucun test unitaire ne verrouille l'équivalence (grep _fast_logpdf dans tests/ : vide).

**Preuve :** Expérience : cov=[[1,0],[0,0]], x=[0.5,1.0] → scipy logpdf = −inf, _fast_logpdf = −1.0439. L1084: `eps = 1e6 * np.finfo(np.float64).eps * max(float(vals.max()), 0.0)` vs scipy `cond * np.max(np.abs(s))`.

**Suggestion :** Soit ajouter le test de support (‖dev − Vᵀ_keep V_keep dev‖ > tol ⇒ −inf) dans _fast_logpdf quand rank < s, soit nuancer le docstring. Ajouter un test unitaire d'équivalence fast-path vs scipy (PD et singulier).

### ✅ [LOW] step() : aucune validation de la dimension de y (broadcast silencieux au 1er pas) ; NaN non gérés (pas d'observations manquantes)

`prg/filter/gss_filter.py:515, 603-606, 624` — statut : confirmed (1 vote(s)) — catégorie : robustness

step() fait reshape(-1,1) sans vérifier que y a s éléments. Vérifié : sur un modèle s=2, passer un y à 1 élément est ACCEPTÉ silencieusement au pas initial (broadcast (1,1)−(2,1)) et produit des estimées fausses sans erreur ; l'erreur ne surgit qu'au 2e pas (y_pred_acc non broadcastable). Par ailleurs un y contenant NaN n'est pas traité comme manquant : il produit E_x=NaN au pas courant, log_lik=−inf à CE pas ET au suivant (y_prev=NaN contamine mean_jk via L624), avec repli silencieux de π — vérifié expérimentalement. Aucun support d'observations manquantes (le point 5 de l'audit : non géré).

**Preuve :** Expérience modèle K2_q1_s2 : `f.step(np.array([1.0]))` → « WRONG-LENGTH y ACCEPTED silently, pi=[0.541, 0.459] » ; NaN : pas n → E_x=[nan], log_lik=−inf ; pas n+1 (y valide) → log_lik=−inf encore.

**Suggestion :** Dans step(), valider `y.size == self._p.s` (ValueError sinon) et soit rejeter les NaN explicitement, soit implémenter le saut d'observation (étape de prédiction pure : π_{n+1} = πᵀP, moments propagés sans correction).

### ✅ [LOW] _safe_solve : le docstring promet un fallback pour matrices mal conditionnées, mais seul LinAlgError (singularité exacte) est intercepté

`prg/filter/gss_filter.py:1029-1042 (et 803-808)` — statut : confirmed (1 vote(s)) — catégorie : numerical-stability

np.linalg.solve ne lève pas pour une matrice mal conditionnée mais inversible : il retourne un résultat imprécis silencieusement ; le fallback lstsq n'est atteint que pour une singularité exacte. Or dans le mode général, S_YY_n provient de la soustraction non-floorée Sig_n = P_z − µµᵀ (L803-805) et peut être quasi singulière ; M_t = _safe_solve(S_YY_n.T, ...) amplifie alors le bruit sans avertissement.

**Preuve :** L1037-1042 : `try: return np.linalg.solve(A, B) except np.linalg.LinAlgError: ... lstsq` ; docstring L1034 : « If A is singular or ill-conditioned, falls back » — la seconde moitié est fausse.

**Suggestion :** Soit corriger le docstring, soit tester cond(A) (ou utiliser scipy.linalg.solve(..., assume_a='sym') + contrôle du résidu) avant de basculer sur lstsq.

### ✅ [LOW] seed=0 traité comme falsy : tag de log « seedrandom » pour une graine légitime

`prg/filter/main.py:215` — statut : confirmed (1 vote(s)) — catégorie : bug

`tag = f"{args.model}_N{args.N or 'csv'}_seed{args.seed or 'random'}"` : avec --seed 0, `0 or 'random'` donne 'random' ; le nom du fichier de log est trompeur. Les chemins de sortie réels (run() L947 et _resolve_output L310) utilisent correctement `if seed is not None`.

**Preuve :** L215: `seed{args.seed or 'random'}` vs gss_filter.py L947: `seed_str = str(seed) if seed is not None else "random"`.

**Suggestion :** `seed{args.seed if args.seed is not None else 'random'}` (idem pour N).

### ✅ [INFO] Les deux modes ignorent π₀/µ_z0/Σ_z0 de l'utilisateur au filtrage : démarrage forcé à la stationnarité, alors que le simulateur tire R₀~π₀, Z₀~N(µ_z0,Σ_z0)

`prg/filter/gss_filter.py:478, 542-563, 700-731` — statut : confirmed (1 vote(s)) — catégorie : math

L'init (deux modes) utilise π_∞ et les moments stationnaires (µ_z0/Σ_z0 ne servent que de point de départ à l'itération de point fixe). GSSSimulator (GSSSimulator.py:133-135) tire en revanche R₀ ~ pi0 et Z₀ ~ N(mu_z0, Sigma_z0). Si pi0 ≠ π_∞ ou si les conditions initiales ne sont pas stationnaires, le filtre est mal spécifié sur les premiers pas et la log-vraisemblance initiale est biaisée. C'est documenté pour h5_exact (commentaire L535-539 : sous (H5) Γ constant EXIGE la stationnarité, donc choix cohérent), mais le mode imm_general — qui n'a pas cette contrainte — applique le même choix silencieusement.

**Preuve :** L478: `self._pi = self._pi_inf.copy()` ; L555: `np.log(self._pi_inf[r] + 1e-300)` — p.pi0 n'est jamais lu par le filtre ; GSSSimulator.py:133: `r_n = int(self._rng.choice(params.K, p=params.pi0))`.

**Suggestion :** Pour imm_general, initialiser depuis (pi0, mu_z0, Sigma_z0) ou au minimum documenter dans le docstring de classe que pi0/mu_z0/Sigma_z0 sont ignorés par le filtre.

### ✅ [INFO] Vérifications positives : récursion h5_exact conforme au papier, chemin rapide exact, log-sum-exp partout, hypothèse Γ time-invariant respectée

`prg/filter/gss_filter.py:444-467, 603-686` — statut : confirmed (1 vote(s)) — catégorie : math

Audit de correction sans anomalie sur le cœur h5_exact : (1) M̃_{j,k} = (F_k Σ(j))[q:,q:] Σ_YY(j)⁻¹ se développe en C_k K_j + D_k, cohérent avec la moyenne C_k(µ_X(j)+K_j(y_n−µ_Y(j))) + D_k y_n + b_Y(k) ; (2) Γ(j,k) = C_k P_post(j) C_kᵀ + Σ_V(k) coïncide avec l'éq. (Gamma_jk) du papier (P_post = Σ_U − ΔΣ_V⁻¹Δᵀ et K_j = Δ_jΣ_{V,j}⁻¹ sous (H5)) ; (3) forme de Joseph algébriquement égale à la forme courte (le gain Σ_XY Σ_YY⁻¹ est optimal pour H=Σ_YX Σ_XX⁻¹, R=S_YY−HΣ_XX Hᵀ) ; (4) le chemin rapide n'est utilisé QUE là où Γ est time-invariant (h5_exact, L632) — imm_general garde scipy par pas (L816), et les inits utilisent scipy une fois ; équivalence end-to-end fast vs scipy : max|Δloglik|=1.8e-15, max|Δπ|=6.7e-16 sur 300 pas ; (5) pondérations : joint = π_n(j)P(j,k) (Σ=1), normalisations en log-domaine avec replis sains (π_∞ / marg_rnp1, qui somment à 1) ; innovation = y − Σ_{j,k} w·mean_jk correcte ; (6) point fixe stationnaire : renversement temporel p_rev = π_∞(j)P(j,k)/π_∞(k) correct, garde de non-convergence loggée ; (7) T=1 et q≠s OK ; K=1 rejeté en amont par FMatrix (« K must be >= 2 »), donc hors périmètre du filtre.

**Preuve :** Tests numériques exécutés dans le venv du projet (scipy 1.17.1, numpy 2.4.4) : diff fast-vs-scipy = 0.0 (cas PD s∈{1,2,4}), 1.8e-15 end-to-end ; π_h5 vs π_general ≈ 1e-5 sous (H5) sur modèle bien conditionné (K2_q1_s1).

**Suggestion :** Rien à corriger ; envisager d'ajouter le test unitaire d'équivalence fast-path (cf. finding _precompute_gaussian_logpdf).

## Contrainte H5/AB (prg/utils/h5_constraint.py)

_Audit de la contrainte H5/AB : lecture intégrale de prg/utils/h5_constraint.py (241 l.) et prg/utils/matrix_checks.py (341 l.), survol de tests/test_h5_constraint.py, du consommateur GSSFilter._check_h5, des classes FMatrix/NoiseCovariance/GSSParams et de l'appendice B du papier, avec vérifications numériques exécutées dans le venv du projet (numpy 2.4.4). La forme close compute_AB et le résidu compute_h5_residual sont mathématiquement corrects — toutes les transpositions vérifiées par re-dérivation indépendante (F = transposée du numérateur de β₁), résidu exactement nul sur la variété AB, et la propriété « toutes paires (r₁,r₂) » de la forme AB est prouvée et confirmée numériquement ; aucun reliquat du bug Δᵀ A historique, et ni inv ni pinv ne sont utilisés. Le problème principal est sémantique : le résidu n'évalue que l'équation de la paire diagonale (j=k), qui est nécessaire mais pas suffisante pour le (H5) complet — contre-exemple construit (résidu diagonal 4e-16, résidu croisé 4.62) contredisant les docstrings « ⇔ / source of truth » et la réduction j=k de l'appendice B du papier (dont l'étape 4 contient en outre deux identités fausses, bien que l'équation finale soit juste). Problèmes secondaires : contrat d'exception violé pour Σ_V non fini (LinAlgError au lieu de ValueError), absence de garde de conditionnement sur M, aucune validation de formes (Δ transposé accepté silencieusement quand q=s), tolérance de symétrie absolue non invariante d'échelle dans matrix_checks qui bloque des covariances légitimes à grande échelle, et compute_SU_from_h5 (cité dans la demande) qui n'existe plus depuis v0.12.0. Santé générale du périmètre : bonne sur le plan algébrique, perfectible sur la robustesse des entrées et la rigueur des prétentions documentaires._

### ✅ [HIGH] Le résidu (H5) par régime n'est qu'une condition NÉCESSAIRE : résidu nul ≠ (H5) complet (équations de paires croisées non vérifiées)

`prg/utils/h5_constraint.py:39-42, 65-102` — statut : confirmed (2 vote(s)) — catégorie : math

compute_h5_residual évalue uniquement l'équation de la paire diagonale (j=k) — les 7 matrices passées appartiennent à un seul régime. Or (H5) (indépendance conditionnelle, eq:H5 du papier) exige β₁=0 pour TOUTES les paires (j,k) avec p_jk>0, où l'équation de paire fait intervenir A_k,B_k,C_k,D_k,Δ_k,Σ_V,k ET les blocs de bruit du régime source j (Σ_U,j, Δ_j, Σ_V,j). Le docstring affirme « ‖F‖ = 0 ⇔ (H5) holds exactly » (l.88) et « source of truth for (H5)-compatibility » (l.40-42) : c'est faux dans le sens ⇒. L'équation diagonale seule donne s équations par ligne de [A B] pour q+s inconnues — toujours sous-déterminée — donc il existe des (A,B) non-AB de résidu diagonal nul qui violent les équations croisées. Contre-exemple numérique construit (q=s=1, K=2) : (a,b)=(-5.2638, 0.7250) donne résidu diagonal 4.4e-16 mais résidu de paire croisée (j≠k) = 4.62. Conséquence : GSSFilter._check_h5 (prg/filter/gss_filter.py:232-276, boucle sur k seulement) peut valider un modèle non-(H5) et le filtre 'h5_exact' serait silencieusement biaisé. Noter aussi que le docstring du module (l.22-30) est lui-même incohérent avec cette prétention : il fonde la nécessité de AB sur l'élimination des K équations de régimes sources (K·s ≥ q+s), ce qui reconnaît implicitement que l'équation diagonale seule ne suffit pas. Cas limite supplémentaire : si p_kk=0, l'équation diagonale n'est même pas requise par (H5) (la paire (k,k) n'apparaît jamais) — le check est alors trop strict. En pratique, l'exposition est limitée car tous les modèles (H5) du dépôt sont construits via compute_AB (qui, lui, satisfait bien toutes les paires — vérifié), mais le contrat documenté de « vérité terrain » est erroné.

**Preuve :** h5_constraint.py:88 « F : ndarray of shape (s, q). ‖F‖ = 0 ⇔ (H5) holds exactly. » ; l.40-42 « This is the source of truth for (H5)-compatibility » ; gss_filter.py:249 « for k in range(p.K): ... compute_h5_residual(A,B,C,D,SU,Dt,SV) » (jamais de paires j≠k). Contre-exemple exécuté : résidu diagonal 4.44e-16, résidu paire (j,k) 4.6187 pour (a,b)=(-5.263793, 0.725000) avec C=0.4, D=0.6, Σ_U,k=1, Δ_k=0.3, Σ_V,k=0.8 et régime source Σ_U,j=2, Δ_j=-0.5, Σ_V,j=1.5.

**Suggestion :** Généraliser compute_h5_residual en compute_h5_pair_residual(A_k,B_k,C_k,D_k,Δ_k,Σ_V,k, Σ_U,j,Δ_j,Σ_V,j) avec T_jk = A_k Σ_U,j C_kᵀ + A_k Δ_j D_kᵀ + B_k Δ_jᵀ C_kᵀ + B_k Σ_V,j D_kᵀ + Δ_k et M_jk = C_k Σ_U,j C_kᵀ + ... + Σ_V,k, puis faire boucler _check_h5 sur les K² paires (ou seulement celles avec p_jk>0). Alternative en régime déterminé (K·s ≥ q+s) : comparer directement (A_k,B_k) à compute_AB(C_k,D_k,Δ_k,Σ_V,k). Reformuler les docstrings : résidu diagonal nul = condition nécessaire (si p_kk>0), pas suffisante.

### ✅ [MEDIUM] Appendice B du papier : l'étape 4 contient des identités fausses et la réduction « j=k nécessaire et suffisant » est incorrecte

`paper/appendix/B_h5_derivation.tex:100-103, 139-152` — statut : confirmed (2 vote(s)) — catégorie : math

Deux problèmes dans la source mathématique citée par le docstring du module. (1) l.100-103 : « For this to hold for all source regimes j, it is sufficient and necessary to enforce (h5_beta1) in the special case j=k » — la direction « suffisant » est fausse : le contre-exemple numérique (voir finding sur le résidu) exhibe (A,B) satisfaisant l'équation diagonale mais violant la paire (j,k), j≠k. Ce qui est vrai (et prouvé ici) : la forme AB satisfait toutes les paires ; mais l'équation diagonale seule ne les implique pas. (2) l.139-152 : l'expansion de T M⁻¹ R s'appuie sur « Q_Aᵀ = C Σ_U + D Δᵀ = Q » — faux : Q_A = A Σ_U + B Δᵀ donc Q_Aᵀ = Σ_U Aᵀ + Δ Bᵀ ≠ Q ; et « Q_A Cᵀ + Q_B Dᵀ + Σ_V = M » — faux aussi (c'est le bloc XY de F P Fᵀ, de forme q×s, dimensionnellement incompatible avec M (s×s) dès que q≠s ; le bloc YY est Q Cᵀ + R Dᵀ). L'équation finale (eq:h5_compact_app, l.155-158) reste néanmoins CORRECTE : elle s'obtient directement en transposant A Δ + B Σ_V = T M⁻¹ R puisque Tᵀ = Q Aᵀ + R Bᵀ + Δᵀ et Rᵀ = P — c'est la dérivation propre, vérifiée ici symboliquement et numériquement. Le code implémente la bonne équation finale ; seul le chemin de preuve du papier est cassé.

**Preuve :** B_h5_derivation.tex:151 « where we used Q_Aᵀ = C Σ_U + D Δᵀ = Q (Q_A = Qᵀ) » alors que Q_A := A Σ_U + B Δᵀ (l.140) ; l.145-146 « Q_A Cᵀ + Q_B Dᵀ + Σ_V is exactly [F·P(r)·Fᵀ + Σ_W]_YY = M » (bloc XY, pas YY ; forme q×s vs s×s). Contre-exemple du caractère non-suffisant de j=k : résidu diagonal 4e-16, résidu croisé 4.62.

**Suggestion :** Remplacer l'étape 4 par la transposition directe (Tᵀ = Q Aᵀ + R Bᵀ + Δᵀ, Rᵀ = P, M = Mᵀ). Reformuler la réduction j=k : l'équation diagonale est nécessaire (si p_kk>0) ; la suffisance pour toutes les paires vaut pour la solution AB spécifiquement (preuve par substitution dans la paire (j,k) générale), pas pour toute solution de l'équation diagonale.

### ✅ [MEDIUM] Tolérance de symétrie ABSOLUE (1e-10) non invariante d'échelle : rejette des covariances légitimes à grande échelle, accepte des asymétries grossières à petite échelle

`prg/utils/matrix_checks.py:36, 191-203` — statut : confirmed (2 vote(s)) — catégorie : robustness

CovarianceMatrix utilise max|M−Mᵀ| < _EPS_SYM = 1e-10 en absolu. Une covariance d'entrées ~1e8 (données non normalisées, prix, volumes) avec une asymétrie purement due aux arrondis float64 (relative ~5e-9, soit l'epsilon machine accumulé) échoue : vérifié, FAIL avec value=1.0. Comme ce checker GATE la construction de GSSNoiseCovariance (NoiseCovariance.py:112-117) et de GSSParams (GSSParams.py:123-128), une boucle EM sur données à grande échelle peut planter avec CovarianceError alors que la matrice est symétrique au sens flottant. Inversement, une matrice 2×2 d'entrées ~1e-11 avec asymétrie relative de 200% passe le check de symétrie (2e-11 < 1e-10). Le même problème d'absolu existe en principe pour _EPS_STOCH mais les matrices stochastiques sont O(1), donc sans conséquence.

**Preuve :** Exécution : M=[[1e8, 2e8+1],[2e8, 5e8]] → « Symmetric FAIL value=1.0 » (asymétrie relative 5e-9) ; M=[[1e-12,3e-11],[1e-11,1e-12]] → check Symmetric OK malgré une asymétrie relative de 2x.

**Suggestion :** Rendre la tolérance relative : sym_err < _EPS_SYM * max(1.0, np.max(np.abs(M))) (ou comparer à ‖M‖∞), et symétriser en amont ((M+Mᵀ)/2) dans les producteurs EM avant validation.

### ❌ [MEDIUM] compute_AB laisse fuir numpy.linalg.LinAlgError pour Σ_V non fini, violant le contrat documenté (ValueError)

`prg/utils/h5_constraint.py:138-147` — statut : refuted (2 vote(s)) — catégorie : api

Le garde-fou cond(Σ_V) (l.138) appelle np.linalg.cond → SVD, qui lève LinAlgError('SVD did not converge') si Σ_V contient NaN/Inf, AVANT le try/except qui n'enveloppe que les solve (l.143-147). Le docstring promet « Raises ValueError If Σ_V is singular or ill-conditioned » : pour une entrée non finie (cas typique d'une itération EM divergente), l'appelant reçoit une LinAlgError brute. Dans apply_AB_constraint, le `except ValueError` (l.204) ne l'attrape pas, donc le message contextuel « regime k=... » est perdu. Vérifié : compute_AB(C, D, Dt, SV_nan) → LinAlgError. Par ailleurs cond(matrice nulle) = inf sous numpy 2.4, donc le chemin Σ_V=0 testé (test_singular_SV_raises) passe bien par le garde-fou, mais c'est un comportement de numpy susceptible de varier (anciennes versions renvoyaient parfois NaN, et NaN > 1e12 est False — le garde-fou serait alors contourné).

**Preuve :** Exécution : « 1. NaN SV -> LinAlgError : SVD did not converge ». Code : le try (l.143) ne couvre pas np.linalg.cond(SV) (l.138).

**Suggestion :** Ajouter en tête de compute_AB : `if not np.all(np.isfinite(SV)): raise ValueError("Σ_V contains NaN/Inf")`, ou envelopper l'appel à cond dans le même try/except LinAlgError → ValueError.

### ✅ [LOW] Affirmation fausse : « M ... ≻ 0 if Σ_U, Σ_V ≻ 0 » — la PD de M exige la PSD de la covariance jointe Σ(r)

`prg/utils/h5_constraint.py:84` — statut : confirmed (1 vote(s)) — catégorie : math

M = [C D] Σ [C D]ᵀ + Σ_V avec Σ = [[Σ_U, Δ],[Δᵀ, Σ_V]]. Si Δ est incompatible avec la PSD jointe, M peut être indéfini même avec Σ_U, Σ_V ≻ 0. Contre-exemple exécuté : q=s=1, C=D=Σ_U=Σ_V=1, Δ=-10 ⇒ M = -17. La condition correcte est Σ(r) ⪰ 0 et Σ_V ≻ 0 (alors M ⪰ Σ_V ≻ 0). La même affirmation imprécise figure dans le papier (paper/sections/04_constraint.tex:38-40 : « M = Mᵀ ≻ 0 whenever Σ_U, Σ_V ≻ 0 »). En pratique le code en aval (GSSNoiseCovariance) impose Σ_W(k) SPD donc la jointe est PD, mais compute_h5_residual est une fonction publique appelée avec des blocs arbitraires (GUI param_panel.py:402, EM) où la jointe peut ne pas être PSD : M singulier/indéfini est alors possible et seul le cas exactement singulier lève LinAlgError.

**Preuve :** Exécution : « M for indefinite joint Sigma = -17.0 (docstring claims M>0 if SU,SV>0) » avec Σ_U=Σ_V=1, Δ=-10.

**Suggestion :** Corriger le docstring (l.84) et le papier : « M ≻ 0 dès que la covariance jointe Σ(r) ⪰ 0 et Σ_V ≻ 0 ».

### ✅ [LOW] compute_h5_residual sans garde de conditionnement sur M (asymétrie avec compute_AB)

`prg/utils/h5_constraint.py:95-101` — statut : confirmed (1 vote(s)) — catégorie : numerical-stability

compute_AB refuse Σ_V si cond > 1e12, mais compute_h5_residual résout M sans aucun contrôle : pour M presque singulier (covariance jointe quasi dégénérée, fréquent en EM), solve renvoie un résultat dominé par l'erreur d'arrondi et le résidu devient non significatif sans avertissement. Le consommateur gss_filter._check_h5 (l.257-269) ne capte que LinAlgError exact (singularité détectée par LAPACK), pas le quasi-singulier : le diagnostic « (H5) ok/violé » peut alors être arbitraire alors qu'il conditionne le choix d'un filtre prétendu exact. Accessoirement, M est symétrique (PD sous les hypothèses) — scipy.linalg.solve(..., assume_a='pos') ou cho_solve serait à la fois plus stable et plus rapide, scipy étant déjà une dépendance.

**Preuve :** l.101 : `residual = Z - P @ np.linalg.solve(M, W)` sans vérification de cond(M), à comparer aux l.138-141 de compute_AB qui imposent cond(SV) ≤ 1e12.

**Suggestion :** Vérifier cond(M) (ou rcond via np.linalg.lstsq / scipy cho_factor) et lever/avertir au-delà d'un seuil cohérent avec H5_TOL=1e-6 du filtre ; utiliser une factorisation de Cholesky avec repli explicite.

### ✅ [LOW] Aucune validation de formes ni de symétrie : Δ transposé accepté silencieusement quand q=s

`prg/utils/h5_constraint.py:108-150, 65-102` — statut : confirmed (1 vote(s)) — catégorie : robustness

Ni compute_AB ni compute_h5_residual ne vérifient la cohérence des dimensions (Dt de forme (q,s) avec q=C.shape[1], s=SV.shape[0]) ni la symétrie de Σ_V (dont dépend la preuve de suffisance de la forme AB : Σ_V⁻ᵀ=Σ_V⁻¹). Pour q≠s, numpy lève des ValueError de matmul cryptiques (sans mention du paramètre fautif) ; pour q=s — configuration majoritaire du dépôt (q1_s1) — passer Δᵀ au lieu de Δ produit silencieusement un (A,B) faux. Vérifié : avec q=s=2, compute_AB(C,D,Dt.T,SV) renvoie ‖A_ok−A_bad‖=1.04 et un résidu (H5) de 0.72 au lieu de 0. Vu l'historique du projet (commit 141fd11 : bug de transposition ayant survécu parce que projecteur et vérificateur partageaient la même erreur), des asserts de forme peu coûteux sont justifiés dans ce module précisément.

**Preuve :** Exécution : « 4. transposed Dt accepted silently; ||A_ok - A_bad|| = 1.0449 ; residual with wrong Dt orientation: 0.7200 ». Aucun check de shape dans les deux fonctions (l.108-150, 65-102).

**Suggestion :** En tête de chaque fonction : valider C.shape=(s,q), D.shape=(s,s), Dt.shape=(q,s), SV.shape=(s,s) à partir de s=SV.shape[0], q=C.shape[1], et `np.allclose(SV, SV.T)` (tolérance relative). La transposition q=s reste indétectable par les formes, mais la symétrie de SV et les autres formes seraient verrouillées.

### ✅ [LOW] apply_AB_constraint peut produire un modèle non stationnaire en moyenne quadratique sans aucun avertissement

`prg/utils/h5_constraint.py:156-241` — statut : confirmed (1 vote(s)) — catégorie : robustness

La forme close remplace A et B par Δ Σ_V⁻¹ C et Δ Σ_V⁻¹ D, dont la norme n'est pas contrôlée : avec Σ_V petit et Δ grand (cas EM réaliste), le F(k) reconstruit peut avoir un rayon spectral (ou un rayon spectral du système commuté) ≥ 1. Ni apply_AB_constraint, ni le constructeur GSSParams ne le détectent ; en aval, _precompute_stationary (gss_filter.py:339-371) itère alors un point fixe divergent pendant 1000 itérations et se contente d'un logger.warning « did not fully converge », laissant potentiellement des inf/NaN dans les moments stationnaires utilisés par le filtre. Notons aussi que la validation de formes de FMatrix appelée lors de la reconstruction est enveloppée dans `if __debug__:` (FMatrix.py:75-77) : sous `python -O`, le GSSParams reconstruit n'est plus validé du tout. Enfin, le nouveau GSSParams partage noise_cov par référence avec l'ancien (l.236) et les accesseurs renvoient les tableaux internes sans copie (NoiseCovariance.py:191-199) — aliasing mutable théorique entre l'objet « nouveau » promis par le docstring et l'original.

**Preuve :** l.230-241 : reconstruction sans contrôle spectral, `noise_cov=params.noise_cov` partagé ; gss_filter.py:364-371 : divergence du point fixe réduite à un warning de log.

**Suggestion :** Après calcul des A_new/B_new, évaluer le rayon spectral de chaque F(k) (et idéalement du produit pondéré par p_rev) et émettre un warning explicite (« le modèle AB-contraint n'est plus stable ») ; documenter le partage de noise_cov ou en faire une copie.

### ✅ [LOW] Lacunes du contrat de test : pas de paires croisées, pas de q≠s pour apply_AB_constraint, pas de comparaison à une vérité indépendante

`tests/test_h5_constraint.py:76-203` — statut : confirmed (1 vote(s)) — catégorie : robustness

La suite teste : formes de sortie, résidu nul sur la variété AB (y compris indépendance en Σ_U — bon test), Σ_V=0 → ValueError, Δ=0 → A=B=0, préservation/idempotence d'apply_AB_constraint, résidu non nul pour (A,B) aléatoires. Manques notables au vu des findings : (1) aucun test des équations de paires croisées (j≠k) — la propriété « K² regime-pair equations » revendiquée par le module n'est jamais vérifiée, et la non-suffisance du résidu diagonal n'est pas documentée par un test ; (2) apply_AB_constraint n'est testé que sur model_gss_K2_q1_s1 (q=s=1), où toute erreur de transposition est invisible — l'expérience du commit 141fd11 montre que résiduel et projection peuvent être faux de manière auto-cohérente, seul un cas rectangulaire q≠s end-to-end le détecterait ; (3) pas de test du contrat d'exception pour Σ_V non fini (qui échoue aujourd'hui : LinAlgError au lieu de ValueError) ; (4) compute_h5_residual n'est jamais confronté à une référence indépendante (ex. β₁ estimé par régression Monte-Carlo sur des tirages gaussiens), seule l'auto-cohérence avec compute_AB est testée — précisément le mode de défaillance qui a masqué le bug historique.

**Preuve :** tests/test_h5_constraint.py : fixture unique params_K2_q1_s1 (l.62-70) pour tout TestApplyABConstraint ; aucune occurrence de paire (j,k) croisée ; compute_AB testé en q=2,s=3 mais seulement contre compute_h5_residual.

**Suggestion :** Ajouter : un test de paires croisées sur la forme AB (résidu (j,k) nul pour tout j) et un test négatif (solution diagonale non-AB → résidu croisé non nul) ; un fixture q≠s pour apply_AB_constraint ; un test Monte-Carlo de β₁≈0 sous AB ; un test NaN/Inf → ValueError une fois le contrat d'exception corrigé.

### ✅ [INFO] Forme close et résidu vérifiés corrects : aucune erreur de transposition résiduelle

`prg/utils/h5_constraint.py:95-102, 108-150` — statut : confirmed (1 vote(s)) — catégorie : math

Audit positif des points 1 du périmètre. (a) Dérivation indépendante : F = Z − P M⁻¹ W est exactement la transposée du numérateur du coefficient de régression partielle β₁ (N = Cov(X+,Yn) − Cov(X+,Y+) M⁻¹ Cov(Y+,Yn) = Fᵀ, avec Wᵀ = T et Rᵀ = P), donc ‖F‖=0 ⇔ β₁=0 pour la paire diagonale (S = Σ_V − P M⁻¹ R inversible). Toutes les transpositions de l.95-101 (Dt.T @ C.T, Q @ A.T, Dt.T @ A.T...) sont conformes — le bug historique Δᵀ A vs Δᵀ Aᵀ (commit 141fd11) est bien purgé, y compris dans le consommateur gss_filter.py:270. (b) Dimensions cohérentes : P,R,M (s,s), Q,W,Z,F (s,q), A (q,q), B (q,s). (c) compute_AB : A = Δ Σ_V⁻¹ C, B = Δ Σ_V⁻¹ D vérifié algébriquement (Z = P Σ_V⁻¹ Δᵀ, W = M Σ_V⁻¹ Δᵀ ⇒ F = 0, en utilisant la symétrie de Σ_V) et numériquement (résidu 0.0 exact). (d) La prétention du docstring de compute_AB (l.119-121) que la forme AB satisfait TOUTES les paires (r₁,r₂) indépendamment de Σ_U est vraie : prouvé analytiquement (Cov(X+,Y+|j,k) = Δ_k Σ_V,k⁻¹ M_jk pour tout j) et confirmé numériquement (résidu croisé 5.6e-17). (e) solve est utilisé partout, jamais inv ni pinv (aucun pinv dans prg/ ni scripts/). (f) apply_AB_constraint préserve les covariances inchangées, donc aucune question de PSD de covariances reconstruites ne se pose dans le module actuel.

**Preuve :** Vérification numérique : « diag residual at AB : 0.0 ; cross residual at AB: 5.55e-17 » et dérivation Wᵀ = A Qᵀ + B Rᵀ + Δ = T, Rᵀ = P ⇒ F = (numérateur de β₁)ᵀ.

**Suggestion :** Rien à corriger sur la forme close elle-même. Ajouter éventuellement un test de paires croisées (j≠k) dans tests/test_h5_constraint.py pour verrouiller la propriété « toutes paires » de compute_AB (actuellement non testée : tous les tests sont mono-régime ou diagonaux).

### ✅ [INFO] Le niveau Status.WARNING est du code mort ; PD strict exigé (PSD-singulier rejeté) ; pas d'alerte de quasi-singularité

`prg/utils/matrix_checks.py:45-49, 205-229` — statut : confirmed (1 vote(s)) — catégorie : style

Trois observations mineures liées. (1) Aucun check n'émet jamais Status.WARNING : l'enum, overall_status et is_ok/is_valid gèrent un niveau qui ne peut pas se produire. (2) CovarianceMatrix exige la PD stricte via Cholesky : une covariance PSD-singulière légitime (ex. Σ_z0 déterministe, diag(1,0)) est classée FAIL — choix cohérent avec l'hypothèse Σ(r) ≻ 0 du cadre (H5), mais non documenté comme tel (le terme « covariance » suggère PSD). (3) Une matrice PD avec valeur propre 1e-300 ou cond=1e15 passe OK sans avertissement, alors que compute_AB refusera ensuite cond>1e12 : le niveau WARNING serait l'endroit naturel pour signaler la quasi-singularité et rendre les diagnostics cohérents avec les seuils du module h5.

**Preuve :** Exécution : CovarianceMatrix(np.diag([1.0, 0.0])).check() → FAIL. Grep : aucun `Status.WARNING` n'est construit dans matrix_checks.py en dehors de l'enum et de l'agrégation.

**Suggestion :** Émettre WARNING quand cond(M) > 1e12 (aligné sur compute_AB) ou min_eig < eps·trace ; documenter explicitement que la PD stricte est requise par conception (H5).

### ✅ [INFO] compute_SU_from_h5 (cité dans la demande d'audit) n'existe plus : supprimé en v0.12.0 et interdit par test

`prg/utils/h5_constraint.py:55-59` — statut : confirmed (1 vote(s)) — catégorie : api

La fonction compute_SU_from_h5 mentionnée dans le périmètre d'audit a été supprimée au commit 3089e5d (« replace per-matrix H5 projections with Lehmann's closed form », v0.12.0) avec les autres projections par matrice (compute_A_from_h5, compute_B_from_h5, compute_C_from_h5), et son nom est désormais activement interdit par tests/test_no_stale_refs.py:35-44. Le module actuel n'expose que compute_AB, apply_AB_constraint et compute_h5_residual ; aucune reconstruction de covariance n'y subsiste, donc les préoccupations de symétrie/PSD d'un Σ_U reconstruit sont sans objet dans le code courant. L'historique confirme par ailleurs que le bug de transposition Δᵀ A → Δᵀ Aᵀ (141fd11) touchait l'ancienne famille de fonctions ; la version actuelle est saine sur ce point.

**Preuve :** git log : 3089e5d « feat(h5)!: replace per-matrix H5 projections with Lehmann's closed form (v0.12.0) » ; tests/test_no_stale_refs.py:35-36 bannit le nom compute_SU_from_h5 ; __all__ = [apply_AB_constraint, compute_AB, compute_h5_residual] (l.55-59).

**Suggestion :** Aucune action code. Mettre à jour les consignes/documents d'audit internes qui référencent encore compute_SU_from_h5.

## Classes du modèle (prg/classes/)

_Audit intégral des quatre classes du modèle (GSSParams.py 425 l., GSSSimulator.py 289 l., NoiseCovariance.py 235 l., FMatrix.py 197 l.) plus prg/simulate.py, avec lecture des dépendances (matrix_checks.py, exceptions.py, h5_constraint.py, base_gss_model.py) et vérifications numériques exécutées dans le venv du dépôt (numpy 2.4.4). Santé générale bonne: le simulateur échantillonne exactement le modèle génératif documenté (ordre régime→état, conditionnement sur le nouveau régime, graine reproductible — validé statistiquement sur 4·10^5 pas), les entrées sont copiées à la construction, et les gardes récents de _compute_stationary sont mathématiquement corrects sur les cas périodiques, réductibles et transitoires (le cas « signes mixtes » produit en fait encore une loi exactement stationnaire grâce aux supports disjoints des classes récurrentes; seuls les libellés des warnings surinterprètent). Les points sérieux sont d'ordre robustesse/API: toute la validation structurelle disparaît sous `python -O` (P non stochastique et formes invalides acceptées silencieusement), x_n/y_n retournés sont des vues de l'état interne du simulateur (mutation par l'appelant = trajectoire corrompue), FMatrix n'a aucun contrôle de finitude (NaN acceptés), b_list n'est pas validée en longueur, tous les accesseurs exposent les tableaux internes par référence mutable, et l'option CLI --output ignore silencieusement le nom de fichier fourni._

### ✅ [HIGH] Toute la validation structurelle est désactivée sous `python -O` (gardée par `__debug__`)

`prg/classes/GSSParams.py:88-94` — statut : confirmed (2 vote(s)) — catégorie : robustness

GSSParams._validate_* (stochasticité de P, forme de pi0, dimensions de mu_z0/Sigma_z0, cohérence K/q/s) n'est exécutée que `if __debug__:`. Idem FMatrix.py:73-75, NoiseCovariance.py:85-87, et le contrôle de finitude du simulateur (GSSSimulator.py:146-148). Sous `python -O`, vérifié empiriquement: une P avec ligne sommant à 1.4 ET un pi0 sommant à 1.4 sont acceptés à la construction (l'échec ne survient que plus tard dans rng.choice, ou jamais si numpy tolère); un mu_z0 de forme (2,) au lieu de (2,1) passe et produit par broadcasting un x_n de forme (1,2) — données silencieusement fausses. Cela contredit la docstring ('It validates every parameter at construction time so that downstream code can assume correctness'). Seuls les contrôles SPD (GSSParams.py:121-129, NoiseCovariance.py:110-119) survivent à -O car hors du bloc __debug__.

**Preuve :** GSSParams.py:88 `if __debug__:` englobe _validate_P/_validate_pi0/_validate_initial_conditions. Test sous -O: « non-stochastic P AND pi0 sum=1.4 ACCEPTED at construction »; « mu shape (2,) accepted; x_n shape = (1, 2) -> silent garbage ».

**Suggestion :** Sortir la validation du bloc `if __debug__:` (le coût est négligeable, une fois à la construction), ou réserver __debug__ aux seuls contrôles per-step du simulateur. A minima, corriger la docstring.

### ✅ [MEDIUM] x_n et y_n retournés sont des VUES de l'état interne _z_prev — une mutation par l'appelant corrompt la trajectoire

`prg/classes/GSSSimulator.py:150-158` — statut : confirmed (2 vote(s)) — catégorie : bug

`x_n = z_n[:q, :]` et `y_n = z_n[q:, :]` sont des slices (vues) du même tableau qui est conservé comme `self._z_prev` (ligne 151). Vérifié: `x.base is sim._z_prev` est True; après `x[0,0] = 999.0` par l'appelant, le pas suivant part de z_prev corrompu et la trajectoire diverge silencieusement. Risque réel en code de recherche où l'utilisateur post-traite les tableaux retournés (normalisation in place, etc.).

**Preuve :** Test: « x is view of z_prev: True | y view: True »; « trajectory corrupted by caller mutation: True ». Code: `self._z_prev = z_n` (l.151) puis `x_n = z_n[:q, :]` (l.156).

**Suggestion :** Retourner des copies: `x_n = z_n[:q, :].copy()` et `y_n = z_n[q:, :].copy()` (coût négligeable pour des vecteurs (q,1)/(s,1)).

### ✅ [MEDIUM] FMatrix n'effectue aucun contrôle de finitude: NaN/Inf dans A/B/C/D acceptés silencieusement

`prg/classes/FMatrix.py:106-124` — statut : confirmed (2 vote(s)) — catégorie : robustness

`_validate_blocks` ne vérifie que les formes. Vérifié: A(0)=[[nan]] est accepté par FMatrix ET par GSSParams; la simulation n'échoue qu'au premier pas qui visite le régime 0 (SimulationError au runtime, et uniquement avec __debug__ actif — sous -O, les NaN se propagent jusque dans le CSV). Pour les filtres (gss_filter.py), un F non fini corromprait les résultats sans aucune alerte. Contraste avec CovarianceMatrix/StochasticMatrix qui ont un check 'Finite'.

**Preuve :** Test: « NaN in A accepted by FMatrix: True »; « simulation with NaN F raised: SimulationError Non-finite values generated at step n=3, r=0 » (détection tardive, et seulement si __debug__).

**Suggestion :** Ajouter `if not np.all(np.isfinite(arr)): raise ParamError(...)` dans FMatrix._validate_blocks (et idem pour b_list et mu_z0_list dans GSSParams).

### ✅ [MEDIUM] Longueur de b_list jamais validée (même avec __debug__)

`prg/classes/GSSParams.py:115-118` — statut : confirmed (2 vote(s)) — catégorie : bug

Contrairement à mu_z0_list/Sigma_z0_list (validées via _validate_initial_conditions), b_list n'a aucun contrôle de longueur: `self._b = [np.array(b).reshape(q+s,1) for b in b_list]` accepte une liste de longueur != K. Vérifié: b_list de longueur 1 avec K=2 est accepté; `params.b(1)` lève IndexError seulement à l'usage (potentiellement au milieu d'une simulation ou d'un filtrage). Une liste trop longue serait tronquée de fait sans erreur du tout côté b(k) pour k<K. Le reshape attrape les tailles incohérentes mais accepte silencieusement (q+s,) ou (1,q+s).

**Preuve :** Test: « b_list of wrong length accepted: 1 » puis « IndexError only at access time: b(1) ». Code l.118: `self._b = [np.array(b, dtype=float).reshape(q + s, 1) for b in b_list]` sans `len(b_list) != K` check.

**Suggestion :** Ajouter dans __init__ (hors __debug__): `if b_list is not None and len(b_list) != K: raise ParamError(...)`, plus contrôle de finitude.

### ✅ [MEDIUM] Tous les accesseurs exposent les tableaux internes par référence — mutation externe silencieuse possible après validation

`prg/classes/GSSParams.py:360-394` — statut : confirmed (2 vote(s)) — catégorie : api

GSSParams.P/pi0/b(k)/mu_z0(k)/Sigma_z0(k)/chol_z0(k), FMatrix.F(k)/A(k)/B(k)/C(k)/D(k) (FMatrix.py:150-174) et GSSNoiseCovariance.Sigma_W(k)/Sigma_U(k)/Delta(k)/Sigma_V(k)/chol_W(k) (NoiseCovariance.py:181-212) retournent les ndarray internes sans copie ni flag read-only. Vérifié: `params.P[0,0] = -5.0` mute l'objet validé; `f_matrix.F(0)[0,0] = 123.0` mute le F caché alors que A(0) reste inchangé — car F(k) (np.block, copie) et les blocs _A/_B/_C/_D sont des stockages indépendants, on peut donc créer une INCOHÉRENCE interne F(k) != [[A,B],[C,D]] sans aucune erreur. Idem Sigma_W(k) vs ses blocs et vs le chol caché. Point positif: les entrées sont bien copiées à la construction (np.array(...) partout, GSSParams.py:100,110-111; FMatrix.py:83-86; NoiseCovariance.py:94-96), donc pas d'aliasing entrant.

**Preuve :** Test: « params.P mutated in place: True »; « F(0) mutated in place: True »; « A(0) unchanged while F(0) was mutated independently: 0.8 ».

**Suggestion :** Après construction, poser `arr.flags.writeable = False` sur tous les tableaux internes (P, pi0, _b, _mu_z0, _Sigma_z0, _chol, _F, _A.., _Sigma_W..). Coût nul, et toute mutation accidentelle lève ValueError immédiatement.

### ✅ [MEDIUM] Option --output: le nom de fichier fourni est silencieusement ignoré

`prg/simulate.py:316-323` — statut : confirmed (2 vote(s)) — catégorie : bug

Le code n'extrait de `args.output` que le répertoire parent (`out_path.parent`, et seulement s'il diffère de '.'); le nom de fichier lui-même n'est jamais transmis à `sim.run()`, qui auto-génère toujours `simulated_<model>_N<N>_seed<seed>.csv` (GSSSimulator.py:208). Donc `--output results.csv` n'a strictement AUCUN effet (le fichier va dans data/simulated avec le nom auto), et `--output dir/results.csv` n'honore que `dir/`. L'aide annonce pourtant « Output CSV filename (auto-generated if omitted) ».

**Preuve :** simulate.py:319-323: `out_path = pathlib.Path(args.output); if out_path.parent != pathlib.Path('.'): out_dir = out_path.parent` puis `sim.run(output_dir=out_dir, model_name=args.model)` — out_path.name jamais utilisé.

**Suggestion :** Ajouter un paramètre `filename` (ou `output_path`) à GSSSimulator.run() et passer out_path.name; ou retirer l'option/corriger l'aide.

### ✅ [LOW] _compute_stationary: gardes mathématiquement sains pour P stochastique validée; message de la garde 1 surinterprète, garde 2 sur-pessimiste

`prg/classes/GSSParams.py:257-327` — statut : confirmed (1 vote(s)) — catégorie : math

Audit math des trois gardes, vérifié numériquement. (a) Chaînes périodiques: OK sans faux positif — 2-cycle donne [0.5,0.5], 3-cycle [1/3,1/3,1/3] (valeurs propres e^{±2πi/3} loin de 1). (b) États transitoires: OK, pi=0 dessus, pas d'alerte (vp 1 simple). (c) Réductible: garde 1 détecte correctement (multiplicité de la vp 1 = nb de classes récurrentes pour une matrice stochastique, vp semi-simple donc bien séparée numériquement, erreurs eig ~1e-15 << tol 1e-8). (d) Garde 2 (signes mixtes): pour une P stochastique, l'espace propre gauche de la vp 1 est engendré par les lois stationnaires des classes récurrentes, à SUPPORTS DISJOINTS; donc |αpi1+βpi2| = |α|pi1+|β|pi2 et le résultat après renormalisation EST exactement stationnaire — vérifié: vecteur mixte trouvé (trial 44), ||piP-pi||=1.2e-16. Le texte « may not be a valid stationary distribution » est donc trop pessimiste dans les cas atteignables; le vrai problème (arbitraire du mélange) est déjà couvert par la garde 1. (e) Tolérance 1e-8 de la garde 1: une chaîne IRRÉDUCTIBLE à mélange très lent (trou spectral < 1e-8, testé avec couplage eps=1e-10) déclenche le message « the Markov chain is reducible » alors qu'elle ne l'est pas (le pi retourné [0.5,0.5] était pourtant correct). (f) Garde 3 (fallback uniforme): saine — atteignable en pratique seulement via NaN (LAPACK normalise les vecteurs propres à norme 1, donc somme après abs >= 1/sqrt(K) > 0); `not np.isfinite(total)` couvre bien NaN/Inf; l'uniforme est une loi valide quoique non stationnaire en général, acceptable comme dernier recours documenté. (g) Pas de garde sur dist_to_one[idx] lui-même: sans conséquence pour une P validée (vp 1 existe toujours), mais la statique appelée sur une matrice sous-stochastique retournerait silencieusement le vecteur de Perron normalisé.

**Preuve :** Tests: réductible bloc-diagonale → pi=[2/3,1/3,0,0], ||piP-pi||=0, 1 warning; périodique → [0.5,0.5] sans warning; transitoire → [0,0.43,0.57] sans warning; vecteur à signes mixtes → pi stationnaire à 1.2e-16 près avec 2 warnings; quasi-réductible eps=1e-10 → warning « is reducible » faux mais pi correct.

**Suggestion :** Reformuler la garde 1: « reducible or nearly reducible (spectral gap < 1e-8) »; reformuler la garde 2 (le résultat reste une loi stationnaire admissible si P est stochastique, seul le mélange est arbitraire). Alternative plus robuste: résoudre le système bordé (I-Pᵀ)pi=0, somme(pi)=1 par moindres carrés.

### ✅ [LOW] __repr__ affiche toujours pi0='given', jamais 'stationary'

`prg/classes/GSSParams.py:421-425` — statut : confirmed (1 vote(s)) — catégorie : bug

Le repr teste `self._pi0 is None`, mais _pi0 n'est jamais None après __init__ (résolu lignes 105-108 vers la distribution stationnaire). Vérifié: un GSSParams construit avec pi0=None affiche `pi0=given`. Information trompeuse en debug/logs.

**Preuve :** Test: `repr: <GSSParams(K=2, q=1, s=1, pi0=given)>` pour pi0=None. Code l.424: `pi0={'stationary' if self._pi0 is None else 'given'}`.

**Suggestion :** Mémoriser un drapeau `self._pi0_was_stationary = pi0 is None` dans __init__ et l'utiliser dans __repr__.

### ✅ [LOW] isinstance(.., int) rejette les entiers numpy et accepte bool (K/q/s, et N du simulateur)

`prg/classes/GSSParams.py:184-191` — statut : confirmed (1 vote(s)) — catégorie : api

Vérifié: K=np.int64(2) lève ParamError avec le message déroutant « K must be an integer >= 2, got np.int64(2) » (l'utilisateur voit un 2 valide rejeté); q=True est accepté et devient silencieusement q=1; N=True est accepté par GSSSimulator (GSSSimulator.py:81) et N=np.int64(10) rejeté. Friction réelle: K/q/s issus de `P.shape[0]` sont des int Python, mais des valeurs chargées de fichiers/np peuvent être np.integer. Schéma dupliqué à l'identique dans FMatrix.py:97-104 et NoiseCovariance.py:125-132 (duplication de code aussi).

**Preuve :** Test: « np.int64 K: REJECTED -> K must be an integer >= 2, got np.int64(2) »; « q=True: accepted (silently becomes q=1) »; « N=True accepted (1 step) »; « N=np.int64(10) rejected ».

**Suggestion :** Utiliser `isinstance(K, (int, np.integer)) and not isinstance(K, bool)` (idem q, s, N), et factoriser _validate_dims dans un helper commun.

### ✅ [LOW] Collision de nom CSV pour seed=None: deux runs non déterministes s'écrasent mutuellement

`prg/classes/GSSSimulator.py:207-209` — statut : confirmed (1 vote(s)) — catégorie : bug

Avec seed=None, seed_str='random' et le nom de fichier est `simulated_<model>_N<N>_seedrandom.csv` — constant entre runs. Deux exécutions non déterministes successives écrasent silencieusement le même fichier (filepath.open('w') l.245), alors que ce sont précisément les runs non reproductibles qu'on ne peut pas régénérer.

**Preuve :** l.207-208: `seed_str = str(self._seed) if self._seed is not None else 'random'; filename = f'simulated_{model_name}_N{self._N}_seed{seed_str}.csv'`

**Suggestion :** Inclure un horodatage dans le nom quand seed est None (comme le fait déjà _setup_logging dans simulate.py:107-108), ou tirer et mémoriser un seed explicite.

### ✅ [LOW] Itérateur épuisé: une seconde boucle for sur le même simulateur produit 0 élément sans avertissement

`prg/classes/GSSSimulator.py:104-105` — statut : confirmed (1 vote(s)) — catégorie : api

`__iter__` retourne self sans reset; après une première itération complète, une seconde boucle `for ... in sim` se termine immédiatement (vérifié: 3 items puis 0). `run()` se protège via reset() (l.229) mais l'itération directe — l'usage mis en avant dans la docstring du module — non. Risque: une expérience qui compare deux passes obtient silencieusement des statistiques vides.

**Preuve :** Test: « second for-loop yields 0 items (first: 3 ) ».

**Suggestion :** Documenter explicitement, ou faire de __iter__ un reset implicite (attention à la reproductibilité), ou logger un warning si __iter__ est appelé sur un itérateur épuisé.

### ✅ [LOW] _load_model instancie la première sous-classe par ordre alphabétique, pas « le » modèle du module

`prg/simulate.py:160-162` — statut : confirmed (1 vote(s)) — catégorie : robustness

`inspect.getmembers(module, inspect.isclass)` trie alphabétiquement et retourne la première sous-classe de BaseGSSModel trouvée. Si un module de modèle importe une autre classe modèle (p.ex. depuis presets.py ou un modèle voisin pour en dériver), la classe instanciée peut être la mauvaise, silencieusement (le nom du module ne contraint pas la classe choisie).

**Preuve :** simulate.py:160-162: `for _, obj in inspect.getmembers(module, inspect.isclass): if issubclass(obj, BaseGSSModel) and obj is not BaseGSSModel: return obj()`

**Suggestion :** Ne retenir que les classes définies dans le module (`obj.__module__ == module.__name__`), et lever une erreur si plusieurs candidats subsistent.

### ✅ [INFO] Validation positive: le simulateur échantillonne exactement le modèle génératif documenté, graine respectée

`prg/classes/GSSSimulator.py:131-158` — statut : confirmed (1 vote(s)) — catégorie : math

L'ordre régime→état est correct et le conditionnement porte bien sur le NOUVEAU régime r_n pour F, b et Sigma_W, conformément à l'équation Z_{n+1} = F(r_{n+1}) Z_n + b(r_{n+1}) + W_{n+1}: r_n ~ P[r_{n-1}] (l.140) puis z_n = F(r_n) z_{n-1} + b(r_n) + chol_W(r_n) @ N(0,I) (l.144); pas initial r_0 ~ pi0, z_0 = mu_z0(r_0) + chol_z0(r_0) @ N(0,I) (l.133-137). Vérification statistique sur N=4·10^5: max|P_hat − P| = 5.1e-4; les innovations e_n = z_n − F(r_n) z_{n-1} − b(r_n) conditionnées à r_n=k ont moyenne ~0 et covariance empirique égale à Sigma_W(k) à 5e-4 près pour chaque k. Reproductibilité: deux runs même graine identiques, graines différentes divergent. Un seul Generator (default_rng) avec flux séquentiel choice→standard_normal: déterminisme garanti. run() fait reset() avant exécution (l.229), donc le CSV est reproductible même après itération partielle.

**Preuve :** Tests: « same-seed reproducible: True »; « max |Phat-P| = 0.00051 »; « k=0: max|Sig_hat-Sigma_W|=0.0005; k=1: 0.0004 ».

**Suggestion :** Aucun correctif nécessaire sur la logique d'échantillonnage.

### ✅ [INFO] Sigma_W(k) construit par np.block est un stockage indépendant de ses blocs (cohérence garantie uniquement à la construction)

`prg/classes/NoiseCovariance.py:99-107` — statut : confirmed (1 vote(s)) — catégorie : style

Sigma_W(k) (np.block copie) et _Sigma_U/_Delta/_Sigma_V sont des tableaux distincts; la symétrie du bloc est garantie par construction (Delta et Delta.T proviennent du même tableau), et le check SPD + Cholesky est bien hors __debug__ (l.110-119) donc toujours actif — bon point. Mais comme les accesseurs exposent des références mutables (cf. trouvaille aliasing), la cohérence blocs/matrice complète/chol n'est garantie que tant que personne ne mute. Même schéma pour FMatrix._F vs _A/_B/_C/_D (FMatrix.py:89-91).

**Preuve :** NoiseCovariance.py:99-107 `np.block([[self._Sigma_U[k], self._Delta[k]], [self._Delta[k].T, self._Sigma_V[k]]])` puis caches _chol l.119.

**Suggestion :** Couvert par la suggestion writeable=False de la trouvaille aliasing; sinon documenter que les objets sont immuables par convention.

### ✅ [INFO] Tolérances strictes (1e-10) pour row-stochasticité et symétrie: rejets bruyants possibles sur paramètres issus de fichiers arrondis

`prg/utils/matrix_checks.py:36-37` — statut : confirmed (1 vote(s)) — catégorie : numerical-stability

_EPS_STOCH=1e-10 et _EPS_SYM=1e-10 sont plus stricts que la tolérance de numpy rng.choice (~1.5e-8). Une P saisie avec 6 décimales (p.ex. lignes [0.333333]*3, erreur 1e-6) est rejetée à la construction — c'est un rejet BRUYANT (ParamError), donc sain, pas silencieux; mais cela peut surprendre pour des paramètres ronds-tripés via CSV/JSON. À l'inverse, la cohérence est bonne: tout P qui passe la validation passe aussi le check interne de rng.choice, et pi0 validé à 1e-10 (GSSParams.py:232) de même. Aucun paramètre invalide ne passe silencieusement par ce chemin (avec __debug__ actif).

**Preuve :** matrix_checks.py:36-37 `_EPS_SYM = 1e-10; _EPS_STOCH = 1e-10`; GSSParams.py:232 `if abs(total - 1.0) > 1e-10`.

**Suggestion :** Optionnel: accepter une tolérance ~1e-8 et renormaliser explicitement les lignes (avec log) plutôt que rejeter, ou documenter la précision requise.

## Apprentissage (prg/learning/)

_Audit complet de prg/learning/supervised.py (737 lignes) et prg/learning/semi_supervised.py (971 lignes), avec vérification croisée de prg/utils/h5_constraint.py (compute_AB, résiduel H5), prg/classes/GSSSimulator.py (convention de régime) et prg/filter/gss_filter.py (H5_TOL=1e-6). Le cœur mathématique est sain : les estimateurs supervisés sont les bons MLE conditionnels (comptages de transitions ligne 326-337 avec indexation destination r_{n+1} confirmée cohérente avec le simulateur GSSSimulator.py:144 ; OLS par régime avec intercept ; Σ_W MLE /N_k documentée), et le Baum-Welch est correctement implémenté en log-domaine (forward/backward/ξ vérifiés indice par indice, logsumexp partout, aucun risque d'underflow), avec M-step pondéré exact pour P, F, b, Σ_W et placement correct de la projection AB (post-hoc par défaut pour préserver la monotonie, GEM optionnel documenté). Les faiblesses réelles sont de second ordre : la mise à jour heuristique de μ_z0/Σ_z0 invalide la garantie de monotonie stricte annoncée ; les planchers de covariance (eps=1e-8 absolu, blockwise seulement, garde Wsum<1e-12) sont insuffisants contre l'effondrement gaussien, que la sélection du meilleur redémarrage par vraisemblance d'entraînement peut alors récompenser ; les petits régimes en supervisé produisent Σ_W≈0 silencieusement sauvegardée ; le π₀ appris par EM est jeté par le générateur de code ; plus quelques docstrings périmés et incohérences mineures. Aucune fuite train/éval : les scripts estiment sur tout le CSV (pas de split prévu), la colonne r est correctement ignorée en semi-supervisé, et la sélection multi-redémarrage in-sample est la pratique standard pour EM._

### ✅ [MEDIUM] Mise à jour de μ_z0/Σ_z0 non conforme au M-step — la monotonie stricte d'EM annoncée n'est pas garantie

`prg/learning/semi_supervised.py:575-584` — statut : confirmed (2 vote(s)) — catégorie : math

Le terme de vraisemblance impliquant μ_z0(k), Σ_z0(k) est uniquement le terme initial log N(Z_0; μ_z0(k), Σ_z0(k)) pondéré par γ_0(k) (cf. log_init, ligne 517-519, injecté dans _forward ligne 523). Or le M-step met à jour ces paramètres par les moments pondérés par γ_n(k) sur TOUTE la séquence (lignes 576-584), ce qui n'est pas l'argmax de la fonction Q (l'argmax exact serait μ_z0(k)=Z_0 et Σ_z0→0, dégénéré — la régularisation est donc compréhensible). Conséquence : même avec constraint_each_iter=False, la log-vraisemblance n'est PAS garantie monotone non-décroissante, contrairement aux affirmations du docstring (lignes 25-27 « log-likelihood remains monotonically non-decreasing during EM iterations » et 736-739 « the EM iterations are standard (monotone log-likelihood) »). L'effet est faible en pratique (un seul point Z_0 sur N), et une baisse éventuelle est masquée par le critère |ΔlogL| (ligne 535), mais l'affirmation théorique est fausse en l'état.

**Preuve :** Lignes 576-584: « for k in range(K): w = gamma[:, k]; denom_k = w.sum(); ... mu = (w[:, None] * Z).sum(axis=0) / denom_k; cov = (centered.T @ (w[:, None] * centered)) / denom_k » — moments pondérés sur tout n, alors que la Q-fonction ne contient ces paramètres que via le terme n=0 (ligne 517: log_init = log N(Z_0; μ_z0(k), Σ_z0(k))). Accessoirement, le clipping de P_new (lignes 548-549) dévie aussi (de façon négligeable, 1e-300) du M-step exact.

**Suggestion :** Soit (a) figer μ_z0/Σ_z0 après l'initialisation (elles ne sont identifiables qu'à partir d'un seul point), soit (b) les traiter explicitement comme heuristique régularisée et corriger le docstring (« quasi-monotone : le terme initial est mis à jour par une heuristique de moments »), soit (c) ajouter un avertissement runtime si log_lik_history décroît de plus que la tolérance numérique.

### ✅ [MEDIUM] Effondrement de covariance insuffisamment gardé + sélection du meilleur redémarrage par logL : les runs dégénérés peuvent gagner

`prg/learning/semi_supervised.py:299-315, 326-339, 779` — statut : confirmed (2 vote(s)) — catégorie : numerical-stability

Trois mécanismes se combinent : (1) le seul plancher sur Σ_W est _nearest_spd (supervised.py:167-178) avec eps=1e-8 ABSOLU appliqué blockwise à Σ_U et Σ_V seulement — Δ n'est jamais ajusté, donc la matrice jointe Σ_W=[[Σ_U,Δ],[Δᵀ,Σ_V]] peut rester singulière (ex. q=s=1, Σ_W=[[1,1],[1,1]] : blocs ≥ eps mais jointe singulière) ; la Cholesky de _log_mvn_batch retombe alors sur un ridge 1e-8 (lignes 115-118) et la log-densité explose (+~9.2 par dim et par point pour var 1e-8). (2) Le garde-fou de _weighted_fit ne se déclenche que pour Wsum < 1e-12 (ligne 297-298) : un régime avec une masse effective de quelques échantillons (Wsum ~ 1e-3…3) passe en moindres carrés pondérés et produit une Σ_W quasi-singulière sans aucun avertissement. (3) La sélection multi-redémarrage (ligne 779 : « info['log_lik'] > best_info['log_lik'] ») choisit le run de plus grande vraisemblance d'entraînement : c'est précisément le run dégénéré (vraisemblance gaussienne non bornée) qui gagne. C'est le mode d'échec classique d'EM gaussien, ici seulement partiellement mitigé.

**Preuve :** semi_supervised.py:315 « SigW = (residuals.T @ (w[:, None] * residuals)) / Wsum » sans plancher relatif ; _apply_constraints (lignes 244-247) ne clampe que SU et SV via _nearest_spd(eps=1e-8 absolu), Dt inchangé ; _log_mvn_batch lignes 115-118 « except LinAlgError: L = cholesky(Sigma + 1e-8*eye(d)) » ; fit_semi_supervised ligne 779 sélectionne par logL maximum sans détection de dégénérescence.

**Suggestion :** (a) Plancher RELATIF à l'échelle des données : eps_k = max(1e-8, c · trace(Σ_W_global)/dim_z) avec c ≈ 1e-6, appliqué à la matrice jointe Σ_W (pas seulement aux blocs) ; (b) avertir / réinitialiser quand la masse effective Wsum < dim_z+1 (pas 1e-12) ; (c) lors de la sélection du meilleur run, rejeter les runs dont min eig(Σ_W(k)) est sous le plancher (signature de dégénérescence) ou pénaliser via une borne type MAP/prior inverse-Wishart faible.

### ✅ [MEDIUM] Régimes rares : résidus OLS identiquement nuls → Σ_W = 0 clampée à 1e-8·I, modèle sauvegardé pathologique avec simple warning

`prg/learning/supervised.py:228-237, 369-376` — statut : confirmed (2 vote(s)) — catégorie : robustness

Pour N_k ≤ dim_z+1, np.linalg.lstsq (solution de norme minimale, ligne 228) interpole exactement les données : residuals ≡ 0 (ligne 235) donc Σ_W = 0 (ligne 236). _nearest_spd la remonte à ~1e-8·I (lignes 254-255), produisant un régime quasi-déterministe absurde qui est néanmoins écrit tel quel dans le fichier modèle. Le code ne fait qu'un _log.warning (lignes 369-376, « OLS solution may be underdetermined »), sans bloquer ni dégrader proprement. Même pour N_k légèrement > dim_z+1, l'estimateur MLE divisé par N_k (ligne 236, choix documenté) est fortement biaisé vers le bas : le biais multiplicatif est (N_k − dim_z − 1)/N_k puisque dim_z+1 paramètres de régression sont ajustés par ligne — pour N_k = 2(dim_z+1) la covariance est sous-estimée d'un facteur ~2.

**Preuve :** Ligne 236 : « SigW = (residuals.T @ residuals) / N_k » — pas de correction de degrés de liberté (dim_z+1 colonnes de Z_aug, ligne 225) ; lignes 254-255 : « SU = _nearest_spd(SU); SV = _nearest_spd(SV) » avec eps=1e-8 (ligne 167) qui masque le cas Σ_W=0 au lieu de le signaler.

**Suggestion :** Diviser par max(N_k − (dim_z+1), 1) (REML-like) ou au minimum lever une erreur (cohérente avec les ValueError des lignes 333 et 357) quand N_k < dim_z + 1 + marge, plutôt que de sauvegarder silencieusement un régime à bruit ~1e-8. À défaut, élever le warning au niveau ERROR et documenter le biais du diviseur N_k dans le docstring.

### ✅ [LOW] Le générateur de code force pi0=None : le π₀ appris par EM est silencieusement jeté dans le modèle sauvegardé

`prg/learning/supervised.py:534` — statut : confirmed (1 vote(s)) — catégorie : bug

_generate_model_code écrit en dur « pi0: np.ndarray | None = None   # None → stationary distribution » (ligne 534). Pour le supervisé c'est cohérent (fit_supervised retourne pi0=None, documenté ligne 309/424). Mais semi_supervised.main() (lignes 943-950) passe à ce même générateur les params EM dont params['pi0'] est le π₀ estimé (mis à jour ligne 557, permuté dans _reorder_regimes ligne 682) : il est ignoré sans avertissement. Le fichier modèle sauvegardé utilise donc la distribution stationnaire de P au lieu du π₀ ayant servi à calculer la vraisemblance rapportée (best_log_lik), et μ_z0/Σ_z0 appris SONT sauvegardés (lignes 536-537), créant une incohérence partielle entre le modèle ajusté et l'artefact.

**Preuve :** supervised.py:534 « "    pi0: np.ndarray | None = None   # None → stationary distribution", » — aucune référence à params['pi0'] dans _generate_model_code, alors que semi_supervised.py:625-641 inclut « "pi0": pi0 » (ndarray) dans le dict params, et que _reorder_regimes (ligne 682) prend la peine de le permuter.

**Suggestion :** Dans _generate_model_code, sérialiser params['pi0'] quand il est un ndarray (avec le commentaire approprié), et garder None seulement quand params['pi0'] is None. Au minimum, documenter dans semi_supervised que le π₀ appris est volontairement remplacé par la distribution stationnaire dans le fichier généré (défendable : π₀ estimé sur une seule séquence est quasi dégénéré, cf. ligne 557 pi0 = gamma[0]/gamma[0].sum()).

### ✅ [LOW] Quand max_iter est atteint, log_lik rapporté est en retard d'un M-step sur les paramètres retournés ; max_iter=0 → IndexError

`prg/learning/semi_supervised.py:512-537, 642-649` — statut : confirmed (1 vote(s)) — catégorie : bug

La boucle EM fait E-step → test de convergence → M-step. Si la sortie se fait par break (convergence, ligne 537), les paramètres retournés sont ceux du dernier M-step et log_lik_history[-1] est bien leur vraisemblance — cohérent. Mais si la boucle s'épuise par max_iter, le dernier M-step est exécuté APRÈS le dernier E-step : les paramètres retournés sont post-M-step alors que info['log_lik'] (ligne 643) est la vraisemblance pré-M-step. La sélection du meilleur run (ligne 779) compare alors des vraisemblances légèrement périmées (sous-estimées d'un step EM, donc biais conservateur mais incohérent entre runs convergés et non convergés). Cas limite : max_iter=0 fait planter log_lik_history[-1] (IndexError, ligne 643). Par ailleurs le critère |Δ logL| < tol ABSOLU (ligne 535) masque les éventuelles décroissances (GEM ou heuristique μ_z0) et est très strict pour de longues séquences (logL ~ O(N)) ; un critère relatif |ΔlogL|/|logL| serait plus robuste.

**Preuve :** Lignes 533-537 : « if it > 0: delta = log_lik - log_lik_history[-2]; if abs(delta) < tol: converged = True; break » placé AVANT le M-step ; ligne 643 « "log_lik": log_lik_history[-1] » alors que F_list/SigW_list/P/pi0 ont été réécrits par le M-step de la dernière itération quand la boucle sort par épuisement de max_iter.

**Suggestion :** Après la boucle, si non convergé, recalculer une dernière fois la log-vraisemblance avec les paramètres finaux (un E-step léger : _compute_log_emissions + _forward) avant de remplir info ; valider max_iter >= 1 ; envisager un critère relatif sur ΔlogL.

### ✅ [LOW] Projection AB post-hoc appliquée par run AVANT la sélection, mais best_log_lik et le docstring du modèle généré reflètent les paramètres NON contraints

`prg/learning/semi_supervised.py:586-616, 779-787, 952-960` — statut : confirmed (1 vote(s)) — catégorie : api

L'architecture est correcte sur le fond (projection une fois en fin d'EM pour préserver la monotonie, GEM optionnel documenté — points 3 du cahier d'audit satisfaits). Mais : la projection post-hoc est appliquée dans _em_run (lignes 589-616) sur chaque run, PUIS la sélection multi-redémarrage (ligne 779) compare les info['log_lik'] qui sont les vraisemblances des paramètres AVANT projection. Le « meilleur » run au sens non contraint n'est pas nécessairement le meilleur après projection (la projection modifie A, B donc la vraisemblance). De plus, main() écrit dans le docstring du fichier modèle « log L={best_log_lik} » (lignes 952-960) comme si c'était la vraisemblance du modèle sauvegardé (projeté) — c'est en réalité celle du modèle pré-projection. Comportement défendable mais non documenté.

**Preuve :** Ligne 589 : « apply_post_hoc = (not constraint_each_iter) and (constraint is not None or delta_zero) » exécuté dans _em_run avant le return ; ligne 779 : « if best_info is None or info["log_lik"] > best_info["log_lik"] » utilise log_lik_history[-1] (pré-projection) ; lignes 953-956 : « em_note = f"... log L={info['best_log_lik']:.4f} ..." » inséré dans le fichier généré.

**Suggestion :** Soit recalculer la log-vraisemblance des paramètres projetés (un E-step) et la rapporter séparément (log_lik_unconstrained / log_lik_projected) en sélectionnant sur la version projetée, soit documenter explicitement que best_log_lik est pré-projection. Noter aussi que _apply_constraints (lignes 249-259) avale les échecs de compute_AB par un simple warning en mode post-hoc : le modèle sauvegardé peut alors NE PAS satisfaire (H5) alors que son docstring affiche « Constraint : AB » — le filtre h5_exact n'émettra qu'un RuntimeWarning (H5_TOL=1e-6).

### ✅ [LOW] Docstring périmé : « constraint : None | 'a' | 'b' | 'su' » alors que la seule valeur acceptée est 'ab'

`prg/learning/semi_supervised.py:712-713` — statut : confirmed (1 vote(s)) — catégorie : style

Le docstring de fit_semi_supervised liste des cibles de contrainte d'une ancienne API ('a', 'b', 'su') qui n'existent plus : le CLI n'accepte que choices=['ab'] (ligne 822) et tout le code ne teste que constraint == 'ab' (_apply_constraints ligne 249, supervised._fit_regime ligne 258). Un appelant programmatique passant 'a' ou 'su' verrait la contrainte silencieusement ignorée (aucune validation de la valeur dans fit_semi_supervised ni fit_supervised).

**Preuve :** Lignes 712-713 : « constraint : None | 'a' | 'b' | 'su' / H5 projection target. » vs ligne 822 « choices=["ab"] » et ligne 249 « if constraint == "ab": ».

**Suggestion :** Corriger le docstring en « None | 'ab' » et ajouter une validation explicite en tête de fit_semi_supervised / fit_supervised : « if constraint not in (None, 'ab'): raise ValueError(...) » pour éviter l'ignorance silencieuse.

### ✅ [INFO] Commentaires trompeurs : « Laplace smoothing » jamais appliqué ; _LOG_FLOOR (1e-300) utilisé comme seuil de comptage

`prg/learning/semi_supervised.py:410-419, 544-545` — statut : confirmed (1 vote(s)) — catégorie : style

Dans _initialize_params_from_R, le commentaire ligne 410 annonce « with Laplace smoothing for missing rows » mais le code ne fait que remplacer le diviseur nul par 1 puis substituer les lignes nulles par l'uniforme (lignes 413-419) — pas de lissage additif. Dans le M-step, _LOG_FLOOR=1e-300 (défini ligne 89 comme plancher de LOG-probabilité) est réutilisé comme seuil sur des sommes de responsabilités en unités de comptage (lignes 545, 552, 579) — sémantiquement c'est un test « exactement zéro », ça fonctionne mais le nom induit en erreur sur l'intention.

**Preuve :** Ligne 410 : « # P from transition counts (with Laplace smoothing for missing rows) » suivi de « row_sums[row_sums == 0] = 1.0 » ; ligne 545 : « denom_safe = np.where(denom > _LOG_FLOOR, denom, 1.0) » où denom = gamma[:-1].sum(axis=0) est de l'ordre de N.

**Suggestion :** Corriger le commentaire (ou appliquer un vrai lissage +α qui serait d'ailleurs bénéfique à l'initialisation), et introduire une constante dédiée _COUNT_FLOOR (par ex. 1e-12, cohérente avec floor_w de _weighted_fit) pour les seuils de masse.

### ✅ [INFO] Réordonnancement canonique par A[0,0] inopérant sous --delta-zero --constraint ab (A(k) ≡ 0 pour tout k)

`prg/learning/semi_supervised.py:657-683` — statut : confirmed (1 vote(s)) — catégorie : robustness

_reorder_regimes trie les régimes par A[0,0] décroissant pour atténuer le label-switching entre redémarrages. Or avec delta_zero=True et constraint='ab', la projection donne A = Δ Σ_V⁻¹ C = 0 pour tous les régimes (Δ=0) : le critère de tri est constant et l'ordre devient arbitraire (ordre du tri stable de Python), donc la « labellisation canonique » annoncée (docstring lignes 40-42) n'a aucun effet dans ce mode pourtant prévu par le CLI.

**Preuve :** Ligne 672 : « order = sorted(range(K), key=lambda k: -params["A_list"][k][0, 0]) » ; avec delta_zero, _apply_constraints ligne 244-245 fait Dt = 0 puis compute_AB retourne A = Dt @ SV_inv_C = 0 (h5_constraint.py:148).

**Suggestion :** Utiliser un critère insensible à la projection, par ex. trier par D[0,0] ou par trace(Σ_V(k)) ou par la norme de C — ces blocs ne sont pas modifiés par la contrainte AB.

### ✅ [INFO] Incohérences mineures : np.cov ddof=1 vs convention MLE /N ailleurs ; docstring « :.8g » vs code « :.10g »

`prg/learning/supervised.py:397-401, 436-451` — statut : confirmed (1 vote(s)) — catégorie : style

(1) Σ_z0 est estimée via np.cov par défaut (ddof=1, diviseur n−1, ligne 400) alors que Σ_W est explicitement MLE (diviseur N_k, ligne 236) et que le M-step EM divise par Σw — incohérence de convention sans conséquence pratique. (2) Le docstring de _fmt_arr (ligne 442) annonce « :.8g » mais le code utilise « :.10g » (ligne 448). À 10 chiffres significatifs, l'erreur d'arrondi au round-trip est ~1e-10 relative : la contrainte AB du modèle sauvegardé n'est plus exacte à la précision machine mais reste très en dessous du H5_TOL=1e-6 du filtre (gss_filter.py:106) — vérifié, donc pas de risque de RuntimeWarning du filtre, mais utiliser repr() complet (17 chiffres) éliminerait toute perte.

**Preuve :** Ligne 400 : « cov_k = np.cov(Z_k, rowvar=False) » (ddof=1 implicite) vs ligne 236 « SigW = (residuals.T @ residuals) / N_k » ; ligne 442 « formatted with ``:.8g`` » vs ligne 448 « f"{v:.10g}" ».

**Suggestion :** Passer np.cov(..., ddof=0) pour homogénéiser la convention MLE ; corriger le docstring de _fmt_arr ou passer à :.17g pour un round-trip exact des float64.

### ✅ [INFO] _read_csv : la clé de tri int(h[2:]) plante sans message clair sur des colonnes x_*/y_* à suffixe non numérique

`prg/learning/supervised.py:124-131` — statut : confirmed (1 vote(s)) — catégorie : robustness

Les colonnes sont détectées par préfixe x_/y_ puis triées par int(suffixe). Une colonne nommée par ex. « x_true » ou « y_pred » (plausible dans un CSV enrichi de résultats de filtrage) lève une ValueError brute dans la lambda de tri, hors du bloc try/except des lignes 148-156, donc sans le message d'erreur contextualisé prévu par le module. Utilisé aussi par semi_supervised via l'import ligne 77-83.

**Preuve :** Lignes 124-131 : « x_cols = sorted([h for h in header if h.startswith("x_")], key=lambda h: int(h[2:])) » — int('true') → ValueError non interceptée.

**Suggestion :** Filtrer sur un motif strict (re.fullmatch(r"x_\d+", h)) avant le tri, et lever un ValueError explicite listant les colonnes ignorées/ambiguës.

---
_Généré automatiquement depuis le résultat du workflow ; les justifications complètes des votes adversariaux sont dans `raw/01-code-core-result.json`._