# Vague 3 — Audit mathématique du papier

Workflow `audit-paper-math` (run `wf_1ee834ca-798`) + 4 lots de vérification progressive
(`verify-batch`, runs `wf_c0228168-40f`, `wf_ac678231-b6d`, `wf_4240165c-a9d`, `wf_5c10da40-d11`).
Méthode : 6 finders par section + vérification adversariale (2 lentilles MATHS/CONTEXTE pour
critical/high/medium, 1 pour low/info). Détails : `raw/03-paper-math-extracted.json`.

**Bilan : 80 trouvailles — 76 confirmées** (4 critical, 18 high, 25 medium, 20 low, 9 info), 2 incertaines, 2 réfutées.

## Trouvailles majeures (critical + high confirmées)

- **[CRITICAL] Eq. (22) : variance d'innovation de paire S^{(j,k)} fausse — mélange de conditionnements, contredit la Proposition 1** — `paper/sections/03_filtering.tex:170-175 (eq. (22), label eq:S_jk)`
- **[CRITICAL] eq:H5_in_X n'est PAS équivalente à eq:H5_compact — le passage clé de la preuve de nécessité est invalide** — `/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/appendix/C_projections.tex:31-39, eq:H5_in_X`
- **[CRITICAL] μ_jk omet le terme de biais C_k(b_{X,j} − Δ_jΣ_{V,j}⁻¹b_{Y,j}) : le noyau (R,Y) du papier est faux dès que b_X ≠ ΔΣ_V⁻¹b_Y** — `/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/02_model_h5.tex:109-112 (eq:mu_jk), Proposition 1`
- **[CRITICAL] eq:(S_jk): conditionnements incohérents, présenté comme exact sans dérivation** — `/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/03_filtering.tex:170-187 (eq:S_jk, eq:M_tilde); cf. appendix/B_h5_derivation.tex:62-70`
- **[HIGH] Eqs (21)-(23) : conditionnement \yn = y_{1:n} incohérent — Cov(Y_{n+1},Y_n|·,y_{1:n}) ≡ 0 et Σ_{YY,n}(j)^{-1} = 0^{-1} sous la boucle de la Remarque 4** — `paper/sections/03_filtering.tex:166-183 (eqs (21)-(23), labels eq:y_pred_jk, eq:S_jk, eq:M_tilde)`
- **[HIGH] Argument d'exactitude de l'étape (III) erroné : la prédictive p(X_{n+1}|r_{n+1}=k, y_{1:n}) N'EST PAS « déjà une seule gaussienne »** — `paper/sections/03_filtering.tex:217-228 (texte entre eqs (26) et (27))`
- **[HIGH] Preuve du retournement temporel : (H5bis) est utilisée à CHAQUE facteur rétrograde, pas seulement au dernier — l'attribution est inversée** — `paper/appendix/A_time_reversal.tex:13-23 (preuve du collapse (25)-(26))`
- **[HIGH] La monotonie EM affirmée n'est pas garantie par le M-step réellement spécifié (moments initiaux, clamp SPD, resets)** — `paper/appendix/D_baum_welch.tex (+ paper/sections/05_estimation.tex):App D lignes 80-112 ; 05_estimation.tex lignes 131-140`
- **[HIGH] Élimination de Σ(r) incohérente : Z(r) « indépendant de Σ(r) » alors que la solution finale Z = Σ_V⁻¹Δᵀ dépend des blocs de Σ(r)** — `/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/appendix/C_projections.tex:43-67 (eq:split_a, eq:split_b)`
- **[HIGH] Nécessité survendue : « the unique closed-form solution » contredit la propre docstring du code (non-unicité si K·s < q+s)** — `/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/04_constraint.tex:62-79 et 90-92 (eq:AB)`
- **[HIGH] Step 4 contient trois identités fausses (dont une dimensionnellement impossible) — la conclusion eq:h5_compact_app reste juste** — `/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/appendix/B_h5_derivation.tex:138-159 (Step 4)`
- **[HIGH] Le décret « noise law » (Var(Z_n|r_n=j) = P(j)) n'est pas justifié par l'annexe A citée, et la Sec. 4 confond covariance de bruit et covariance marginale par régime** — `/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/appendix/B_h5_derivation.tex:22-27 (Step 1) ; aussi 04_constraint.tex l.53-64 et 46-47 (Step 2)`
- **[HIGH] « Sufficient and necessary to enforce j=k » est faux tel quel ; et nulle part le papier ne prouve AB ⇒ (H5) pour toutes les paires (j,k)** — `/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/appendix/B_h5_derivation.tex:100-104 (Step 3) ; 04_constraint.tex l.81-89 (Prop. 4.1)`
- **[HIGH] Remarque 2 (« Bias on X is invisible ») fausse : b_X du régime source entre dans le noyau dès que C ≠ 0** — `/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/02_model_h5.tex:120-130 (Remark rem:bX_invisible)`
- **[HIGH] (H3), 2e partie : conditionnement sur (x_n, y_n) manquant — telle qu'écrite, l'hypothèse contredit le modèle qu'elle définit** — `/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/02_model_h5.tex:21-25 (eq:H3, seconde équation)`
- **[HIGH] Preuve A : la factorisation rétrograde n'est PAS obtenue par « chain rule + (H1) » seuls — (H5bis) est utilisé au mauvais endroit** — `paper/appendix/A_time_reversal.tex:13-22 (factorisation display l.15-20 et phrase l.21-22)`
- **[HIGH] Énoncé faux : « for any matrix K » avec membre de gauche Var[X_{n+1} | r_{n+1}=k, y_{n+1}]** — `paper/appendix/E_joseph.tex:53-62 (eq:joseph_proof)`
- **[HIGH] Moments de régime (μ_n(k), P_n(k)): trois conventions de conditionnement incompatibles** — `/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/03_filtering.tex:35-38, 166-183, 285-302, 361-363`
- **[HIGH] Symbole P surchargé avec cinq sens différents (idem R, et trois notations pour Σ_W)** — `/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/04_constraint.tex:04:29; 03:87,156; 05:9,25; 06:18,25; B:16-22,110`
- **[HIGH] \Vark/\Covk impriment des moments NON centrés mais sont utilisés comme covariances centrées** — `/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/macros.tex:macros:23-26; 02:152-157; 03:32,103-105; B:62-70,119-127; E:57`
- **[HIGH] Step 4 de la dérivation H5 : justifications fausses et équation intermédiaire dimensionnellement invalide** — `/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/appendix/B_h5_derivation.tex:136-152 (Step 4)`
- **[HIGH] « Équivalence » eq:H5_in_X non établie + condition de nécessité « K·s ≥ q+s » fantôme et revendications incohérentes** — `/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/appendix/C_projections.tex:C:31-39; 01:84-85; 04:67-68; abstract:33-34`

## Section 3 — Filtrage (cœur mathématique)

_Audit complet de paper/sections/03_filtering.tex (366 lignes, eqs (14)-(37) du PDF), avec re-dérivation manuelle de chaque équation, croisement avec 02_model_h5.tex (Prop. 1), les appendices A (retournement temporel) et E (Joseph), le code prg/filter/gss_filter.py et le script verify_19_20_equivalence.py (relancé : PASS à 1e-16). Verdict global : l'architecture du filtre est exacte sous (H5) et la plupart des équations sont correctes — j'ai validé (15)-(18) (propagation, y compris les termes de biais et la PSD du centré), (19), (24), (27)-(31) (étape Kalman, exacte sous H5), (32) (algèbre de Joseph re-vérifiée), (33)-(37), et la moyenne (21)+(23) qui redonne exactement μ_jk(y_n) de la Prop. 1. Le défaut majeur est l'équation (22) (S_jk, l.170-174) : sa variance mélange un terme marginal sur r_n (Σ_YY,n+1(k)) avec une cross-covariance conditionnelle à la paire — c'est une erreur d'énoncé présentée comme exacte (aucune mention d'approximation), en contradiction interne avec Γ_jk de la Prop. 1 du même papier, confirmée divergente dans le code ; correction minimale : S^(j,k) = [F_k Σ_n(j) F_kᵀ + Σ_W(k)]_YY = C_k P_{n|n}^(j) C_kᵀ + Σ_V(k) = Γ_jk sous H5. S'y ajoutent : un conditionnement y_{1:n} incohérent dans (22)-(23) (rendant M̃ indéfini sous le bouclage postérieur de la Remarque 4), un argument d'exactitude de l'étape (III) fondé sur une affirmation fausse (la prédictive n'est pas « déjà une gaussienne »), une preuve d'appendice A qui attribue (H5) au mauvais facteur, et une Remarque 2 qui confond gain de Kalman et statistiques de paires. Aucune complexité O(K²) n'est revendiquée (l'affirmation « indépendant de n » est correcte) ; pas de problème dimensionnel q≠s, mais les hypothèses d'inversibilité ne sont jamais posées._

### ✅ [CRITICAL] Eq. (22) : variance d'innovation de paire S^{(j,k)} fausse — mélange de conditionnements, contredit la Proposition 1

`paper/sections/03_filtering.tex:170-175 (eq. (22), label eq:S_jk)` — statut : confirmed (2 vote(s)) — catégorie : math-error

L'équation (22) écrit S^{(j,k)} = Σ_{YY,n+1}(k) − Cov(Y_{n+1},Y_n|j,k)·Σ_{YY,n}(j)^{-1}·Cov(Y_n,Y_{n+1}|j,k). Le premier terme Σ_{YY,n+1}(k) est, par l'étape (I) (eqs 15-18), la variance MARGINALE de Y_{n+1} sachant (r_{n+1}=k, y_{1:n}) — c.-à-d. mélangée sur r_n — alors que le terme soustrait est conditionnel à la PAIRE (r_n=j, r_{n+1}=k). Ce n'est donc pas un complément de Schur valide : S^{(j,k)} peut être non-PSD, et la vraisemblance (20) qui l'utilise dans la mise à jour des poids (24) n'est plus exacte — ce qui contamine π_{n+1}, donc les étapes (I) et (IV) suivantes. Re-dérivation : la variance exacte est Var(Y_{n+1}|r_n=j, r_{n+1}=k, y_{1:n}) = [F_k Σ_n(j) F_kᵀ + Σ_W(k)]_{YY} = C_k P_{n|n}^{(j)} C_kᵀ + Σ_V(k), qui sous (H5) vaut exactement Γ_{jk} de la Proposition 1 (eq. 11) — j'ai vérifié l'identité C_k(Σ_{U,j} − Δ_jΣ_{V,j}^{-1}Δ_jᵀ)C_kᵀ + Σ_{V,k} = Γ_{jk}. La version (22) est donc en CONTRADICTION INTERNE avec la Prop. 1 du même papier : la vraisemblance de paire exacte est N(μ_{jk}(y_n), Γ_{jk}), invariante en n, alors que (22) donne un objet dépendant de n et du mélange sur j. Rien dans le texte ne présente (22) comme une approximation : la section affirme partout l'exactitude (lignes 14-15, 44-46, 217). C'est une erreur d'énoncé (aucune dérivation de (20)-(23) n'est donnée dans le papier), pas un choix assumé. Confirmé par le code : prg/filter/gss_filter.py lignes 793-809 implémente (22) littéralement (Gamma = S_YY_np1 − M_t·Cov_Ynp1_Ynᵀ) et c'est le mode imm_general dont la divergence sous (H5) a été établie numériquement (max|Δπ|≈0.25-0.46).

**Preuve :** 03_filtering.tex l.170-174 : « S_{n+1}^{(j,k)} = \Sigma_{YY,n+1}(k) − \Covk{Y_{n+1}}{Y_n}{j, k, \yn}\,\Sigma_{YY,n}(j)^{-1} × \Covk{Y_n}{Y_{n+1}}{j, k, \yn} » ; à comparer à 02_model_h5.tex l.113-116 : « Γ_{jk} = Σ_{V,k} + C_kΣ_{U,j}C_kᵀ − C_kΔ_jΣ_{V,j}^{-1}Δ_jᵀC_kᵀ » qui est la variance exacte de la même loi p(y_{n+1}|r_n=j,r_{n+1}=k,y_n) selon (9)-(11). gss_filter.py l.809 : Gamma = _psd_floor(_sym(S_YY_np1 - M_t @ Cov_Ynp1_Yn.T)) — le _psd_floor trahit la non-PSD possible.

**Suggestion :** Correction minimale : remplacer le premier terme de (22) par la variance prédictive conditionnelle à la paire, S^{(j,k)} = [F_k Σ_n(j) F_kᵀ + Σ_W(k)]_{YY} − M̃ Σ_{YY,n}(j) M̃ᵀ (lecture « moments prédits »), ou, avec les moments postérieurs de la Remarque 4 : S^{(j,k)} = C_k P_{n|n}^{(j)} C_kᵀ + Σ_V(k) sans terme de régression. Plus simple encore : invoquer la Prop. 1 et poser Λ_{n+1}(j,k) = N(y_{n+1}; μ_{jk}(y_n), Γ_{jk}) (exact, fermé, invariant en n) — les eqs (21)-(23) deviennent superflues sous (H5).

**Ajustement de sévérité (vérificateurs) :** Keep critical. The error sits in the mode-probability update of the main recursion, contaminates pi_{n+1} and the subsequent steps (I)/(IV), and directly falsifies the paper's headline exactness claim for the general (time-varying) mode - the mode Remark 1 recommends in transient/non-stationary settings. Sole mitigation: the h5-exact stationary mode bypasses (22) via Gamma_jk and remains correct, so the contribution survives once (22) is fixed (pair-conditional predictive variance, or directly invoking Prop. 1's Gamma_jk under H5). | Keep critical. Eq (22) sits in the mode-probability update at the heart of the paper's central claim of exactness; it contaminates pi_{n+1} via eqs (20)/(24) and all downstream steps, contradicts Prop. 1 internally, and the numerically confirmed divergence under (H5) is large (max|Delta pi|~0.25-0.46). The fix is local (invoke Prop. 1: Lambda = N(mu_jk(y_n), Gamma_jk), or use the pair-conditional first term [F_k Sigma_n(j) F_k^T + Sigma_W(k)]_YY), but as written the main recursion of Section 3 is false while being advertised as exact.

### ✅ [HIGH] Eqs (21)-(23) : conditionnement \yn = y_{1:n} incohérent — Cov(Y_{n+1},Y_n|·,y_{1:n}) ≡ 0 et Σ_{YY,n}(j)^{-1} = 0^{-1} sous la boucle de la Remarque 4

`paper/sections/03_filtering.tex:166-183 (eqs (21)-(23), labels eq:y_pred_jk, eq:S_jk, eq:M_tilde)` — statut : confirmed (2 vote(s)) — catégorie : math-error

Les cross-covariances de (22)-(23) sont écrites conditionnelles à \yn = y_{1:n} (macro confirmée, macros.tex l.64). Or conditionner sur y_{1:n} rend Y_n dégénéré (Y_n ≡ y_n p.s.), donc Cov(Y_{n+1},Y_n|j,k,y_{1:n}) = 0 et M̃ = 0 : la correction de régression de (21) s'annule identiquement et (22) se réduit à S = Σ_{YY,n+1}(k). Pire : si, conformément à la Remarque 4 (l.285-302), les moments (μ_n(j),P_n(j)) injectés sont les moments POSTÉRIEURS (bloc YY de Σ_n(j) nul), alors Σ_{YY,n}(j)^{-1} dans (23) est l'inverse d'une matrice nulle — l'équation est indéfinie dans la récursion même du papier. La lecture cohérente (celle du code, gss_filter.py l.803-812) demande des moments PRÉDITS conditionnés à y_{1:n−1} : sous cette lecture j'ai vérifié que la MOYENNE (21)+(23) est exacte sous (H5) (M̃ = C_kM_j + D_k avec M_j = Σ_{XY,n}(j)Σ_{YY,n}(j)^{-1} = Δ_jΣ_{V,j}^{-1}, et (21) = μ_{jk}(y_n) + b_{Y,k}-terme, conforme à eq. (10)) — seule la variance (22) reste fausse. Donc deux défauts distincts : (a) conditionnement noté y_{1:n} au lieu de y_{1:n−1} ; (b) surcharge du symbole (μ_n, P_n) entre prédits et postérieurs qui rend les formules tantôt indéfinies, tantôt approximatives.

**Preuve :** macros.tex l.64 : \newcommand{\yn}{y_{1:n}} ; 03_filtering.tex l.172 : \Covk{Y_{n+1}}{Y_n}{j, k, \yn} ; l.181 : \widetilde M = \Covk{Y_{n+1}}{Y_n}{j,k,\yn}\,\Sigma_{YY,n}(j)^{-1} ; Remarque 4 l.297-301 : les moments postérieurs « serve as (μ_{n+1}(k), P_{n+1}(k)) in step (I) of the next time step » — bloc YY postérieur nul puisque Y_{n+1} = y_{n+1} est observé (l.294).

**Suggestion :** Écrire explicitement les conditionnements : moments prédits μ_{n|n−1}(j) := E[Z_n|r_n=j, y_{1:n−1}] dans (21)-(23) avec Cov(·|j,k,y_{1:n−1}), OU passer aux moments postérieurs et supprimer la régression : ŷ^{(j,k)} = C_k x̂_{n|n}^{(j)} + D_k y_n + b_{Y,k}, S^{(j,k)} = C_k P_{n|n}^{(j)} C_kᵀ + Σ_V(k). Introduire deux symboles distincts (prédit / postérieur) pour lever la surcharge.

**Ajustement de sévérité (vérificateurs) :** Keep high, with two caveats: (i) the (22)-variance error itself overlaps the already-confirmed wave-1/2 finding — the new content is the y_{1:n} vs y_{1:n−1} conditioning error and the predicted/posterior symbol overload that renders (23) indefinite (0^{-1}) under the paper's own Remark-4 loop; (ii) the defect is paper-only (notation/exposition in the central mode-update equations of a section claiming exactness) — the code implements the coherent reading, so the fix is editorial: state the conditioning y_{1:n−1} explicitly and introduce distinct predicted/posterior symbols (or move to the posterior formulation ŷ = C_k x̂_{n|n}^{(j)} + D_k y_n + b_{Y,k}, S = C_k P_{n|n}^{(j)} C_kᵀ + Σ_V(k)). | Keep high. The defect makes Step (II) of the paper's central recursion indefinite (inverse of a zero matrix) under the paper's own definitions and Remark 4, so a reader cannot execute the published algorithm as written. One calibration note: it shares eqs (21)-(23) with the already-confirmed S_jk variance bug (wave 1-2) — report it as a distinct notational/conditioning defect (wrong conditioning set y_{1:n} vs y_{1:n-1}, plus predicted/posterior symbol overload), not as a second independent numerical error, to avoid double-counting. The proposed fix (explicit predicted/posterior symbols, or posterior form yhat = C_k xhat + D_k y_n + b_{Y,k} with S = C_k P C_k^T + Sigma_V(k)) is correct and matches both prop:markov and the code's h5_exact Gamma (gss_filter.py ll.431-461).

### ✅ [HIGH] Argument d'exactitude de l'étape (III) erroné : la prédictive p(X_{n+1}|r_{n+1}=k, y_{1:n}) N'EST PAS « déjà une seule gaussienne »

`paper/sections/03_filtering.tex:217-228 (texte entre eqs (26) et (27))` — statut : confirmed (2 vote(s)) — catégorie : derivation-gap

Le texte affirme que le mélange p(X_{n+1}|r_{n+1}=k, \yn) = Σ_j μ_{j|k} p(X_{n+1}|r_n=j,r_{n+1}=k,\yn) « is already a single Gaussian, namely p(X_{n+1}|r_{n+1}=k, y_{n+1}) ». C'est faux à deux titres : (i) les ensembles de conditionnement diffèrent (y_{1:n} à gauche, y_{n+1} à droite) — une densité ne peut pas être égale à une densité conditionnée à une autre variable ; (ii) la loi PRÉDICTIVE sachant y_{1:n} est bel et bien un mélange gaussien sur r_n (H5 ne donne aucun collapse sans conditionner sur y_{n+1}). De même, l.226-228, « The Kalman update is therefore a Gaussian conditioning of the regime-conditional joint p(Z_{n+1}|r_{n+1}=k, \yn) » : ce joint n'est pas gaussien. La CONCLUSION reste vraie — j'ai vérifié que la mise à jour par moments (27)-(31) est exacte sous (H5) : chaque composante j du mélange a la MÊME conditionnelle p(x_{n+1}|r_{n+1}=k, y_{n+1}) (par le collapse (25)-(26) appliqué composante par composante), et quand toutes les composantes partagent une conditionnelle linéaire-gaussienne commune, les formules Σ_XY Σ_YY^{-1} appliquées aux moments du mélange restituent exactement cette conditionnelle (Σ_XY = M_kΣ_YY et Σ_XX = G_k + M_kΣ_YY M_kᵀ par variance totale). Mais cet argument — le seul qui justifie (27)-(31) — n'est nulle part dans le papier ; tel qu'écrit, le raisonnement central d'exactitude comporte une étape fausse.

**Preuve :** l.218-224 : « the Gaussian mixture p(X_{n+1}\mid r_{n+1}=k, \yn) … is unnecessary here because that posterior is already a single Gaussian, namely p(X_{n+1}\mid r_{n+1}=k, y_{n+1}) » ; l.226-228 : « a Gaussian conditioning of the regime-conditional joint p(Z_{n+1}\mid r_{n+1} = k, \yn) on Y_{n+1} = y_{n+1} ».

**Suggestion :** Reformuler : « la prédictive est un mélange, mais après conditionnement sur Y_{n+1}=y_{n+1} chaque composante (j) donne la même conditionnelle p(x_{n+1}|r_{n+1}=k,y_{n+1}) (conséquence de (25)-(26)) ; le posterior filtré est donc une gaussienne unique, et le conditionnement linéaire sur les moments du mélange (27)-(31) la restitue exactement (loi de la variance totale + Σ_XY = K S). » Ajouter ces deux lignes de preuve.

**Ajustement de sévérité (vérificateurs) :** Keep high (or at minimum medium-high). It is a false statement inside the bolded central exactness argument of the paper — a reviewer would flag it immediately and it is the only main-text justification of step (III). However, it does not invalidate any equation, result, or code: the recursion is exact and independently grounded in Appendix A and the companion paper. The fix is the two-line proof proposed in the finding (per-component shared conditional + law of total variance), which I verified is correct. | Keep high for a paper-correctness audit: the only printed justification of the paper's headline exactness claim is a false mathematical statement (predictive equated to a posterior; non-Gaussian joint called Gaussian), in the section explicitly titled around exactness. However, note the formulas (27)-(31) and the algorithm are correct and the final result is unaffected — if the severity scale reserves "high" for incorrect results/formulas rather than incorrect proofs, downgrade to medium. The proposed fix (two-line argument: common component-wise conditional via (25)-(26) + total-variance/Σ_XY=KS identity) is the right repair.

### ✅ [HIGH] Preuve du retournement temporel : (H5bis) est utilisée à CHAQUE facteur rétrograde, pas seulement au dernier — l'attribution est inversée

`paper/appendix/A_time_reversal.tex:13-23 (preuve du collapse (25)-(26))` — statut : confirmed (2 vote(s)) — catégorie : derivation-gap

La preuve affirme que la factorisation affichée découle « du chain rule et de (H1) », puis que (H5) n'intervient que via le dernier facteur p(x_{n+1}|w_{n+1}). C'est inexact. La factorisation rétrograde exacte de la chaîne (X_m, W_m) donne p(x_{1:n+1}, w_{1:n+1}) = p(w_{n+1})p(x_{n+1}|w_{n+1}) Π_m p(w_m|x_{m+1},w_{m+1}) p(x_m|w_m,x_{m+1},w_{m+1}). Pour obtenir le premier crochet du papier [p(w_1|w_2)···p(w_n|w_{n+1})p(w_{n+1})], il faut remplacer p(w_m|x_{m+1},w_{m+1}) par p(w_m|w_{m+1}) à CHAQUE m ≤ n — c'est précisément (H5bis) (eq. 5), appliquée n fois. À l'inverse, le facteur final p(x_{n+1}|w_{n+1}) sort gratuitement du chain rule (p(x_{n+1},w_{n+1}) = p(w_{n+1})p(x_{n+1}|w_{n+1})) et n'utilise PAS (H5). La conclusion p(x_{n+1}|w_{1:n+1}) = p(x_{n+1}|w_{n+1}) est correcte, mais la preuve telle qu'écrite est fausse dans son attribution des hypothèses : un lecteur ne peut pas la vérifier ligne à ligne. Cette preuve soutient (25)-(26), pilier de l'exactitude de la Section 3.

**Preuve :** l.13-14 : « Repeated application of the chain rule and (H1) yields » [la factorisation avec p(w_m|w_{m+1}) sans x_{m+1}] ; l.21-22 : « The factor p(x_{n+1}|w_{n+1}) at the end of the second bracket is the time-reversed counterpart of (H5) ».

**Suggestion :** Réécrire : factoriser d'abord en p(w_m|x_{m+1},w_{m+1})·p(x_m|w_m,x_{m+1},w_{m+1}), puis invoquer (H5bis) pour chaque facteur p(w_m|x_{m+1},w_{m+1}) = p(w_m|w_{m+1}), en notant que p(x_{n+1}|w_{n+1}) provient du seul chain rule. Préciser aussi que la division finale par p(w_{1:n+1}) utilise la markovianité de (R,Y) (ou s'obtient en intégrant l'identité sur x_{n+1}).

**Ajustement de sévérité (vérificateurs) :** Keep high. The flawed step is the load-bearing justification in the unique proof of eq:H5_collapse_mean/var, which 03_filtering.tex (line 217) calls the key feature making the IMM exact; as written the proof is unverifiable line-by-line and asserts a false implication ((H1)+chain rule => display). Mitigating factor: the lemma is true and the fix is a local rewrite, so this is a proof-rigor defect, not a false result. | Keep high (proof of the paper's central exactness lemma is unverifiable as written; a referee would flag it). If the audit scale reserves 'high' for false results or numerical impact, medium is acceptable: the lemma is true, the code is unaffected, and the fix is a local rewrite of the proof's hypothesis attribution.

### ✅ [MEDIUM] Remarque 2 : le gain stationnaire K^{(k)} et S^{(k)} ne « matchent » pas Γ_{jk} et μ_{jk}(·) — mauvais objets ; la vraie forme fermée (validée par le script) n'est ni énoncée ni prouvée

`paper/sections/03_filtering.tex:48-64 (Remark 2, rem:h5exact)` — statut : confirmed (2 vote(s)) — catégorie : claim-overreach

La remarque affirme qu'en mode h5-exact « the Kalman gain K^{(k)} and the innovation covariance S^{(k)} are also constant, matching the explicit expressions Γ_{jk} and μ_{jk}(·) of Proposition 1 ». Confusion d'objets : Γ_{jk}/μ_{jk} sont les statistiques de la VRAISEMBLANCE DE PAIRE (étape II, indexées (j,k)) ; K^{(k)}/S^{(k)} sont le gain et la covariance d'innovation du Kalman conditionnel au mode (étape III, indexées k seul). Ils ne coïncident pas : S^{(k)} stationnaire = Σ_{YY}^∞(k) (mélange sur j) ≠ Γ_{jk}. La vraie forme fermée — que j'ai re-dérivée (sous AB : X_{n+1} = Δ_kΣ_{V,k}^{-1}Y_{n+1} + (b_{X,k} − Δ_kΣ_{V,k}^{-1}b_{Y,k}) + W' avec W' ⊥ (Y_{n+1}, passé) sachant r_{n+1}=k) et que scripts/verify_19_20_equivalence.py confirme à 1e-16 — est : K^{(k)} = Δ_kΣ_{V,k}^{-1} et P_{n+1|n+1}^{(k)} = Σ_{U,k} − Δ_kΣ_{V,k}^{-1}Δ_kᵀ. Ce résultat clef (le gain ne dépend QUE de la covariance de bruit) n'apparaît nulle part dans le papier. Par ailleurs la remarque inverse la hiérarchie de fiabilité : avec (22) telle qu'imprimée, le mode « général » n'est PAS exact même sous (H5) (divergence confirmée numériquement), alors que le mode h5-exact l'est ; et la « explicit characterisation of when the two modes agree » promise par l'intro (01_introduction.tex l.69-72) n'est jamais donnée.

**Preuve :** l.56-59 : « the Kalman gain K^{(k)} and the innovation covariance S^{(k)} are also constant, matching the explicit expressions Γ_{jk} and μ_{jk}(·) of Proposition~\ref{prop:markov} » ; l.62-63 : « The general (time-varying) mode remains more reliable » ; sortie du script : ‖M_a − M_b‖_F ≤ 2.7e-16, ‖Γ_a − Γ_b‖_F ≤ 9.2e-16 (M_b = ΔΣ_V^{-1}, Γ_b = Σ_U − ΔΣ_V^{-1}Δᵀ).

**Suggestion :** Corriger la remarque : « K^{(k)} = Δ_kΣ_{V,k}^{-1}, P^{(k)} = Σ_{U,k} − Δ_kΣ_{V,k}^{-1}Δ_kᵀ (forme fermée, indépendante de la récursion (16)-(18)), et les statistiques de paire stationnaires coïncident avec μ_{jk}, Γ_{jk} de la Prop. 1 ». Ajouter un court lemme avec la preuve W' (3 lignes, incluant le terme correctif b_X − ΔΣ_V^{-1}b_Y pour b_r ≠ 0), ce qui fournirait au passage la caractérisation promise par l'intro.

**Ajustement de sévérité (vérificateurs) :** Keep medium. The error sits in a Remark rather than a theorem, but it misstates a mathematical identity in print, the missing closed-form lemma is the natural centerpiece of the h5-exact mode, and the unfulfilled "explicit characterisation" is part of stated Contribution 1 — solidly medium, at the upper end. | Keep medium. The error sits in a remark rather than a theorem, but it is a false mathematical assertion, the inverted reliability claim misguides practitioners between the two implemented modes, and the unfulfilled "explicit characterisation" promise is a referee-visible gap in a stated contribution. One caveat for aggregation: the reliability-hierarchy sub-claim partially overlaps the already-confirmed eq:S_jk finding and should not be double-counted; the independently novel core (wrong objects + missing closed-form lemma K^{(k)}=Δ_kΣ_{V,k}^{-1}, P^{(k)}=Σ_{U,k}−Δ_kΣ_{V,k}^{-1}Δ_k^T) justifies medium on its own, especially since the proposed three-line lemma would simultaneously fix the remark and deliver the promised characterisation.

### ✅ [MEDIUM] Incohérence de bouclage : Remarque 4 (moments postérieurs) vs init §3.6 (moments a priori) vs code (moments prédits, jamais bouclés)

`paper/sections/03_filtering.tex:285-302 (Remark 4) vs 361-366 (init) vs eqs (21)-(23)` — statut : confirmed (2 vote(s)) — catégorie : derivation-gap

Trois descriptions incompatibles du même recyclage de moments : (a) la Remarque 4 dit que les moments POSTÉRIEURS (μ^post avec bloc Y = y_{n+1}) alimentent l'étape (I) suivante « ensuring that the prediction at time n+2 conditions on the full history » ; (b) l'initialisation §3.6 (l.361-364) alimente la première étape (I) avec les moments A PRIORI μ_1(k) = μ_{z_0}(k), P_1(k) = Σ_{z_0}(k) + μμᵀ, NON conditionnés à y_1 — en contradiction avec la logique de (a) (l'information de y_1 n'entre que via π_1) ; (c) le code imm_general (gss_filter.py l.871-874 : self._mu = mu_np1, self._P_z = P_np1) stocke les moments PRÉDITS et ne fait jamais le bouclage postérieur de la Remarque 4 — le papier décrit donc une architecture que l'implémentation de référence ne suit pas. Sous la lecture (a), la récursion est exacte (je l'ai vérifié étape par étape) mais (23) devient 0·0^{-1} ; sous (b)/(c), les moments ne sont jamais conditionnés aux observations et la récursion n'est plus exacte même avec (22) corrigée.

**Preuve :** Remark 4, l.297-301 : « These posterior values … serve as (μ_{n+1}(k), P_{n+1}(k)) in step (I) of the next time step » ; §3.6 l.361-363 : « The regime moments at n = 1 are simply μ_1(k) = μ_{z_0}(k) and P_1(k) = Σ_{z_0}(k) + μ_{z_0}(k)μ_{z_0}(k)ᵀ, which feed the first application of Step (I) » ; gss_filter.py l.871-874.

**Suggestion :** Choisir et écrire UNE architecture : bouclage postérieur partout (alors §3.6 doit injecter (x̂_{1|1}^{(k)}, P_{1|1}^{(k)}, y_1) dans la première étape (I), et (21)-(23) se simplifient en formes sans régression), et aligner le code imm_general ou signaler explicitement qu'il implémente une variante héritée différente.

**Ajustement de sévérité (vérificateurs) :** Maintenir medium. Ne pas escalader : la sous-affirmation « plus exacte même avec (22) corrigée » est probablement fausse sous AB (robustesse structurelle aux moments), donc le défaut est surtout une incohérence d'exposition/architecture, pas une perte d'exactitude démontrée du filtre implémenté.

### ✅ [MEDIUM] Macros \Covk/\Vark définies comme moments NON centrés mais utilisées comme covariances centrées — eq. (17) littéralement fausse sous la définition de la macro

`paper/macros.tex:22-26 ; 03_filtering.tex l.31-32, 103-105, 171-173, 336-337 ; 02_model_h5.tex l.152-157` — statut : confirmed (2 vote(s)) — catégorie : notation

macros.tex définit \Covk{A}{B}{c} := E[ABᵀ|c] et \Vark{A}{c} := E[AAᵀ|c] (seconds moments non centrés). Or la plupart des usages sont des covariances centrées : (i) eq. (17) (l.103-105) : sous la définition littérale, E[Z_{n+1}Z_nᵀ|j,k,y_{1:n}] = F_kP_n(j) + b_kμ_n(j)ᵀ, ce qui ne vaut PAS le membre de droite F_k(P_n(j) − μ_n(j)μ_n(j)ᵀ) — l'équation n'est correcte que si \Covk désigne la covariance centrée ; (ii) P_{n|n}^{(k)} := \Vark{X_n}{r_n=k,\yn} (l.31-32) est utilisé ensuite comme covariance de Kalman (centrée) ; (iii) eq. (13) de §2 (décomposition de variance totale) exige la lecture centrée ; (iv) à l'inverse, l.336-337, les auteurs centrent MANUELLEMENT (Σ_{z_0}(k) = \Vark{Z_1 − μ_{z_0}(k)}{r_1=k}), preuve que la macro est bien non centrée. L'usage est donc incohérent d'une équation à l'autre, et (8) n'est correcte que parce que W est centré.

**Preuve :** macros.tex l.22-26 : « %% Cross-covariance (conditional): \Covk{A}{B}{c} → E[ A B^T | c ] \newcommand{\Covk}[3]{\Ek{#1\,#2\tp}{#3}} » ; 03_filtering.tex l.103-104 : « \Covk{Z_{n+1}}{Z_n}{r_n = j, r_{n+1} = k, \yn} = F_k(P_n(j) − μ_n(j)μ_n(j)ᵀ) » ; l.337 : « \Sigma_{z_0}(k) = \Vark{Z_1 − \mu_{z_0}(k)}{r_1 = k} ».

**Suggestion :** Redéfinir \Covk/\Vark comme covariances centrées (Cov, Var au sens usuel) et introduire une macro distincte pour les seconds moments non centrés (utilisée dans (14) et (18)) ; supprimer alors le centrage manuel de l.337 et vérifier chaque occurrence.

**Ajustement de sévérité (vérificateurs) :** keep medium — pervasive notation bug making printed equations (esp. eq. 17) literally false, but no algorithmic or code impact; fix is purely editorial | keep medium — notational/rendering correctness issue making several printed equations literally false, but intent is recoverable, derivations are correct under the centered reading, and code/results are unaffected; not low because the paper demonstrably uses the same macro in two contradictory senses (l.103-105 centered vs l.337 raw), which a reviewer or careful reader cannot resolve from the text alone

### ✅ [LOW] Cas dégénérés passés sous silence : inversibilité de S^{(k)}, Σ_{YY,0}(k), Σ_{XX,n+1}(k) (Joseph) et Σ_{V,j} jamais hypothéquée

`paper/sections/03_filtering.tex:236-241 (eqs 28-29), 256-283 (Remark 3, eq. 32), 345-360 (eqs 35-37) ; 02_model_h5.tex eq. (11)` — statut : confirmed (1 vote(s)) — catégorie : derivation-gap

La section inverse plusieurs matrices sans condition : (i) S^{(k)} = Σ_{YY,n+1}(k) dans le gain (29) — peut être singulière si Σ_{V,k} est singulière ou si s-composantes dégénèrent ; (ii) Σ_{YY,0}(k) à l'init (35)-(37) ; (iii) la forme de Joseph (Remark 3, l.273) requiert Σ_{XX,n+1}(k)^{-1} — singulière dès qu'une composante d'état est déterministe (j'ai vérifié l'algèbre de (32) : elle est correcte et redonne (31) pour K = Σ_XY S^{-1}, mais uniquement si H est défini, donc Σ_XX ≻ 0, et R^{(k)} ⪰ 0 n'est garanti que si la jointe est PSD) ; (iv) Γ_{jk} (Prop. 1) et la contrainte AB requièrent Σ_{V,j} ≻ 0, énoncé en §4/App. C mais jamais rappelé pour le filtre de §3 ; (v) π_0 := « the stationary distribution of the transition kernel » (l.334-335) suppose existence/unicité (irréductibilité + apériodicité), non mentionnées ; (vi) le dénominateur de (24) peut s'annuler numériquement (le code se protège par logsumexp/fallbacks, le papier est muet). Aucun problème dimensionnel q ≠ s en revanche : toutes les équations de §3 sont dimensionnellement correctes pour q ≠ s (vérifié bloc par bloc).

**Preuve :** l.239 : K^{(k)} := Σ_{XY,n+1}(k)[S^{(k)}]^{-1} ; l.273 : H^{(k)} := Σ_{YX,n+1}(k)Σ_{XX,n+1}(k)^{-1} ; l.351 : K_1^{(k)} := Σ_{XY,0}(k)Σ_{YY,0}(k)^{-1} ; aucune hypothèse de définie-positivité dans toute la section 3.

**Suggestion :** Ajouter une hypothèse de régularité en tête de §3 (Σ_{V,k} ≻ 0 ∀k, chaîne R irréductible apériodique) et une phrase pour les cas singuliers (pseudo-inverse / forme de Joseph réservée à Σ_XX ≻ 0), à l'image des garde-fous déjà présents dans le code (_safe_solve, _psd_floor, allow_singular).

**Ajustement de sévérité (vérificateurs) :** Keep [low] — correctly calibrated. Real but standard exposition gap (implicit regularity assumptions common in applied filtering papers), fixable with one sentence at the head of §3; slightly aggravated by the asymmetry that §4/App. C do state ≻0 hypotheses while §3 and Prop. 1 never do, and by the code needing explicit guards (_safe_solve, _psd_floor, allow_singular) the paper never mentions.

### ✅ [LOW] Complexité jamais quantifiée : aucun O(K²) dans le papier ; seule affirmation « operations independent of n » (correcte)

`paper/sections/03_filtering.tex:section entière (cf. aussi 02_model_h5.tex l.159-161)` — statut : confirmed (1 vote(s)) — catégorie : claim-overreach

Contrairement à ce qu'on attend d'un papier vendant un filtre « fast/exact », aucune complexité par pas n'est énoncée (grep « O(K », « complexity », « quadratic » : zéro occurrence dans paper/). La seule affirmation est en §2 (l.159-161) : « in a number of operations independent of n » — exacte et justifiée (étape II : K² vraisemblances de paires avec résolutions s×s, soit O(K²s³) ; étapes I/III : O(K(q+s)³) ; total O(K²s³ + K(q+s)³) par pas, O(N·…) au total). L'exactitude/optimalité revendiquée (l.14-15 « this IMM is exact », l.44-46 « no Gaussian-collapse approximation is made at any stage », titre « Optimal filtering recursion ») est justifiée pour l'architecture à bouclage postérieur SOUS RÉSERVE de la correction de (22) : telle qu'imprimée, (22) rend la récursion non exacte, donc la revendication centrale de la section est actuellement contredite par sa propre équation (cf. finding critique).

**Preuve :** Recherche exhaustive : aucun « O(K » ni « complexity » dans paper/sections/ ni paper/appendix/ ; 02_model_h5.tex l.159-161 : « in a number of operations independent of n ».

**Suggestion :** Ajouter une phrase de complexité explicite en fin de §3 : « chaque pas coûte O(K² s³ + K (q+s)³) opérations — K² vraisemblances de paires gaussiennes et K mises à jour de Kalman — indépendamment de n », ce qui renforce le positionnement IMM.

**Ajustement de sévérité (vérificateurs) :** Keep at low. It is an editorial completeness gap (missing asymptotic complexity statement in a paper whose selling point includes speed), not an error. One nuance for the report: the paper does quantify speed empirically (309 µs/step, 06_experiments.tex l.166), so phrase the finding as "no asymptotic/big-O complexity stated" rather than "complexity never quantified". The exactness-contradiction remark belongs to the separate critical finding on eq:S_jk and should not inflate this one's severity.

### ✅ [INFO] Numérotation (16)-(17)/(19)-(20)/(21)-(22) du script obsolète vs papier actuel ; l'équivalence vérifiée est vraie mais absente du papier

`scripts/verify_19_20_equivalence.py:1-63 (docstring) ; paper.aux eqs (16)-(23)` — statut : confirmed (1 vote(s)) — catégorie : notation

Le script vérifie l'équivalence entre (a) itération des moments stationnaires + conditionnement gaussien et (b) formes fermées E[X_{n+1}|r_{n+1},y_{n+1}] = ΔΣ_V^{-1}y_{n+1}, Var = Σ_U − ΔΣ_V^{-1}Δᵀ sous AB — je l'ai relancé (PASS, écarts ≤ 9.2e-16) et re-démontrée analytiquement (argument W' = U − ΔΣ_V^{-1}V, Cov(W',V|r) = 0). Mais ses références « (19)-(20) », « (16)-(17) de Wojciech », « (21)-(22) » correspondent à une ANCIENNE numérotation (companion paper / draft CS_FinaleBis) : dans le papier actuel (paper.aux), (19)-(20) = eq:joint_factor/eq:y_transition (factorisation et vraisemblance de paire), (21)-(22) = eq:y_pred_jk/eq:S_jk, et la récursion des moments est (16)-(18), pas (16)-(17). Quiconque lit le script avec le papier actuel sous les yeux est induit en erreur. L'équivalence elle-même — l'un des résultats les plus utiles du cadre (gain de Kalman = ΔΣ_V^{-1}, calculable sans aucune récursion) — n'est ni énoncée ni démontrée dans le papier (seule la Remarque 2 y fait une allusion erronée, cf. finding dédié) ; le script note d'ailleurs lui-même le terme correctif b_X − ΔΣ_V^{-1}b_Y pour b ≠ 0, absent du papier.

**Preuve :** Docstring l.6-9 : « (19) E[X_{n+1} | r_{n+1}, y_{n+1}] ; (20) E[X_{n+1} X_{n+1}^T | r_{n+1}, y_{n+1}] » vs paper.aux : eq (19) = eq:joint_factor, eq (20) = eq:y_transition ; l.27 : « la récursion (16)-(17) de Wojciech » vs papier : moments = (16)-(18). Exécution : « OK : methods (a) and (b) agree to within 1e-09 for every trial » (max 9.2e-16).

**Suggestion :** Mettre à jour la docstring du script avec les labels LaTeX (eq:mu_propagate/eq:var_Z, eq:kalman_mean/eq:kalman_var…) plutôt que des numéros, et ajouter dans le papier un lemme « forme fermée du gain sous AB » (avec le terme de biais b_X − ΔΣ_V^{-1}b_Y) que ce script certifierait.

## Section 4 + annexes B/C — Contrainte AB

_Audit de la Section 4 (contrainte AB) et des annexes B et C, par redérivation manuelle complète (dimensions, transposes, conditions d'existence) et contre-vérification numérique (numpy) des points litigieux, plus comparaison ligne à ligne avec prg/utils/h5_constraint.py. Le cœur positif est solide : la forme close AB est correcte, sa preuve de suffisance (Prop. 4.1 / annexe C §3) est juste, et le code implémente exactement les formules du papier (compute_AB ↔ eq:AB, compute_h5_residual ↔ eq:H5_compact ; résidus ~1e-17 sur toutes les paires de régimes). En revanche, la chaîne de NÉCESSITÉ est défaillante à trois endroits : le passage eq:H5_compact → eq:H5_in_X de l'annexe C est invalide (identité s×q promue en (q+s)×q ; l'ensemble des solutions de eq:H5_full est en fait un espace affine de dimension q², vérifié numériquement), l'élimination « pour tout Σ(r) » est incohérente (Z supposé indépendant de Σ alors que Z = Σ_V⁻¹Δᵀ, blocs Δ/Σ_V à la fois variés et fixés) et contredit la docstring du code qui admet la non-unicité quand K·s < q+s. L'annexe B contient en outre un Step 4 imprimé faux (trois identités erronées, une équation dimensionnellement impossible — la conclusion reste juste par simple transposition), un décret « noise law » non justifié par l'annexe A citée, et une affirmation « sufficient and necessary » du cas j=k réfutée numériquement (une solution non-AB du cas j=k viole les paires j≠k). S'y ajoutent des conditions d'existence erronées ou manquantes (M ≻ 0 faux sous les seules hypothèses énoncées, M singulière possible sous la Prop. 4.1, Σ_V singulière jamais traitée) et l'usage non défini du mot « projection » (aucune métrique, annexe « C_projections » sans projections). Le décompte dof (eq:bic_d) est arithmétiquement correct et cohérent avec les cinq blocs libres, à l'omission près des moments initiaux estimés._

### ✅ [CRITICAL] eq:H5_in_X n'est PAS équivalente à eq:H5_compact — le passage clé de la preuve de nécessité est invalide

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/appendix/C_projections.tex:31-39, eq:H5_in_X` — statut : confirmed (2 vote(s)) — catégorie : math-error

L'annexe C affirme que « Multiplying the compact form (eq:H5_compact) on the left by M(r) and re-arranging yields the equivalent identity » eq:H5_in_X : X = Hᵀ[HΣHᵀ+Σ_V]⁻¹[HΣX+Δᵀ]. C'est impossible : eq:H5_compact est une identité s×q (elle s'écrit [Δᵀ, Σ_V]·(X − HᵀM⁻¹(HΣX+Δᵀ)) = 0), alors que eq:H5_in_X est une identité (q+s)×q. Passer de l'une à l'autre revient à supprimer le facteur gauche [Δᵀ, Σ_V] = [0,I]Σ, qui a un noyau de dimension q : eq:H5_in_X est strictement plus forte. Multiplier à gauche par M (s×s) ne change ni la taille ni ne fait apparaître l'identité annoncée. Vérification numérique (q=2, s=1) : l'ensemble des (A,B) solutions de eq:H5_full pour des blocs (C,D,Σ_U,Δ,Σ_V) fixés est un espace affine de dimension q²=4 (rang du système linéaire = 1) ; le point AB en est UN élément, pas le seul. Toute la « nécessité » de la contrainte AB repose sur ce pas.

**Preuve :** C_projections.tex l.31-33 : « Multiplying the compact form~\eqref{eq:H5_compact} on the left by M(r) and re-arranging yields the equivalent identity » ; eq:H5_in_X l.34-39. Numérique : rank([Δᵀ−PM⁻¹Q, Σ_V−PM⁻¹R]) = 1 sur (q+s)=3 colonnes ⇒ solutions affines de dim 4 ; un point non-AB (‖A−A_AB‖=2.74) donne un résidu eq:H5_full ≈ 9e-16.

**Suggestion :** Reformuler honnêtement : eq:H5_compact n'implique eq:H5_in_X que sous le quantificateur « pour toute statistique du second ordre de Z_n » (et avec des conditions de généricité : H ≠ 0, rang(H) = s). Sinon, présenter AB comme une solution canonique suffisante (Prop. 4.1) et renvoyer la discussion de nécessité à un énoncé précis avec hypothèses.

**Ajustement de sévérité (vérificateurs) :** critical → high

### ✅ [HIGH] Élimination de Σ(r) incohérente : Z(r) « indépendant de Σ(r) » alors que la solution finale Z = Σ_V⁻¹Δᵀ dépend des blocs de Σ(r)

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/appendix/C_projections.tex:43-67 (eq:split_a, eq:split_b)` — statut : confirmed (2 vote(s)) — catégorie : derivation-gap

L'argument postule (l.47-53) un Z(r) indépendant de Σ(r) tel que HΣX + Δᵀ = (HΣHᵀ + Σ_V)Z, puis « splitte » par dépendance en Σ : HΣX = HΣHᵀZ (eq:split_a) et Σ_V Z = Δᵀ (eq:split_b). Or Δ et Σ_V SONT des blocs de Σ(r) (définie l.22-27) : les traiter comme constants pendant que Σ varie sur tout le cône est contradictoire, et la solution obtenue Z = Σ_V⁻¹Δᵀ dépend de Σ(r), contredisant l'hypothèse de départ. De même l.46-47 « X(r) is a model parameter ... cannot depend on Σ(r) » est réfuté par le résultat final A = ΔΣ_V⁻¹C qui dépend de Δ, Σ_V. L'argument ne devient cohérent que si seul Σ_U varie (Δ, Σ_V fixés) — auquel cas l'étape « eq:split_a pour tout Σ ⇒ X = HᵀZ » doit être refaite et requiert C ≠ 0 et R = CΔ+DΣ_V inversible, conditions jamais énoncées. L'existence même de la décomposition en Z constant (ansatz) requiert Hᵀ injectif (rang(H) = s), non énoncé non plus.

**Preuve :** C_projections.tex l.47-49 : « there must exist Z(r) ∈ R^{s×q} — independent of Σ(r) — such that ... » ; l.62 : « Z(r) = Σ_V(r)⁻¹Δ(r)ᵀ » (dépend de Σ(r)) ; l.60-61 : « required for all Σ(r) ≻ 0, which forces X = HᵀZ » sans condition sur H.

**Suggestion :** Préciser le quantificateur (Σ_U libre, (Δ, Σ_V) fixés — cohérent avec « independently of Σ_U » de la Sec. 4 l.79), ajouter les hypothèses de généricité (C ≠ 0, rang(H) = s, R inversible), et démontrer le pas pour-tout-Σ_U proprement (différences de Σ_U engendrent les symétriques ⇒ CWV₁ = 0 ∀W ⇒ V₁ = 0 si C ≠ 0, puis RV₂ = 0).

### ✅ [HIGH] Nécessité survendue : « the unique closed-form solution » contredit la propre docstring du code (non-unicité si K·s < q+s)

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/04_constraint.tex:62-79 et 90-92 (eq:AB)` — statut : confirmed (2 vote(s)) — catégorie : claim-overreach

La Sec. 4 affirme (l.67-69) que l'élimination donne « the unique closed-form solution », puis (l.77-79) que « A and B are both uniquely determined by the remaining five blocks », et (l.90-92) que « the (H5)-compatible family of GSS models is parametrised by the five free blocks » — i.e. AB ⇔ (H5). Or (i) le quantificateur qui rend l'unicité vraie (« for (H5) to hold across the family of admissible Σ(r) ≻ 0 », l.62-63) n'est pas justifié : (H5) eq:H5 est une propriété d'UN modèle donné, dont la statistique du second ordre est fixée, pas un « pour tout Σ » ; rien dans le papier n'explique pourquoi (H5) devrait tenir pour des statistiques arbitraires. (ii) Le code du dépôt dit explicitement le contraire du papier : prg/utils/h5_constraint.py l.21-30 — « Necessity is more subtle ... generically when K·s ≥ q+s. In the sub-determined regime K·s < q+s, (H5)-compatible models exist that are not of the AB form: AB is one specific point in a (q+s−Ks)·q-dimensional affine space of solutions per regime. » Le comptage K·s ≥ q+s n'apparaît nulle part dans le papier. Vérifié numériquement : pour un régime fixé, les solutions de eq:H5_full forment un espace affine de dimension q².

**Preuve :** 04_constraint.tex l.67-69 : « A short elimination argument (Appendix C) shows that the unique closed-form solution is » ; l.90-92 : « the (H5)-compatible family of GSS models is parametrised by the five free blocks ». vs prg/utils/h5_constraint.py l.26-28 : « (H5)-compatible models exist that are not of the AB form ».

**Suggestion :** Aligner papier et code : énoncer AB comme suffisante toujours (Prop. 4.1), et nécessaire (a) sous le quantificateur explicite « pour toute initialisation / toute statistique second ordre admissible », ou (b) génériquement quand K·s ≥ q+s pour un modèle fixé, en important le comptage de la docstring. Sinon affaiblir « unique » en « canonique ».

**Ajustement de sévérité (vérificateurs) :** Maintenir high (défendable) : l'énoncé encadré et la phrase « parametrised by the five free blocks » de la Sec. 4 sont faux à la lettre sans hypothèse de généricité/comptage K·s ≥ q+s. Un déclassement vers medium serait acceptable au vu du hedging correct déjà présent dans l'abstract et la conclusion (« generically necessary »), qui montre que seule la formulation de la Sec. 4 est en cause, pas la conscience des auteurs du problème. | high → low (la prétendue contradiction papier/code n'existe pas ; reste une phrase non hedgée en Sec. 4 l.89-92 et un renvoi vers l'Appendix C qui ne contient pas l'argument K·s ≥ q+s promis)

### ✅ [HIGH] Step 4 contient trois identités fausses (dont une dimensionnellement impossible) — la conclusion eq:h5_compact_app reste juste

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/appendix/B_h5_derivation.tex:138-159 (Step 4)` — statut : confirmed (2 vote(s)) — catégorie : math-error

(i) l.145-146 : « Q_A Cᵀ + Q_B Dᵀ + Σ_V is exactly [F·P(r)·Fᵀ + Σ_W]_YY = M » est faux : Q_ACᵀ + Q_BDᵀ est q×s (on ne peut même pas lui ajouter Σ_V s×s si q≠s) ; en réalité Q_ACᵀ + Q_BDᵀ + Δ = [FPFᵀ+Σ_W]_XY = T, et c'est QCᵀ + RDᵀ + Σ_V qui vaut M. (ii) L'équation non numérotée l.147-150, T M⁻¹R = QAᵀM⁻¹R + RBᵀM⁻¹R + ΔM⁻¹R, est dimensionnellement incohérente pour q≠s (QAᵀ est s×q, M⁻¹ est s×s) et fausse pour q=s (elle vaudrait TᵀM⁻¹R + (Δ−Δᵀ)M⁻¹R ≠ TM⁻¹R). (iii) l.151-152 : « Q_Aᵀ = CΣ_U + DΔᵀ = Q (Q_A = Qᵀ) » et « Q_Bᵀ = CΔ + DΣ_V = R » sont faux : Q_Aᵀ = Σ_UAᵀ + ΔBᵀ et Q_Bᵀ = ΔᵀAᵀ + Σ_VBᵀ (mauvaises dimensions et mauvais blocs : A,B vs C,D). Le résultat final eq:h5_compact_app (l.154-159) est néanmoins correct : il suffit de transposer eq:h5_first (ΔᵀAᵀ + Σ_VBᵀ = RᵀM⁻¹Tᵀ = PM⁻¹Tᵀ) et d'utiliser Tᵀ = (Q_ACᵀ + Q_BDᵀ + Δ)ᵀ = C(Σ_UAᵀ+ΔBᵀ) + D(ΔᵀAᵀ+Σ_VBᵀ) + Δᵀ = QAᵀ + RBᵀ + Δᵀ — j'ai revérifié cette chaîne à la main.

**Preuve :** B_h5_derivation.tex l.145-146 : « Note that Q_A Cᵀ + Q_B Dᵀ + Σ_V is exactly [F·P(r)·Fᵀ + Σ_W]_YY = M » ; l.151 : « where we used Q_Aᵀ = CΣ_U + DΔᵀ = Q ». Contre-vérification : Q_A = AΣ_U + BΔᵀ (q×q) ⇒ Q_Aᵀ = Σ_UAᵀ + ΔBᵀ ≠ Q = CΣ_U + DΔᵀ (s×q).

**Suggestion :** Remplacer tout le Step 4 par la transposition directe : eq:h5_first ⇒ ΔᵀAᵀ + Σ_VBᵀ = PM⁻¹Tᵀ (R = Pᵀ, M = Mᵀ), puis Tᵀ = QAᵀ + RBᵀ + Δᵀ (deux lignes de calcul). Supprimer les identités fausses Q_Aᵀ = Q, Q_Bᵀ = R et l'équation non numérotée.

### ✅ [HIGH] Le décret « noise law » (Var(Z_n|r_n=j) = P(j)) n'est pas justifié par l'annexe A citée, et la Sec. 4 confond covariance de bruit et covariance marginale par régime

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/appendix/B_h5_derivation.tex:22-27 (Step 1) ; aussi 04_constraint.tex l.53-64 et 46-47 (Step 2)` — statut : confirmed (2 vote(s)) — catégorie : derivation-gap

Step 1 (l.22-27) affirme : « The Markovianity argument of Appendix A ... shows that all quantities relevant to (H5) may be evaluated with the noise law, i.e. with Z_n | r_n = j having mean zero and covariance P(j) ». Or l'annexe A prouve uniquement le collapse p(x_{n+1}|w_{1:n+1}) = p(x_{n+1}|w_{n+1}) — rien qui identifie Var(Z_n|r_n=j) à la covariance de bruit P(j) = Var(W_n|r_n=j). En général Var(Z_n|r_n=j) est la covariance marginale (mélange sur les histoires de régimes / point fixe stationnaire) ≠ P(j). De plus Step 2 (l.46-47) asserte que la loi de (X_{n+1},Y_n,Y_{n+1})|r_n=j,r_{n+1}=k « is Gaussian », alors que conditionnellement à (r_n,r_{n+1}) seuls c'est un mélange gaussien (la gaussianité par régime est une conséquence de (H5), Remark rem:marginal_form — utilisable pour la nécessité mais à citer). La Sec. 4 (l.55-64) entérine la confusion en appelant les blocs de bruit (Σ_U, Δ, Σ_V) de eq:noise_cov « the joint per-regime covariance ... second-order statistics of (X_n, Y_n)|r_n = r », en contradiction directe avec eq:noise_cov (02_model_h5.tex l.70-77) où ces mêmes blocs sont Var(W_{n+1}|r_{n+1}). Dans M, le Σ_V additif est un bloc de bruit du régime k tandis que le Σ_V de DΣ_VDᵀ joue le rôle de Var(Y_n|r_n=j) : deux objets distincts notés pareil.

**Preuve :** B_h5_derivation.tex l.24-26 : « all quantities relevant to (H5) may be evaluated with the noise law » ; A_time_reversal.tex l.7-8 ne prouve que p(x_{n+1}|w_{1:n+1}) = p(x_{n+1}|w_{n+1}) ; 04_constraint.tex l.56-64 : « the joint per-regime covariance Σ(r) ... arbitrary second-order statistics of (X_n, Y_n) | r_n = r » avec les blocs de eq:noise_cov.

**Suggestion :** Soit (a) justifier précisément : (H5) doit tenir pour toute initialisation admissible (Σ_z0(j) libre dans le modèle), donc la covariance de Z_n|r_n=j parcourt une famille riche incluant P(j) — auquel cas le « pour tout Σ » devient légitime et le décret noise-law inutile ; soit (b) mener la dérivation avec la covariance réelle Σ̃(j) et noter que la condition β₁=0 ne dépend de Σ̃(j) que linéairement, l'AB la satisfaisant pour TOUTE matrice PSD. Distinguer notationnellement le Σ_V de bruit du Var(Y_n|r_n).

### ✅ [HIGH] « Sufficient and necessary to enforce j=k » est faux tel quel ; et nulle part le papier ne prouve AB ⇒ (H5) pour toutes les paires (j,k)

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/appendix/B_h5_derivation.tex:100-104 (Step 3) ; 04_constraint.tex l.81-89 (Prop. 4.1)` — statut : confirmed (2 vote(s)) — catégorie : derivation-gap

Step 3 affirme : « For this to hold for all source regimes j, it is sufficient and necessary to enforce eq:h5_beta1 in the special case j=k ». La direction « suffisant » est fausse pour une solution générale de l'équation j=k : vérifié numériquement, un (A,B) non-AB satisfaisant exactement la contrainte j=k (résidu 9e-16) viole l'équation de paire pour j≠k (résidu 1.68). Elle n'est vraie que pour le point AB spécifiquement : avec [A_k,B_k] = Δ_kΣ_{V,k}⁻¹[C_k,D_k], la factorisation (G_kΣHᵀ_k + Δ_k) = Δ_kΣ_{V,k}⁻¹(H_kΣHᵀ_k + Σ_{V,k}) annule β₁ pour TOUTE covariance source Σ ⪰ 0 — preuve d'une ligne absente du papier. La « nécessité » du cas j=k suppose en outre P_{jj} > 0 (auto-transitions possibles), jamais énoncé. Conséquence : la Prop. 4.1 ne vérifie que l'identité algébrique j=k (eq:H5_full) ; le papier ne démontre jamais que AB implique (H5) au sens plein (toutes paires, indépendance conditionnelle complète, pas seulement β₁=0). C'est pourtant vrai et structurel : sous AB, X_{n+1} = Δ_{r_{n+1}}Σ_{V,r_{n+1}}⁻¹Y_{n+1} + (U_{n+1} − Δ_{r_{n+1}}Σ_{V,r_{n+1}}⁻¹V_{n+1}), où le second terme est indépendant de tout le passé et de Y_{n+1} — (H5) en découle immédiatement. La docstring du code (« satisfies the K² regime-pair equations ... by construction », h5_constraint.py l.17-19) énonce ce qui manque au papier.

**Preuve :** B_h5_derivation.tex l.100-103 : « it is sufficient and necessary to enforce~\eqref{eq:h5_beta1} in the special case j = k » suivi seulement des calculs j=k ; numérique : résidu paire j≠k = 1.68 pour une solution non-AB de la contrainte j=k, = 1e-17 pour AB.

**Suggestion :** Remplacer la phrase par : la contrainte j=k est nécessaire (si P_{rr} > 0) ; la solution AB qui en découle satisfait ensuite toutes les paires (j,k) — ajouter la preuve d'une ligne via la factorisation Δ_kΣ_{V,k}⁻¹, ou l'argument structurel X_{n+1} = Δ_kΣ_{V,k}⁻¹Y_{n+1} + bruit ⊥ passé, qui établit (H5) en toute rigueur (et renforce la Prop. 4.1).

### ✅ [MEDIUM] « M ≻ 0 whenever Σ_U, Σ_V ≻ 0 » est faux ; l'hypothèse de la Prop. 4.1 (« for every choice ... with Σ_V ≻ 0 ») n'exclut pas M singulière

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/04_constraint.tex:38-40 et 81-86 (Prop. 4.1)` — statut : confirmed (2 vote(s)) — catégorie : math-error

M = HΣ(r)Hᵀ + Σ_V n'est PSD que si la covariance jointe Σ(r) = [[Σ_U,Δ],[Δᵀ,Σ_V]] est ⪰ 0 ; Σ_U, Σ_V ≻ 0 seuls ne suffisent pas. Contre-exemple (q=s=1) : Σ_U=Σ_V=1, Δ=2, C=1, D=−1 ⇒ M = −1. De même, la Prop. 4.1 quantifie sur « every choice of (C,D,Σ_U,Δ,Σ_V) with Σ_V ≻ 0 », mais avec C=D=Σ_U=Σ_V=1, Δ=−1.5 on obtient M = 0 : eq:H5_full (qui contient M⁻¹) est alors indéfinie, donc l'énoncé est faux à la lettre. La preuve de l'annexe C (l.69-92, vérifiée correcte par ailleurs : LHS = PΣ_V⁻¹Δᵀ, crochet = MΣ_V⁻¹Δᵀ, RHS = PM⁻¹MΣ_V⁻¹Δᵀ) n'utilise que l'inversibilité de M, qu'il faut donc ajouter en hypothèse — automatique si Σ(r) ⪰ 0 et Σ_V ≻ 0.

**Preuve :** 04_constraint.tex l.38-40 : « M = Mᵀ ≻ 0 whenever Σ_U, Σ_V ≻ 0 » ; l.82-84 : « For every choice of (C, D, Σ_U, Δ, Σ_V) with Σ_V ≻ 0 ». Numérique : M = [[-1.]] (Δ=2, D=−1) ; M = [[0.]] (Δ=−1.5).

**Suggestion :** Corriger en « M = Mᵀ ≻ 0 dès que Σ(r) ⪰ 0 et Σ_V ≻ 0 » et ajouter cette hypothèse (naturelle : Σ(r) est une covariance) à la Prop. 4.1.

**Ajustement de sévérité (vérificateurs) :** medium → low (hypothesis-precision flaw; counterexamples require Σ(r) non-PSD, impossible for the noise covariance defined in eq:noise_cov, and Σ(r) ≻ 0 is displayed twice in the same section/appendix; fix is the one-line hypothesis addition already proposed in the finding)

### ✅ [MEDIUM] « AB projection » : aucune métrique, aucune optimalité, idempotence non énoncée — et l'annexe nommée C_projections ne contient aucune analyse de projection

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/appendix/C_projections.tex:fichier entier ; 04_constraint.tex l.143 ; 05_estimation.tex l.26-30 ; 06_experiments.tex l.324` — statut : confirmed (2 vote(s)) — catégorie : claim-overreach

Le papier appelle systématiquement « projection » l'écrasement (A,B) ← (ΔΣ_V⁻¹C, ΔΣ_V⁻¹D) (Sec. 4 l.143 « post-hoc projection », Alg. 1 l.57-60, et 06 l.324 « projection onto the H5 constraint manifold »). Or : (i) aucune métrique n'est définie ; (ii) ce n'est PAS la projection au plus proche point sur l'ensemble des solutions de eq:H5_full — pour (C,D,Σ_W) fixés cet ensemble est un espace affine de dimension q² en (A,B) (vérifié numériquement), et le point AB n'est généralement pas le plus proche d'un estimateur OLS au sens de Frobenius ; (iii) la seule propriété vraie, l'idempotence (la carte ne dépend que de (C,D,Δ,Σ_V), inchangés), n'est ni énoncée ni démontrée ; (iv) le fichier d'annexe s'appelle C_projections.tex mais son contenu est « Derivation and proof of the AB constraint » — aucune propriété de projection n'y figure, suggérant un contenu prévu puis abandonné.

**Preuve :** 06_experiments.tex l.324 : « projection onto the H5 constraint manifold » ; C_projections.tex titre l.4 : « Derivation and proof of the AB constraint » (aucune métrique/optimalité dans les 92 lignes) ; dimension q² de l'ensemble des solutions vérifiée numériquement.

**Suggestion :** Soit renommer en « AB substitution/retraction » (terminologie neutre), soit définir la projection (p.ex. au sens KL ou Frobenius sous (C,D,Σ_W) fixés) et démontrer le statut du point AB ; au minimum énoncer l'idempotence. Renommer le fichier ou son titre pour cohérence.

**Ajustement de sévérité (vérificateurs) :** medium → low

### ✅ [LOW] Cas C=0 : « B = 0 (which requires D = 0 or Δ = 0) » est faux dès que s ≥ 2

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/04_constraint.tex:121-125 (table tab:h5_special_cases, ligne C = 0)` — statut : confirmed (1 vote(s)) — catégorie : math-error

B = ΔΣ_V⁻¹D = 0 n'exige pas D = 0 ou Δ = 0 : il suffit que les images se croisent orthogonalement. Contre-exemple (q=1, s=2) : Δ = [1, 0], Σ_V = I₂, D = [[0,0],[1,0]] ⇒ ΔΣ_V⁻¹D = [0, 0] avec Δ ≠ 0 et D ≠ 0 (et Σ_U = 2 rend la covariance jointe PSD). L'affirmation n'est vraie que pour s = 1.

**Preuve :** 04_constraint.tex l.124-125 : « when further B = 0 (which requires D = 0 or Δ = 0) ». Numérique : Δ Σ_V⁻¹ D = [[0., 0.]] avec D ≠ 0, Δ ≠ 0.

**Suggestion :** Écrire « e.g. D = 0 or Δ = 0 » ou « which for s = 1 requires D = 0 or Δ = 0 », ou la condition exacte ΔΣ_V⁻¹D = 0.

### ❌ [LOW] Cas Σ_V singulière jamais discuté (existence/unicité de la forme close)

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/04_constraint.tex:69-76 (eq:AB) ; C_projections.tex l.58 (eq:split_b)` — statut : refuted (1 vote(s)) — catégorie : derivation-gap

La forme close eq:AB requiert Σ_V⁻¹ et l'élimination requiert Σ_V Z = Δᵀ (eq:split_b). Le papier ne dit nulle part ce qui se passe si Σ_V est singulière : si la covariance jointe Σ(r) ⪰ 0, alors range(Δᵀ) ⊆ range(Σ_V) et Z = Σ_V⁺Δᵀ existe, mais Z (donc (A,B)) n'est plus unique (composantes libres dans ker Σ_V). La boîte eq:AB est énoncée sans hypothèse (la condition Σ_V ≻ 0 n'apparaît que dans la Prop. 4.1). Le code, lui, gère le cas : compute_AB lève ValueError si cond(Σ_V) > 1e12 (prg/utils/h5_constraint.py l.138-147) et compute_h5_residual propage LinAlgError si M singulière — comportements non reflétés dans le texte.

**Preuve :** 04_constraint.tex l.69-76 : boîte eq:AB sans condition d'existence ; prg/utils/h5_constraint.py l.138-141 : garde cond > 1e12.

**Suggestion :** Ajouter à côté de eq:AB : « valid for Σ_V ≻ 0 ; for singular Σ_V with Σ(r) ⪰ 0, Σ_V⁻¹ may be replaced by the pseudo-inverse but uniqueness is lost », et mentionner la garde numérique.

**Ajustement de sévérité (vérificateurs) :** n/a (rejetée)

### ✅ [LOW] Décompte dof eq:bic_d : arithmétique correcte et cohérente avec les « five free blocks », mais omet les paramètres initiaux pourtant estimés par l'Algorithme 1

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/06_experiments.tex:365-379 (eq:bic_d)` — statut : confirmed (1 vote(s)) — catégorie : claim-overreach

d_H5 = K[qs + s² + (q+s)(q+s+1)/2 + (q+s)] + K² − 1 : vérifié — C_k (qs), D_k (s²), Σ_W,k symétrique ((q+s)(q+s+1)/2 = q(q+1)/2 + s(s+1)/2 + qs, cohérent avec les cinq blocs libres de Sec. 4 l.90-92), b_k (q+s), P stochastique (K(K−1)) + π₀ (K−1) = K²−1 ; A_k, B_k exclus à juste titre. En revanche les moments initiaux (μ_z0,k, Σ_z0,k) sont estimés par l'Algorithme 1 (05_estimation.tex l.62-63) et entrent dans la vraisemblance, mais ne sont pas comptés — choix défendable asymptotiquement (termes O(1) en N) mais non signalé.

**Preuve :** 06_experiments.tex l.367-372 (eq:bic_d) vs 05_estimation.tex l.62-63 : « (μ̂_{z_0,k}, Σ̂_{z_0,k}) ← sample mean/covariance ».

**Suggestion :** Une phrase : « initial-state moments are excluded from d_H5 as their contribution to the log-likelihood is O(1) in N ».

### ✅ [LOW] Petites incohérences de notation : intercept b_{X,k} vs « α » non défini ; \Covk/\Vark sont des moments non centrés ; renvoi H5bis vs H5

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/appendix/B_h5_derivation.tex:53, 168 ; macros.tex l.23-26 ; 04_constraint.tex l.20-22` — statut : confirmed (1 vote(s)) — catégorie : notation

(i) eq:mean_X_linear (B l.53) nomme l'intercept b_{X,k}, mais la Remark (l.168) parle de « the intercept α », symbole jamais défini ; en outre, sous la convention « noise law, mean zero » décrétée au Step 1, l'intercept devrait être 0, et dans le cas biaisé il vaut b_{X,k} − β₁E[Y_n] − β₂E[Y_{n+1}] + ..., pas b_{X,k}. (ii) macros.tex définit \Covk{X}{Y}{·} = E[XYᵀ|·] et \Vark{X}{·} = E[XXᵀ|·] — des moments NON centrés, utilisés partout comme covariances : correct uniquement sous la convention zéro-moyenne, jamais signalé. (iii) 04 l.20-22 annonce une dérivation « from the conditional independence relation eq:H5bis » alors que l'annexe B part de eq:H5 (Step 3, l.89).

**Preuve :** B_h5_derivation.tex l.53 : « b_{X,k} + β₁ y_n + β₂ y_{n+1} » ; l.168 : « it enters the mean ... through the intercept α » ; macros.tex l.23 : \newcommand{\Covk}[3]{\Ek{#1\,#2\tp}{#3}}.

**Suggestion :** Unifier le symbole d'intercept (α partout, défini), noter une fois la convention zéro-moyenne pour \Covk/\Vark, corriger le renvoi H5bis → H5 (ou inversement).

### ✅ [INFO] Cohérence papier ↔ code : formules strictement identiques ; l'annexe B utilise par ailleurs la variance pair-conditionnelle correcte

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/prg/utils/h5_constraint.py:95-102 et 144-150 (vs eq:AB, eq:H5_compact)` — statut : confirmed (1 vote(s)) — catégorie : claim-overreach

Vérification positive : compute_AB implémente exactement eq:AB (A = ΔΣ_V⁻¹C via solve(SV,C), B = ΔΣ_V⁻¹D), et compute_h5_residual implémente exactement le résidu de eq:H5_compact/eq:H5_full avec P, Q, R, M conformes à eq:def_PQ–eq:def_RM (mêmes transposes, mêmes ordres de produit ; résidu (s×q) conforme). Résidu numérique au point AB ≈ 3e-17, y compris sur les équations de paires j≠k. À noter : l'annexe B (l.70) définit Σ_YY,n+1(k,j) = Var(Y_{n+1}|r_n=j, r_{n+1}=k) — l'objet pair-conditionnel correct — ce qui corrobore, côté papier, que l'usage de la variance marginale Σ_YY,n+1(k) dans la récursion IMM générale de la Sec. 3 (anomalie déjà confirmée par l'audit code) est bien l'élément incohérent.

**Preuve :** h5_constraint.py l.95-101 : P = Dt.T@C.T + SV@D.T ; Q = C@SU + D@Dt.T ; R = C@Dt + D@SV ; M = Q@C.T + R@D.T + SV ; residual = Z − P@solve(M,W) — terme à terme identique à eq:H5_compact.

**Suggestion :** Rien à corriger côté formules ; envisager d'importer dans le papier les nuances de nécessité déjà documentées dans la docstring (l.21-30).

## Section 2 — Modèle et hypothèses

_Audit de la section 2 (modèle et hypothèse H5) de paper/sections/02_model_h5.tex, avec macros.tex, abstract, introduction, appendices A-B, croisement avec la section 3 et le code prg/filter/gss_filter.py, et revérification Monte-Carlo (600k trajectoires, modèle AB biaisé K=2). Toutes les équations ont été redérivées à la main. Trouvaille principale (critique) : la moyenne du noyau de la Proposition 1 (eq:mu_jk) omet le terme de biais C_k(b_{X,j} − Δ_jΣ_{V,j}⁻¹b_{Y,j}) — confirmé numériquement — ce qui invalide la Remarque 2 (« bias on X is invisible ») et l'affirmation correspondante de l'abstract ; or l'extension « avec biais » est la contribution annoncée du papier, et le code, lui, implémente la version correcte. Trois autres problèmes sérieux : la seconde équation de (H3) est mal formulée (conditionnement sur (x_n, y_n) manquant, la version écrite contredit le modèle), la Remarque 1 affirme à tort la gaussianité de p(x_n, y_n | r_n) (c'est un mélange ; seule p(x_n | r_n, y_n) est gaussienne sous H5), et la Prop. 1 n'a ni loi initiale ni hypothèse d'homogénéité au premier pas, avec un appui circulaire de l'App. B sur l'App. A. En revanche Γ_jk, la pente du noyau, la Remarque 3, la caractérisation (H4) ⇒ C = 0, les attributions CGOMSM/CGPMSM/pairwise-triplet et toutes les dimensions/transposées sont corrects ; une erreur d'auteurs dans paper.bib (abbassi_cgpmsm_2015) a été détectée et sourcée._

### ✅ [CRITICAL] μ_jk omet le terme de biais C_k(b_{X,j} − Δ_jΣ_{V,j}⁻¹b_{Y,j}) : le noyau (R,Y) du papier est faux dès que b_X ≠ ΔΣ_V⁻¹b_Y

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/02_model_h5.tex:109-112 (eq:mu_jk), Proposition 1` — statut : confirmed (2 vote(s)) — catégorie : math-error

Redérivation : sous AB (paramétrisation effective de (H5)), X_n = Δ_jΣ_{V,j}⁻¹Y_n + c_j + ξ_n avec c_j = b_{X,j} − Δ_jΣ_{V,j}⁻¹b_{Y,j} et ξ_n ⊥ (Z_{n−1}, V_n), d'où p(x_n|r_n=j, y_n) = N(Δ_jΣ_{V,j}⁻¹y_n + c_j, Σ_{U,j} − Δ_jΣ_{V,j}⁻¹Δ_jᵀ) pour n ≥ 2, quelle que soit l'initialisation. En substituant dans E[Y_{n+1}|j,k,y_n] = C_k m_j(y_n) + D_k y_n + b_{Y,k}, on obtient μ_jk(y_n) = b_{Y,k} + (D_k + C_kΔ_jΣ_{V,j}⁻¹)y_n + C_k c_j. L'éq. (eq:mu_jk) omet le terme constant C_k c_j. La pente (D_k + C_kΔ_jΣ_{V,j}⁻¹) et Γ_jk (eq:Gamma_jk) sont, eux, corrects. Le code prg/filter/gss_filter.py (_precompute, l. 452–464) implémente la forme par moments stationnaires μ_Y(j,k) + M̃_{jk}(y_n − μ_Y(j)) qui CONTIENT ce terme (algébriquement C_k[μ_X(j) − K_jμ_Y(j)] = C_k c_j sous AB) ; de même la récursion de la section 3 (eq:y_pred_jk, 03_filtering.tex l. 165–177) le contient via μ_n(j). Le papier est donc incohérent avec son propre filtre : la Remarque rem:h5exact (03_filtering.tex l. 48–63, « matching the explicit expressions Γ_jk and μ_jk(·) of Proposition 1 ») n'est vraie qu'avec la formule corrigée. L'erreur se propage à l'abstract (00_abstract.tex l. 12–16) et à la contribution 1 de l'intro (01_introduction.tex l. 64–66, « explicit bias-aware Gaussian kernel »). C'est précisément l'extension « avec biais » — la contribution annoncée de ce papier par rapport au companion non biaisé — qui est fausse dans la Prop. 1.

**Preuve :** Monte-Carlo (600 000 trajectoires, K=2, q=s=1, modèle AB avec b_X=[1,−2], b_Y=[0.5,1]) : intercepts empiriques de E[Y_{n+1}|j,k,y_n] = {(0,0): 0.8317, (0,1): 0.6011, (1,0): −0.2009, (1,1): 1.8395} ; formule corrigée b_{Y,k}+C_k c_j = {0.8333, 0.6, −0.2, 1.84} (accord < 0.002) ; formule du papier b_{Y,k} = {0.5, 1.0, 0.5, 1.0} (écarts jusqu'à 0.84). Pentes et variances résiduelles : accord avec eq:mu_jk (partie linéaire) et eq:Gamma_jk à la précision MC.

**Suggestion :** Corriger eq:mu_jk en μ_jk(y_n) = b_{Y,k} + (D_k + C_kΔ_jΣ_{V,j}⁻¹)y_n + C_k(b_{X,j} − Δ_jΣ_{V,j}⁻¹b_{Y,j}), avec la précision « pour n ≥ 2, et pour n = 1 sous l'hypothèse d'initialisation stationnaire ». Mettre en cohérence l'abstract, la contribution 1 de l'intro et la Remarque rem:h5exact de la section 3.

**Ajustement de sévérité (vérificateurs) :** Keep critical. A headline proposition of the paper is mathematically false as printed, and the false claim is propagated to the abstract, intro contribution 1, Remark rem:bX_invisible, and rem:h5exact — and it hits precisely the bias extension that is this paper's announced contribution over the unbiased companion. Sole mitigating nuance for the report: the Sec. 3 filter recursion and the code are unaffected (they contain the term), so the fix is a one-term correction plus prose alignment, with no impact on algorithms or experimental results. | Keep critical. The false statement is Proposition 1 itself — the headline structural result of the paper — and its falsehood is reiterated as a selling point in the abstract ("b_{X,r} ... invisible in the observable pair"), the intro (contribution 1), and a dedicated Remark (rem:bX_invisible). Mitigating context (does not justify downgrade, but worth noting in the report): the implemented filter and hence all numerical results are unaffected, since both the Section 3 recursion and prg/filter/gss_filter.py already contain the missing term; the fix is a one-line correction to eq:mu_jk plus rewording of the abstract, intro, rem:bX_invisible, and the n>=2 / stationary-initialisation qualifier.

### ✅ [HIGH] Remarque 2 (« Bias on X is invisible ») fausse : b_X du régime source entre dans le noyau dès que C ≠ 0

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/02_model_h5.tex:120-130 (Remark rem:bX_invisible)` — statut : confirmed (2 vote(s)) — catégorie : claim-overreach

Conséquence directe du finding précédent : le noyau de transition de (R_n, Y_n) dépend du biais d'état b_{X,j} du régime SOURCE via C_k(b_{X,j} − Δ_jΣ_{V,j}⁻¹b_{Y,j}). Seule la partie « ne dépend pas de b_{X,k} du régime cible » est littéralement vraie. La phrase « the drift on X is absorbed in the prior law of X_n and does not affect the transition of the observable pair (R_n, Y_n) » est fausse — et l'invisibilité de b_X ne vaut que si C_k = 0, c.-à-d. exactement dans le cas CGOMSM/(H4) que le papier cherche à dépasser ; sous (H5) avec C ≠ 0, X (et son biais) rétroagit sur Y, c'est le point même du modèle. L'abstract (l. 13–16 : « depends on the observation bias b_{Y,r} but not on the state bias b_{X,r} (the latter is invisible in the observable pair) ») reprend la même affirmation fausse. La conclusion pratique de la remarque (mettre à jour les moments stationnaires de Z quand b_r ≠ 0, Sec. init) reste, elle, correcte — c'est justement par les moments stationnaires que b_X entre dans le noyau.

**Preuve :** MC ci-dessus : en changeant b_X seul (b_Y fixé), l'intercept empirique du noyau passe p.ex. de 1.0 à 1.84 pour (j,k)=(1,1) ; dépendance en b_{X,j} confirmée pour les 4 paires (j,k).

**Suggestion :** Reformuler : le noyau ne dépend pas de b_{X,k} (régime cible) mais dépend de b_{X,j} (régime source) via C_k ; b_X n'est invisible que dans le cas dégénéré C = 0 (CGOMSM). Corriger l'abstract en conséquence.

**Ajustement de sévérité (vérificateurs) :** Keep high. Although it shares a root cause with the missing-intercept error in eq:mu_jk (the "previous finding") and should be fixed jointly with it, this finding stands on its own: a false structural claim stated in the abstract and in a named Remark, false precisely in the C ≠ 0 regime that is the paper's main selling point. Mitigating note: the code is unaffected, so published numerics are not invalidated — it is a paper-statement error, not an algorithmic one. | Keep high for the paper text: a false structural claim is advertised in the abstract and in a labeled Remark, and it props up an incorrect closed-form proposition. One scoping note for the report: impact is confined to exposition (Prop. 1 closed form, Remark 2, abstract). The filter recursion of Sec. 3 (eq:y_pred_jk via regime moments, paper/sections/03_filtering.tex:165-183) and the code (prg/filter/gss_filter.py:444-467, mu_Y_jk = [C_k,D_k]μ(j)+b_Y(k)) carry the missing term through the stationary moments and are correct — no algorithmic or numerical consequence. The proposed fix (kernel independent of b_{X,k} but dependent on b_{X,j} via C_k; invisibility only when C=0; correct the abstract) is the right one.

### ✅ [HIGH] (H3), 2e partie : conditionnement sur (x_n, y_n) manquant — telle qu'écrite, l'hypothèse contredit le modèle qu'elle définit

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/02_model_h5.tex:21-25 (eq:H3, seconde équation)` — statut : confirmed (2 vote(s)) — catégorie : math-error

La seconde équation de (H3) s'écrit p(x_{n+1}, y_{n+1} | r_n, r_{n+1}) = p(x_{n+1}, y_{n+1} | r_{n+1}) — une contrainte sur les lois MARGINALES de z_{n+1} sachant les régimes. Or sous la forme état-espace (eq:dynamics), E[Z_{n+1}|r_n=j, r_{n+1}=k] = F_k μ_Z(j) + b_k et Var(Z_{n+1}|j,k) = F_k Σ_Z(j)F_kᵀ + Q_k dépendent de j dès que les moments par régime μ_Z(j), Σ_Z(j) diffèrent (cas générique, et cas du code qui calcule des μ_z[j], Σ[j] distincts). Telle qu'énoncée, (H3) forcerait donc tous les régimes à avoir les mêmes moments marginaux — absurde et incompatible avec eq:dynamics. La version voulue (celle qui livre F_{r_{n+1}}, b_{r_{n+1}}, W_{n+1} indexés par r_{n+1} seul) est conditionnelle à z_n : p(x_{n+1}, y_{n+1} | x_n, y_n, r_n, r_{n+1}) = p(x_{n+1}, y_{n+1} | x_n, y_n, r_{n+1}). Sans cette correction, la dérivation de eq:dynamics depuis (H1)–(H3) ne tient pas.

**Preuve :** eq:dynamics (l. 58–62) indexe F et b par r_{n+1} seul, ce qui est la conséquence de la version conditionnée à z_n ; la version marginale écrite l. 23–24 impliquerait F_kμ_Z(j)+b_k indépendant de j, i.e. μ_Z(1)=…=μ_Z(K), contredit par le modèle à régimes distincts (et par _precompute_stationary du code).

**Suggestion :** Écrire p(x_{n+1}, y_{n+1} | x_n, y_n, r_n, r_{n+1}) = p(x_{n+1}, y_{n+1} | x_n, y_n, r_{n+1}).

**Ajustement de sévérité (vérificateurs) :** Keep high, with one calibration note: this is a statement/transcription error localized to the hypothesis block — every downstream derivation (Prop. 1, Appendices A/B, the filter, the code) already uses the conditional semantics via eq:dynamics, so no result, algorithm, or numerical claim changes and the fix is one line. High is justified because the error sits in the paper's foundational assumptions and, read literally, makes the stated implication (H1)-(H3)+(H5) => eq:dynamics false and the model class degenerate — a referee-blocking must-fix. If the audit scale reserves high for errors that propagate into results, medium-high is acceptable. | Keep high. It is a definitional error in the foundational assumption set: as written, (H1)-(H3) cannot deliver eq:dynamics, and (H3) literally contradicts the paper's own Prop. 1, Appendix B, and the implemented model. Mitigating nuance for the report: the error is confined to the STATEMENT of (H3) — all downstream derivations implicitly use the correct conditional version, so no result is invalidated and the fix is a one-line edit (add conditioning on x_n, y_n to both sides).

### ❌ [HIGH] Remarque 1 : la factorisation est triviale et la gaussianité de p(x_n, y_n | r_n) est fausse en général

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/02_model_h5.tex:84-91 (Remark rem:marginal_form)` — statut : refuted (2 vote(s)) — catégorie : math-error

« p(x_n, r_n, y_n) = p(r_n) p(x_n, y_n | r_n) » est la règle de chaînage, vraie pour toute loi jointe — ce n'est pas une conséquence de (H1)–(H3)+(H5). Le contenu substantiel — « p(x_n, y_n | r_n) Gaussian » — est faux en général : p(z_n | r_n) = Σ_{r_{1:n−1}} p(r_{1:n−1}|r_n) p(z_n | r_{1:n}) est un MÉLANGE de gaussiennes sur les chemins de régimes (K^{n−1} composantes), non gaussien dès que K ≥ 2 et que les régimes diffèrent. Ce qui est vrai sous (H1)–(H3)+(H5) — et c'est ce dont la Prop. 1 a réellement besoin — est la gaussianité de p(x_n | r_n, y_n) : par (H2), p(x_n | r_{1:n}, y_{1:n}) est gaussienne, et par l'effondrement de l'App. A elle égale p(x_n | r_n, y_n), donc cette dernière est une gaussienne unique. En revanche p(y_n | r_n) reste un mélange, donc le produit p(x_n,y_n|r_n) n'est pas gaussien. L'expression « stationary in the form of its marginals » n'est ni définie ni démontrée (l'homogénéité (H1) n'implique aucune stationnarité sans hypothèse sur la loi initiale).

**Preuve :** Contre-argument structurel : même avec z_1|r_1 gaussienne par régime, p(z_2|r_2) = Σ_j p(r_1=j|r_2) N(F_{r_2}μ(j)+b, ·) est un mélange à K composantes distinctes ; aucune des hypothèses (H1)–(H3), (H5) ne le réduit à une gaussienne.

**Suggestion :** Remplacer la remarque par l'énoncé correct et utile : sous (H1)–(H3)+(H5), p(x_n | r_n, y_n) est gaussienne et (sous AB, pour n ≥ 2) indépendante de n ; préciser que les moments par régime de Z utilisés en pratique relèvent d'une hypothèse de stationnarité initiale explicite.

**Ajustement de sévérité (vérificateurs) :** Si on tenait à garder quelque chose : sévérité low, recentrée sur la rédaction (clause de factorisation triviale, locution « stationary in the form of its marginals » informelle) — mais le cœur mathématique de la trouvaille (gaussianité « fausse en général » sous (H1)–(H3)+(H5)) est réfuté par la seconde équation de (H3) telle qu'imprimée. | n/a (trouvaille réfutée ; au mieux un nitpick de formulation de sévérité low sur la locution « stationary in the form of its marginals »)

### ✅ [MEDIUM] Prop. 1 : homogénéité non justifiée (loi initiale jamais spécifiée) et appui circulaire sur la « noise law » de l'App. B

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/02_model_h5.tex:99-118 (Proposition 1) ; appendix/B_h5_derivation.tex l. 23-26` — statut : confirmed (2 vote(s)) — catégorie : derivation-gap

(a) L'homogénéité du noyau (eq:RY_kernel) exige que p(x_n | r_n=j, y_n) soit indépendante de n. Sous AB cela vaut pour n ≥ 2 quelle que soit l'initialisation (fait non trivial qui mériterait d'être énoncé), mais la transition n=1→2 exige que p(x_1 | r_1, y_1) ait déjà la forme N(Δ_jΣ_{V,j}⁻¹y_1 + c_j, Σ_{U,j} − Δ_jΣ_{V,j}⁻¹Δ_jᵀ) — or la section 2 ne spécifie jamais la loi initiale (π_1, p(z_1|r_1)) ni la matrice de transition de R. Sous (H5) seule (sans AB, cas non générique), il faut en plus une hypothèse de stationnarité pour que les moments conditionnels soient donnés par les blocs de bruit. (b) Le seul support interne du papier pour évaluer m_j, P_j « avec la loi du bruit » est l'App. B Step 1 (l. 23–26 : « The Markovianity argument of Appendix A … shows that all quantities relevant to (H5) may be evaluated with the noise law, i.e. with Z_n | r_n = j having mean zero and covariance P(j) ») — or l'App. A ne démontre que l'effondrement p(x_{n+1}|w_{1:n+1}) = p(x_{n+1}|w_{n+1}), rien sur les moments. Et avec biais, la « loi du bruit » CENTRÉE (mean zero) donne exactement le μ_jk erroné du finding critique : le centre effectif de la loi conditionnelle est c_j ≠ 0. Même problème dans eq:mean_X_linear de l'App. B (l. 51–54) où l'intercept est écrit b_{X,k} alors qu'il vaut b_{X,k} − β₂ b_{Y,k} sous la loi de bruit centrée (sans conséquence sur β₁ = 0, la remarque finale de l'App. B reste valide).

**Preuve :** Aucune occurrence de p(r_1), p(z_1|r_1) ou d'une hypothèse d'initialisation dans 02_model_h5.tex ; App. A l. 7–33 prouve uniquement l'effondrement ; le code (gss_filter.py l. 533–539) initialise explicitement le filtre aux moments stationnaires précalculés, hypothèse que le papier n'énonce pas.

**Suggestion :** Ajouter à la section 2 : la matrice de transition P de R (cf. finding notation), la loi initiale, et l'hypothèse sous laquelle le noyau est homogène dès n = 1 (initialisation stationnaire) ; dans l'App. B, remplacer la justification « noise law via App. A » par la dérivation directe X_n = Δ_jΣ_{V,j}⁻¹Y_n + c_j + ξ_n sous AB.

**Ajustement de sévérité (vérificateurs) :** keep medium | keep medium (part (a) slightly mitigated by the explicit companion-paper deferral in 02_model_h5.tex:27-28 and 96-97; avoid double-counting the bias-in-mu_jk consequence, already covered by the separate critical finding)

### ✅ [MEDIUM] Objectif de filtrage défini avec y_{1:N} (lissage) au lieu de y_{1:n} (filtrage)

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/02_model_h5.tex:143-144` — statut : confirmed (2 vote(s)) — catégorie : typo-math

La sous-section « The filtering problem » définit \hat X_n := \Ek{X_n}{\yN}, qui se rend E[X_n | y_{1:N}] (macro \yN = y_{1:N}, macros.tex l. 66) — c'est le LISSEUR. Tout le reste du papier (eq:mix_mean, eq:mix_var, section 3) cible E[X_{n+1} | y_{1:n+1}], le filtre. Il faut \yn (y_{1:n}).

**Preuve :** macros.tex l. 64-66 : \yn → y_{1:n}, \yN → y_{1:N} ; eq:mix_mean/eq:mix_var (l. 148-158) conditionnent sur \ynp = y_{1:n+1}.

**Suggestion :** Remplacer \Ek{X_n}{\yN} par \Ek{X_n}{\yn} ligne 144.

**Ajustement de sévérité (vérificateurs) :** medium → low-medium. Real and worth fixing — it misstates the paper's central objective in a paper whose contribution hinges on the filter/smoother distinction (smoothing is explicitly future work). But it is a single-token typo, used nowhere else, non-propagating into any derivation, and the intended meaning is unambiguous from the equations three lines below. "low" would also be defensible; "medium" is the ceiling. | Lower from medium to low. \hat X_n is never referenced again anywhere in sections/ or appendix/, and every downstream derivation correctly conditions on y_{1:n+1}, so the typo has no mathematical consequence — it is a one-macro fix. Still worth correcting because it misdefines the paper's central objective in the problem-statement subsection.

### ✅ [MEDIUM] Macro \Vark = E[AAᵀ|·] (moment d'ordre 2) utilisée comme variance centrée : eq:mix_var est fausse telle qu'affichée

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/macros.tex:macros.tex l. 25-26 ; 02_model_h5.tex l. 152-157 (eq:mix_var)` — statut : confirmed (2 vote(s)) — catégorie : notation

macros.tex définit \Vark{A}{c} → E[A Aᵀ | c] (moment brut, commentaire « Auto-covariance »). eq:mix_var écrit \Vark{X_{n+1}}{\ynp} = Σ_k π_k [\Vark{X_{n+1}}{k,\ynp} + (spread)(·)ᵀ] : c'est la loi de la variance TOTALE, correcte seulement pour des variances centrées. Avec la sémantique réelle de la macro, l'équation affichée dit E[XXᵀ|y] = Σ π_k E[XXᵀ|k,y] + Σ π_k (μ_k−μ)(μ_k−μ)ᵀ, ce qui est faux (le terme de dispersion est compté en double : E[XXᵀ|y] = Σ π_k E[XXᵀ|k,y] sans spread). Le papier mélange les deux sens : 03_filtering.tex l. 37 utilise \Ek{Z Zᵀ}{·} sciemment comme moment brut (P_n(k)), tandis que l. 32 appelle \Vark une « covariance ». Par ailleurs eq:noise_cov (l. 70-77) n'est correcte que si E[W_{n+1}|r_{n+1}] = 0, jamais énoncé.

**Preuve :** macros.tex l. 26 : \newcommand{\Vark}[2]{\Ek{#1\,#1\tp}{#2}} ; 02_model_h5.tex l. 152-157 : structure loi-de-variance-totale avec terme de dispersion.

**Suggestion :** Introduire une macro de variance centrée (p.ex. \Varc{A}{c} → Var(A|c) ou E[(A−E[A|c])(·)ᵀ|c]) et l'utiliser dans eq:mix_var ; énoncer W_{n+1} | r_{n+1} ~ N(0, ·).

**Ajustement de sévérité (vérificateurs) :** Keep medium. The error is notational with zero algorithmic impact, but it is pervasive (eq:mix_var, 03_filtering.tex:32, appendix E eq:joseph_proof l. 57, appendix B l. 70) and yields several displayed equations that are literally false under the paper's own demonstrated macro semantics. Drop or downgrade the eq:noise_cov/W-zero-mean sub-claim: w_{n+1} ~ N(0, Σ_{W,k}) is stated at 05_estimation.tex:20, just not in Sec. 2. | Keep medium. It is a genuinely false displayed equation plus inconsistent notation visible to any careful reviewer, but purely presentational: the actual recursion (eq:combine_var uses centered Kalman covariances) and the code are correct, so there is no algorithmic consequence. Soften the secondary eq:noise_cov claim from 'never stated' to 'not stated in Secs. 2-3 where it is used (only in Sec. 5, l. 20, in the estimation context)'.

### ✅ [LOW] P_{j,k} jamais défini ; propriétés du bruit W_{n+1} (moyenne nulle, blancheur, indépendance de Z_n) non énoncées

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/02_model_h5.tex:103-107 (eq:RY_kernel) ; 58-77` — statut : confirmed (1 vote(s)) — catégorie : notation

La matrice de transition P_{j,k} = p(r_{n+1}=k | r_n=j) apparaît pour la première fois dans eq:RY_kernel sans définition nulle part dans le papier (la section 2 ne définit pas non plus la distribution initiale de r_1). De même, eq:dynamics introduit W_{n+1} sans énoncer qu'il est gaussien, centré, indépendant de (Z_{1:n}, R_{1:n}) conditionnellement à r_{n+1}, et blanc en temps — propriétés toutes utilisées dans la Prop. 1 et l'App. B (elles découlent de (H1)–(H3) corrigées, mais devraient être énoncées dans une section « modèle »).

**Preuve :** grep : aucune définition de P_{j,k} dans sections/*.tex ; eq:dynamics--eq:noise_cov (l. 58-77) ne donnent que Var(W|r_{n+1}).

**Suggestion :** Ajouter après (H1) : « R est une chaîne homogène de matrice de transition P = (P_{j,k}) et de loi initiale ν » ; après eq:dynamics : « W_{n+1} | r_{n+1} ~ N(0, ·), indépendant de (Z_{1:n}, R_{1:n}) ».

**Ajustement de sévérité (vérificateurs) :** Downgrade from low to nitpick. The two headline claims (P_{j,k} never defined; initial law never given) are factually wrong — both exist in 03_filtering.tex (lines 156 and 334); only a forward-reference ordering nit and the implicit-noise-conventions point survive, in a recap section that explicitly defers the full model to the companion paper. The description should be rewritten to: "P_{j,k} and pi_0 are defined in Sec. 3 after first use in Sec. 2; noise conventions for W are implicit."

### ✅ [LOW] Équivalence (H5) ⟺ (H5bis) attribuée à un « time reversal » et à l'App. A : c'est la simple symétrie de l'indépendance conditionnelle

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/02_model_h5.tex:46-52 (eq:H5bis)` — statut : confirmed (1 vote(s)) — catégorie : derivation-gap

(H5) dit X_{n+1} ⫫ (R_n,Y_n) | (R_{n+1},Y_{n+1}) ; (H5bis) dit (R_n,Y_n) ⫫ X_{n+1} | (R_{n+1},Y_{n+1}). C'est la même indépendance conditionnelle, l'équivalence p(x|a,b) = p(x|b) ⟺ p(a|b,x) = p(a|b) étant la symétrie élémentaire de ⫫ (densités positives) — aucun renversement du temps n'est nécessaire. L'App. A, citée comme preuve, démontre un énoncé différent et plus fort (l'effondrement p(x_{n+1}|w_{1:n+1}) = p(x_{n+1}|w_{n+1})). Le renvoi est donc trompeur.

**Preuve :** 02_model_h5.tex l. 46-47 : « which is equivalent (after time reversal, see Appendix~\ref{app:time_reversal}) to » ; App. A l. 7-9 prouve p(x_{n+1}|w_{1:n+1}) = p(x_{n+1}|w_{n+1}), pas l'équivalence H5/H5bis.

**Suggestion :** Écrire « équivalent par symétrie de l'indépendance conditionnelle » et réserver le renvoi à l'App. A pour l'effondrement (eq:H5_collapse_*).

**Ajustement de sévérité (vérificateurs) :** none — [low] is correct (presentation/cross-reference defect; the underlying mathematical claim is true)

### ✅ [LOW] « Under (H1)–(H3) and (H5), the joint dynamics admit a fully coupled state-space representation » : (H5) n'y joue aucun rôle

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/02_model_h5.tex:56-58` — statut : confirmed (1 vote(s)) — catégorie : claim-overreach

La représentation eq:dynamics–eq:noise_cov découle de (H1)–(H3) seules (version corrigée de H3) ; (H5) ne rend pas la représentation possible, elle CONTRAINT ses blocs (c'est l'objet de la section 4 : une équation matricielle sur (A,…,Σ_V)). La formulation suggère à tort que (H5) participe à l'existence de la forme état-espace, alors que son rôle est inverse — restreindre les paramètres pour permettre le filtrage exact.

**Preuve :** Section 4 / App. B dérivent (H5) COMME contrainte sur les paramètres de eq:dynamics, supposée déjà acquise sous (H1)–(H3).

**Suggestion :** « Under (H1)–(H3), the joint dynamics admit … ; (H5) will further constrain the blocks (Sec. IV). »

**Ajustement de sévérité (vérificateurs) :** Keep at low. It is a wording/attribution imprecision in exposition, not a mathematical error — a one-line fix with no impact on any derivation or result. Low is the correct floor; do not raise.

### ✅ [LOW] « it computes an exact mode-probability update » : les probabilités de mode de l'IMM ne sont exactes qu'au premier pas

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/01_introduction.tex:19-26` — statut : confirmed (1 vote(s)) — catégorie : claim-overreach

L'intro affirme que l'IMM « computes an exact mode-probability update but, in the state branch, it forces the per-mode state mixture … back to a single Gaussian ». La FORMULE de mise à jour des probabilités de mode est exacte, mais elle est évaluée avec des vraisemblances issues des gaussiennes effondrées du pas précédent : dès le deuxième cycle, les probabilités de mode sont elles aussi approchées. La formulation de l'abstract (« its only error source is the Gaussian collapse ») est la version défendable ; celle de l'intro surinterprète. Accessoirement, le poids μ_{j|k} dans la mixture affichée (l. 22) n'est pas défini et devrait être p(r_{n−1}=j | r_n=k, y_{1:n}) pour que la décomposition affichée soit exacte.

**Preuve :** 01_introduction.tex l. 19-21 ; comparer 00_abstract.tex l. 5-8.

**Suggestion :** « the Gaussian collapse of the per-mode mixture is the algorithm's single approximation, which then contaminates both the state and the mode-probability branches ».

**Ajustement de sévérité (vérificateurs) :** Keep at low. It is a wording/precision issue confined to the introduction (the abstract already states it correctly, and no derivation in Sec. 3 or the appendices relies on the erroneous phrasing), but it is worth fixing in a paper whose central claim is exactness, and the undefined mu_{j|k} (also at 03_filtering.tex:221) should be defined in the same pass.

### ✅ [LOW] Liste d'auteurs erronée pour abbassi_cgpmsm_2015 : il manque Benboudjema et Derrode ; « Benmiloud » n'est pas auteur

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/paper.bib:102-111 (entrée abbassi_cgpmsm_2015)` — statut : confirmed (1 vote(s)) — catégorie : typo

L'entrée bib donne « Abbassi, Noufel and Benmiloud, Bachir and Pieczynski, Wojciech ». Les auteurs réels de « Optimal Filter Approximations in Conditionally Gaussian Pairwise Markov Switching Models » (IEEE TAC 60(4):1104–1109, 2015) sont N. Abbassi, D. Benboudjema, S. Derrode et W. Pieczynski — l'entrée omet deux auteurs (dont S. Derrode lui-même) et substitue B. Benmiloud (collaborateur d'autres travaux HMC des années 1990) à D. Benboudjema. Sources : [IEEE Xplore 6858006](https://ieeexplore.ieee.org/document/6858006), [HAL hal-01157814](https://hal.science/hal-01157814). Le reste des attributions vérifiées est correct : derrode_cgomsm_2013 (Derrode & Pieczynski, IEEE SPL 20(7):701–704, 2013, CGOMSM), pieczynski_pairwise_2003, pieczynski_triplet_2002, blom_interacting_1988.

**Preuve :** paper.bib l. 103 : « author = {Abbassi, Noufel and Benmiloud, Bachir and Pieczynski, Wojciech} » vs page IEEE Xplore/HAL du même article (volume/numéro/pages identiques à l'entrée).

**Suggestion :** author = {Abbassi, Noufel and Benboudjema, Dalila and Derrode, St\'ephane and Pieczynski, Wojciech}.

**Ajustement de sévérité (vérificateurs) :** Keep at low: purely bibliographic, no impact on math or results. Slightly notable in context since the omitted author (Derrode) is the paper's own author, making it an easy and worthwhile one-line fix.

### ✅ [INFO] Éléments revérifiés et corrects : Γ_jk, pente de μ_jk, Remarque 3, caractérisation de (H4) ⇒ C = 0, dimensions/transposées

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/02_model_h5.tex:32-37, 113-116, 132-138, 148-158` — statut : confirmed (1 vote(s)) — catégorie : verified-ok

Pour traçabilité, les vérifications qui PASSENT : (i) eq:Gamma_jk est correcte (Γ_jk = Σ_{V,k} + C_k(Σ_{U,j} − Δ_jΣ_{V,j}⁻¹Δ_jᵀ)C_kᵀ = Σ_{V,k} + C_k Var(X_n|j,y_n) C_kᵀ, redérivée et confirmée par MC, et PSD par construction — Schur du bruit) ; cohérente avec le correctif déjà validé côté code pour la section 3. (ii) La partie linéaire de eq:mu_jk (pente D_k + C_kΔ_jΣ_{V,j}⁻¹) est correcte. (iii) La Remarque 3 (rem:Gamma_bias_free) est vraie : Γ_jk est indépendante des biais (vérifié MC). (iv) L'affirmation l. 32-37 que (H4) force C_r = 0 est correcte : sous la forme état-espace, p(y_{n+1}|x_n, y_n, r_{n:n+1}) a pour moyenne C_{r_{n+1}}x_n + …, indépendante de x_n ssi C = 0 (support plein) ; attribution CGOMSM/CGPMSM cohérente avec la littérature. (v) Dimensions et transposées de toutes les équations de la section (q, s, blocs A,B,C,D, Δ q×s, b_r ∈ R^{q+s}) cohérentes entre elles, avec macros.tex et avec le code (q = dim X, s = dim Y). (vi) eq:mix_mean et la structure de eq:mix_var (loi de la variance totale, modulo le problème de macro signalé) sont correctes.

**Preuve :** MC 600k trajectoires : var. résiduelles {0.3671, 0.5894, 0.4113, 0.6516} vs Γ_jk théoriques {0.3667, 0.5960, 0.4050, 0.6512} ; invariance de Γ sous changement de b_X, b_Y vérifiée.

**Ajustement de sévérité (vérificateurs) :** None — [info] is the right level for a positive traceability/verification record; keep as is.

## Section 5 + annexe D — Estimation

_Audit de paper/sections/05_estimation.tex (190 l.) et paper/appendix/D_baum_welch.tex (112 l.), avec redérivation manuelle (factorisation de la vraisemblance supervisée, fonction Q de l'EM et M-steps exacts, récursions forward-backward) et confrontation à prg/learning/supervised.py, prg/learning/semi_supervised.py et prg/utils/h5_constraint.py. Le coeur est sain : conditionnement sur r_{n+1} correct et conforme au modèle (02_model_h5.tex l.59-60), OLS=MLE pour (F,b,Sigma_W), forward/backward/gamma/xi corrects et fidèlement implémentés, M-step pondéré (équation normale, covariance résiduelle) exact, initialisation k-means sur les différences premières et tri par A_k[0,0] conformes au code. Trois problèmes substantiels : la monotonie EM affirmée (et répétée dans abstract/intro/expériences/conclusion) n'est pas garantie car la mise à jour des moments initiaux de l'annexe D n'est pas un M-step (le seul terme de vraisemblance est gamma_1(k) log N(z_1;.)) ; le mode contraint-à-chaque-itération est incorrectement qualifié de « Generalized EM (Wu 1983) » alors qu'un GEM est précisément monotone ; et pi_0 est décrit (fréquences lissées / gamma_1) mais le code retourne ou exporte systématiquement pi0=None (distribution stationnaire). Sur le point 3 de la commande : le papier décrit bien la projection AB post-hoc par défaut, et l'Algorithme 2 reflète fidèlement la sélection multi-restart sur vraisemblances pré-projection — sans toutefois signaler que la vraisemblance rapportée ne correspond pas au modèle projeté retourné. Le reste est de la précision de pseudo-code (bornes de sommes de eq:M_step_P, alignement des labels k-means, l^{(0)} indéfini, conventions de régimes dégénérés)._

### ✅ [HIGH] La monotonie EM affirmée n'est pas garantie par le M-step réellement spécifié (moments initiaux, clamp SPD, resets)

`paper/appendix/D_baum_welch.tex (+ paper/sections/05_estimation.tex):App D lignes 80-112 ; 05_estimation.tex lignes 131-140` — statut : confirmed (2 vote(s)) — catégorie : claim-overreach

Le papier affirme qu'en mode post-hoc « the EM iterations themselves are unconstrained and the log-likelihood remains monotone non-decreasing » (05, l.133-135 ; répété dans l'abstract l.38, l'intro l.98, 06_experiments l.322 « strictly monotone, as guaranteed by theory », conclusion l.25). Or le M-step décrit en annexe D n'est pas le maximiseur exact de Q sur tous les paramètres : (1) la mise à jour de (mu_z0,k, Sigma_z0,k) (eqs mu_z0_weighted/Sigma_z0_weighted, App D l.93-103) pondère TOUS les z_n par gamma_n(k), alors que la vraisemblance ne contient qu'UN terme en ces paramètres — l'émission initiale iota(k)=log N(z_1; mu_z0,k, Sigma_z0,k) (App D l.33-35), de sorte que le terme de Q correspondant est gamma_1(k) log N(z_1; ., .). Le vrai maximiseur est dégénéré (mu=z_1, Sigma->0) ; la formule implémentée est une heuristique de moment-matching qui peut faire DÉCROÎTRE la vraisemblance (ces paramètres entrent dans log p(z_{1:N}) via alpha_1, cf. code _forward l.172). (2) Le clamp SPD (App D l.88-89 et l.103) et (3) le reset des régimes dégénérés (App D l.105-112) dévient aussi du M-step exact. Les mises à jour de P, pi_0, (F_k, b_k, Sigma_W,k) sont, elles, exactes (vérifié : OLS pondéré = maximiseur GLS indépendant de Sigma ; covariance résiduelle pondérée avec le nouveau Theta = maximiseur joint). La monotonie est donc vraie « à l'heuristique des moments initiaux et aux garde-fous numériques près », pas inconditionnellement.

**Preuve :** 05_estimation.tex l.133-135 : « the EM iterations themselves are unconstrained and the log-likelihood remains monotone non-decreasing ». App D l.95-101 : mu_z0,k = sum_n gamma_n(k) z_n / sum_n gamma_n(k) (somme sur tout n), alors que le seul terme de vraisemblance dépendant de mu_z0,k est iota(k) en n=1 (App D l.33-35). Code conforme à l'annexe : semi_supervised.py l.575-584 (mise à jour pondérée sur tout n) et l.516-519 (log_init évalué en Z[0] seulement).

**Suggestion :** Soit geler (mu_z0, Sigma_z0) pendant l'EM (les fixer à l'init) et la monotonie redevient exacte pour les paramètres restants ; soit qualifier la phrase : « monotone à l'exception de la mise à jour heuristique des moments initiaux (poids 1/N dans la vraisemblance) et des garde-fous SPD » ; et adoucir 06 l.322 (« strictly monotone, as guaranteed by theory »).

**Ajustement de sévérité (vérificateurs) :** High is defensible for a paper-correctness audit since the false guarantee ("as guaranteed by theory", "monotone by construction") is repeated in five places (abstract, intro, 05, 06, conclusion); medium would also be reasonable because the offending term has weight ~1/N in the likelihood, the empirical curves are monotone in practice, the main contribution (exact filter) is unaffected, and the fix is a local wording change or freezing the initial moments. | Lower from high to medium: the mathematical gap is real and the 'guaranteed by theory' wording is wrong, but it affects only the O(1) initial-moment terms of the likelihood, has negligible practical effect on the experiments, and is fixed by qualifying the claim (05 l.133-135, 06 l.322, abstract/intro/conclusion) or freezing (mu_z0, Sigma_z0) during EM.

### ✅ [MEDIUM] Le mode (ii) est appelé « Generalized EM (Wu 1983) » alors qu'un GEM garantit la monotonie — contradiction interne

`paper/sections/05_estimation.tex:lignes 135-138 (et 08_conclusion.tex l.25)` — statut : confirmed (2 vote(s)) — catégorie : claim-overreach

Le papier écrit : « at every M-step, yielding a Generalized EM [wu_convergence_1983] in which the constraint is satisfied throughout ... but the log-likelihood is no longer guaranteed to be monotone ». C'est contradictoire : par définition (Wu 1983), un GEM choisit theta' tel que Q(theta'|theta) >= Q(theta|theta), ce qui IMPLIQUE la monotonie de la vraisemblance. L'écrasement A,B <- (Delta Sigma_V^{-1} C, Delta Sigma_V^{-1} D) après maximisation de Q ne garantit pas l'augmentation de Q : ce n'est pas un GEM mais un « EM projeté / contraint » heuristique. Le docstring du code fait la même confusion (semi_supervised.py l.29-31, 734-736).

**Preuve :** 05_estimation.tex l.135-138 : « yielding a Generalized EM~\cite{wu_convergence_1983} in which the constraint is satisfied throughout the optimisation but the log-likelihood is no longer guaranteed to be monotone ». Wu (1983), Def. GEM : tout M-step augmentant Q préserve L(theta^{(i+1)}) >= L(theta^{(i)}).

**Suggestion :** Renommer « projected EM » ou « constrained-EM heuristic » (ou montrer que la projection augmente Q, ce qui n'est pas le cas en général) ; corriger aussi conclusion l.25.

**Ajustement de sévérité (vérificateurs) :** Keep medium. It is a terminology/citation-correctness error with no numerical impact, but it is pervasive (abstract, introduction, estimation, experiments, real-data, conclusion, algorithm pseudo-code, and code docstrings), so the fix touches more locations than the two cited in the finding. | Keep medium. Not a derivation error (the non-monotonicity caveat itself is honest and empirically illustrated), but the mislabel is broader than the finding lists: it also appears in 00_abstract.tex l.39, 01_introduction.tex l.99, 06_experiments.tex l.299/317-336 (including the false inference that oscillations show the GEM interpretation is "empirically sound"), 07_real_data.tex l.145/164-166, generated table captions, and multiple code docstrings/CLI help. Fix is mechanical (rename to "projected EM"/"EM with M-step projection", drop or recontextualize the Wu citation, and reword the 06_experiments sentence) but must be applied at all occurrences.

### ✅ [MEDIUM] pi_0 : le papier dit « fréquences avec lissage de Laplace » (supervisé) et « gamma_1 » (EM), mais le code n'exporte jamais un pi_0 estimé

`paper/sections/05_estimation.tex:Algorithme 1, lignes 43-44 ; eq:M_step_P ligne 105` — statut : confirmed (2 vote(s)) — catégorie : code-paper-mismatch

Algorithme 1 (l.43-44) : « Estimate pi_0(k) from regime frequencies (with Laplace smoothing) ». Or l'estimateur supervisé du code retourne pi0=None, interprété comme « distribution stationnaire de P » (prg/learning/supervised.py l.424 et commentaire l.534 du fichier généré). L'estimateur par fréquences lissées n'existe dans le code QUE comme initialiseur interne de l'EM (prg/learning/semi_supervised.py l.421-423). Côté EM, la mise à jour pi_0(k)=gamma_1(k) (eq:M_step_P l.105) est bien calculée (semi_supervised.py l.557) et utilisée pendant les itérations, mais _generate_model_code (partagé par les deux modules) écrit en dur « pi0: ... = None  # None -> stationary distribution » (supervised.py l.534) : le pi_0 estimé est silencieusement jeté à l'export du modèle. Estimer en outre P(r_1=k) par les fréquences d'occupation de toute la trajectoire estime la loi stationnaire, pas la loi initiale — ce n'est pas un MLE (le MLE sur une trajectoire est l'indicatrice de r_1).

**Preuve :** supervised.py l.424 : '"pi0": None' ; l.534 : '"    pi0: np.ndarray | None = None   # None → stationary distribution"' (littéral codé en dur, ignore params["pi0"]) ; semi_supervised.py l.557 : 'pi0 = gamma[0] / gamma[0].sum()' jamais persisté dans le .py généré.

**Suggestion :** Soit aligner le papier (« pi_0 est pris égal à la distribution stationnaire de P_hat dans l'implémentation ») soit corriger le code pour exporter params['pi0'] ; au minimum supprimer la mention « Laplace smoothing » de l'Algorithme 1, qui décrit l'initialiseur EM et non l'estimateur supervisé.

**Ajustement de sévérité (vérificateurs) :** Keep medium or lower to low-medium. The finding is fully real, but its practical impact is confined to the initial distribution: for the supervised path, frequencies and the stationary law of P-hat nearly coincide (difference is O(1/N) plus Laplace smoothing), so that half is near-cosmetic; the substantive half is the EM-estimated gamma_1 silently replaced by the stationary distribution at export, which only affects the first time step of downstream simulation/filtering and none of the paper's main theoretical claims. The proposed fix (align the paper text, drop the 'Laplace smoothing' mention from Algorithm 1, or export params['pi0']) is appropriate. | Lower from medium to low. It is a genuine paper/code and paper-internal (Sec. 3 vs Sec. 5) consistency defect plus a minor export/reproducibility wart, but it has no bearing on the paper's mathematical claims or experimental results, and the stationary-distribution convention the code actually follows is announced in Sec. 3 (ssec:init).

### ✅ [MEDIUM] Les moments initiaux (mu_z0,k, Sigma_z0,k) sont présentés sous l'étiquette MLE alors qu'ils sont un moment-matching hors vraisemblance

`paper/sections/05_estimation.tex:lignes 16-25 ; Algorithme 1 lignes 62-63` — statut : confirmed (2 vote(s)) — catégorie : claim-overreach

La sous-section s'ouvre par « Maximum-likelihood estimation factorises across regimes » puis enchaîne « Sample moments (mu_z0,k, Sigma_z0,k) provide the initial distributions » (l.23-25 ; Alg. 1 l.62-63 : moyenne/covariance de {z_n : r_n = k} sur toute la trajectoire). Or la vraisemblance complète p(r_{1:N}, z_{1:N}) = p(r_1) p(z_1|r_1) prod p(r_{n+1}|r_n) p(z_{n+1}|z_n, r_{n+1}) ne contient qu'un seul facteur p(z_1|r_1) impliquant ces paramètres : leur MLE sur une trajectoire est dégénéré. Les moments empiriques par régime estiment la loi marginale intra-régime (≈ stationnaire), pas la loi initiale Z_1|r_1=k définie en 03_filtering.tex l.336-337. La factorisation MLE est exacte pour (F_k, b_k, Sigma_W,k) et P (vérifié : conditionnement sur r_{n+1}=k conforme au modèle Z_{n+1}=F_{r_{n+1}}Z_n+b_{r_{n+1}}+W_{n+1}, 02_model_h5.tex l.59-60 ; diviseur N_k du MLE pour Sigma_W cohérent avec supervised.py l.236), mais pas pour les conditions initiales.

**Preuve :** 05_estimation.tex l.17-25 : « Maximum-likelihood estimation factorises across regimes ... Sample moments (mu_z0,k, Sigma_z0,k) provide the initial distributions » ; 03_filtering.tex l.336-337 définit mu_z0(k)=E[Z_1|r_1=k]. Le code fait la même heuristique (supervised.py l.394-401).

**Suggestion :** Une phrase suffit : « the initial-state moments are set by per-regime moment matching (the single-trajectory MLE of the initial law is degenerate), implicitly assuming within-regime stationarity ».

**Ajustement de sévérité (vérificateurs) :** Baisser de medium à low : imprécision d'exposition réelle (heuristique de moment-matching placée sous la bannière MLE sans la distinguer), mais sans affirmation explicitement fausse, sans équation erronée ni conséquence en aval, et cohérente avec le cadre stationnaire que le papier assume déjà ; correction en une phrase. | Lower from medium to low (or low-to-medium). It is a presentational/precision issue with zero numerical consequence (initialization affects one filter step and one likelihood factor out of N), the heuristic is standard practice, and the paper's "stationary moments of Z" remark in 02_model_h5.tex l.127-128 partially documents the intended convention. A one-sentence caveat in Sec. 5 fully resolves it; optionally extend the caveat to the EM monotonicity statement (l.133-140), which is technically affected through the initial-emission term.

### ✅ [MEDIUM] Sélection multi-restart sur la vraisemblance PRÉ-projection : le papier décrit fidèlement le code mais n'avertit jamais que l_hat ne correspond pas au modèle retourné

`paper/sections/05_estimation.tex:Algorithme 2, lignes 183-186 ; ligne 129` — statut : confirmed (2 vote(s)) — catégorie : derivation-gap

Réponse au point 3 de l'audit : oui, le papier décrit le même comportement que le code — Alg. 2 place « theta_hat <- argmax l » (l.183) AVANT « Apply AB constraint to each regime of theta_hat » (l.184-186), donc la comparaison des restarts se fait sur des vraisemblances non contraintes, puis le modèle est écrasé par la projection AB. Mais ni la section 5 ni l'annexe D ne signalent que (i) après l'écrasement A,B la vraisemblance du modèle retourné diffère (typiquement diminue) de l rapportée ; (ii) le classement des restarts selon l non contrainte peut différer du classement post-projection — le « best run » pour la famille contrainte n'est pas garanti ; (iii) la « highest converged log-likelihood » (l.129) n'est donc pas la vraisemblance de theta_hat final. Le code aggrave le point : le docstring du modèle généré affiche « log L=... » pré-projection comme si c'était celle du modèle sauvegardé (semi_supervised.py l.952-960).

**Preuve :** 05_estimation.tex l.183-186 : argmax sur B puis « Apply AB constraint ... to each regime of theta_hat » ; semi_supervised.py l.779 (sélection sur info['log_lik'] non contraint), l.589-616 (projection post-hoc après sélection... en fait dans _em_run, avant sélection, mais log_lik enregistré avant projection), l.952-960 (docstring avec log L pré-projection).

**Suggestion :** Recalculer et rapporter log p(z_{1:N} | theta_hat_projeté) (un seul passage forward suffit) ; idéalement sélectionner le restart sur cette quantité quand ab=true ; au minimum ajouter une phrase d'avertissement dans 5.2.4 et corriger le docstring généré.

**Ajustement de sévérité (vérificateurs) :** Keep medium, but at the lower edge. Point (ii) (restart re-ranking) should be presented as a theoretical caveat only — the paper's own restart study shows it cannot have affected the reported results. The strongest concrete element is not the missing warning in §5.2.4 (the timing is openly disclosed there and in Alg. 2) but the misattributed 'train log L̂' for V1 in tab_enso_em (§7.3) and the code-generated docstring; the recommended fix (recompute log p(z|θ̂_projected) with one forward pass and report it, plus one caveat sentence in §5.2.4) is correct and cheap. | Keep medium. The core theory (exact H5 filtering) is unaffected, but the issue is stronger than a pure documentation nit: the paper's Table tab_enso_em and the text at 07_real_data.tex l.153-157 present the pre-projection likelihood as the V1 (post-hoc AB) variant's training log-likelihood and draw a conclusion from its equality with V0. Medium, on the firm side.

### ✅ [LOW] Bornes des sommes implicites dans la réestimation de P : lue avec n=1..N au dénominateur, P_hat n'est plus stochastique

`paper/sections/05_estimation.tex:eq:M_step_P, ligne 103` — statut : confirmed (1 vote(s)) — catégorie : notation

P_hat_{j,k} = sum_n xi_n(j,k) / sum_n gamma_n(j) sans bornes. xi_n n'est défini que pour n=1..N-1 ; pour que les lignes somment à 1 (via l'identité sum_k xi_n(j,k)=gamma_n(j), App D l.59-60), le dénominateur doit être sum_{n=1}^{N-1} gamma_n(j). Si le lecteur inclut n=N, P_hat est sous-normalisée. Le code utilise la bonne plage : gamma[:-1].sum(axis=0) (semi_supervised.py l.544).

**Preuve :** 05_estimation.tex l.102-104 : « \widehat P_{j,k} = \frac{\sum_n \xi_n(j,k)}{\sum_n \gamma_n(j)} » — aucune borne ; semi_supervised.py l.544 : 'denom = gamma[:-1].sum(axis=0)'.

**Suggestion :** Écrire explicitement sum_{n=1}^{N-1} au numérateur et au dénominateur.

**Ajustement de sévérité (vérificateurs) :** Keep at low. It is a genuine notational ambiguity that can mislead a reader re-implementing from the paper, but the asymptotic bias is O(1/N), the reference code is correct (and re-normalises rows anyway), and the fix is a one-line edit. Minor nit: the code citation should be line 543 rather than 544.

### ✅ [LOW] L'Algorithme 2 idéalise l'implémentation : initialiseur différent d'Alg. 1, alignement k-means non spécifié, l^{(0)} indéfini, l rapportée décalée d'un M-step à I_max

`paper/sections/05_estimation.tex:Algorithme 2, lignes 166-181` — statut : confirmed (1 vote(s)) — catégorie : code-paper-mismatch

(a) L.167 « Run Algorithm 1 on (R^(0), X, Y) » : le code n'appelle pas l'estimateur supervisé mais _initialize_params_from_R (semi_supervised.py l.384-463), plus tolérant : lignes de P vides remplacées (uniforme), clusters < dim_z+1 points -> F=I, Sigma_W = cov globale + 0.1 I, pi_0 fréquences lissées ; Alg. 1/fit_supervised lèverait une erreur sur régime vide (supervised.py l.331-337, 356-360). (b) Le k-means sur {Delta z_n} produit N-1 labels ; la convention du code (label de Delta z_n affecté au temps n+1, r_1 := r_2 ; semi_supervised.py l.371-373) n'est pas dans le papier alors qu'Alg. 1 exige N labels. (c) L.178 : le test |l^{(i)} - l^{(i-1)}| < eps utilise l^{(0)} jamais défini (le code saute le test à it=0, l.533). (d) Le code évalue l au E-step SUIVANT : si I_max est atteint sans convergence, la paire (theta, l) ajoutée à B est incohérente — theta est post-dernier-M-step, l est pré-dernier-M-step (semi_supervised.py l.512-537 vs Alg. 2 l.177 qui calcule l^{(i)} sur theta^{(i)}). (e) L.129 « highest converged log-likelihood » : les runs non convergés (I_max) concourent aussi (code l.775-781).

**Preuve :** semi_supervised.py l.384-463 (_initialize_params_from_R), l.371-373 (R_init[1:]=labels ; R_init[0]=labels[0]), l.528-537 (log_lik enregistré avant le M-step de l'itération), l.775-781 (sélection sans filtre de convergence).

**Suggestion :** Préciser dans Alg. 2 la convention d'alignement des labels et remplacer « Run Algorithm 1 » par « run a robustified supervised fit (Appendix D conventions) » ; définir l^{(0)} = log p(z|theta^{(0)}) ; évaluer l après le dernier M-step.

**Ajustement de sévérité (vérificateurs) :** Keep at low — correctly calibrated. The discrepancies are documentation/idealization issues in Algorithm 2's presentation, not mathematical errors; sub-points (d) and (e) could mildly affect reproducibility and restart selection but not the method's correctness.

### ✅ [LOW] Régimes dégénérés : pour (mu_z0, Sigma_z0) le papier annonce un reset, le code conserve les valeurs précédentes

`paper/appendix/D_baum_welch.tex:lignes 105-112` — statut : confirmed (1 vote(s)) — catégorie : code-paper-mismatch

App D : le M-step est « skipped and the parameters (F_k, b_k, Sigma_W,k) are reset to identity dynamics with unit noise covariance. The same convention applies to (mu_z0,k, Sigma_z0,k) ». Pour (F, b, Sigma_W) le code est conforme (semi_supervised.py l.297-302 : F=I, b=0, Sigma_W=I — noter que le commentaire du code dit « large noise » mais utilise l'identité). Pour (mu_z0, Sigma_z0), en revanche, le code SAUTE simplement la mise à jour et garde les valeurs de l'itération précédente (l.576-584 : 'if denom_k > _LOG_FLOOR'), sans reset. « The same convention » est donc inexact.

**Preuve :** semi_supervised.py l.576-584 : la branche else n'existe pas — pas de reset de mu_z0_list[k]/Sigma_z0_list[k] ; App D l.110-112 affirme la même convention que le reset de F.

**Suggestion :** Écrire « ... the update of (mu_z0,k, Sigma_z0,k) is likewise skipped (previous values are retained) when the regime is empty ».

**Ajustement de sévérité (vérificateurs) :** Keep [low]. It is a paper/code consistency error in an appendix describing degenerate edge-case handling: no mathematical error, observable behavioral difference only when a regime collapses to zero weight during EM. The fix is a one-sentence wording change in App D (or a two-line else-branch in the code if reset semantics are preferred).

### ✅ [LOW] Ordre canonique par A_k[0,0] : critère dégénéré précisément dans les cas spéciaux du papier (delta-zero / C=0 => A_k=0 pour tout k)

`paper/sections/05_estimation.tex:lignes 142-147 ; Algorithme 2 lignes 184-187` — statut : confirmed (1 vote(s)) — catégorie : claim-overreach

Le tri décroissant sur A_k[0,0] (l.145-146 ; code _reorder_regimes, semi_supervised.py l.672, conforme) est appliqué APRÈS la projection AB (Alg. 2 l.184-187). Or sous la contrainte AB, A_k = Delta_k Sigma_V,k^{-1} C_k : avec l'option --delta-zero (Delta=0) ou dans le cas C=0 mis en avant par le papier (04_constraint.tex, table tab:h5_special_cases l.120-125, cas CGOMSM), A_k = 0 pour TOUS les régimes — la clé de tri est identiquement nulle et l'ordre « canonique » devient l'ordre arbitraire du sort stable. Le but affiché (« comparable across runs or to a ground-truth assignment ») n'est alors pas atteint. Par ailleurs « Any other permutation invariant ordering » (l.147) : formulation impropre (il s'agit d'un critère équivariant par permutation des labels, et il manque le trait d'union). Identifiabilité au sens large : seul le label switching est traité ; K supposé connu et l'identifiabilité de la famille (H5) ne sont pas discutés (acceptable, mais à signaler).

**Preuve :** 05_estimation.tex l.145-146 : « sorting regimes in decreasing order of A_k[0,0] » ; Alg. 2 l.184-187 : projection AB puis reorder ; 04_constraint.tex l.120-121 : « C = 0 => A = 0 » ; semi_supervised.py l.672 : 'sorted(range(K), key=lambda k: -params["A_list"][k][0, 0])'.

**Suggestion :** Choisir une clé jamais dégénérée sous AB, p.ex. trier sur D_k[0,0] ou sur trace(Sigma_V,k), ou trier AVANT projection ; corriger « permutation-invariant » -> « label-permutation-equivariant criterion ».

**Ajustement de sévérité (vérificateurs) :** Keep [low] — correctly calibrated. Edge-case robustness defect in a label-switching convention; no impact on filtering theory or estimation math. Affects reproducibility only when AB projection is combined with Delta=0 or C=0 (CGOMSM-style) settings.

### ✅ [INFO] M-step en norme euclidienne : correct, mais l'équivalence OLS=GLS qui le justifie n'est pas mentionnée

`paper/sections/05_estimation.tex (+ appendix/D_baum_welch.tex):eq:M_step_Theta lignes 112-118 ; App D lignes 62-89` — statut : confirmed (1 vote(s)) — catégorie : derivation-gap

Vérifié par redérivation : le terme de Q pour le régime k est sum_n gamma_{n+1}(k) log N(z_{n+1}; Theta^T Zbar_n, Sigma_W,k), dont l'objectif exact est la norme de Mahalanobis pondérée par Sigma_W,k^{-1}. Le papier écrit l'argmin en norme euclidienne ||z_{n+1} - Theta^T Zbar_n||^2 (eq:M_step_Theta). Les deux argmins coïncident parce que la régression est multivariée à covariance commune (le GLS Theta_hat=(Zbar^T W Zbar)^{-1} Zbar^T W Z' ne dépend pas de Sigma) — App D l.77 donne d'ailleurs la bonne équation normale, et Sigma_hat_W (eq:Sigma_W_weighted, l.81-87, avec les NOUVEAUX F_k, b_k) complète le maximiseur joint exact. Le code est conforme (sqrt-w + lstsq, covariance pondérée par Wsum ; semi_supervised.py l.304-316). Tout est correct ; il manque juste la phrase justifiant pourquoi la norme euclidienne suffit.

**Preuve :** 05_estimation.tex l.112-116 (argmin euclidien) vs terme de Q en norme Sigma^{-1} ; App D l.75-78 (équation normale pondérée) ; semi_supervised.py l.309-315.

**Suggestion :** Ajouter : « since the regression is multivariate with a common covariance, the Sigma-weighted and Euclidean weighted least-squares problems share the same minimiser, so plain weighted OLS is the exact M-step ».

**Ajustement de sévérité (vérificateurs) :** Keep at [info] — correctly calibrated. It is a one-sentence expository improvement (the math and code are exact as written), not an error; do not escalate.

### ✅ [INFO] Formulations à polir : « each pair as a single observation » (paires chevauchantes) et « converged blocks » dans le cas supervisé one-shot

`paper/sections/05_estimation.tex:lignes 75-79 et 26-28` — statut : confirmed (1 vote(s)) — catégorie : notation

(1) L.75-79 : « Treating each pair (Z_n, Z_{n+1}) as a single observation conditioned on R_{n+1} » — les paires se chevauchent (Z_n partagé) ; la structure exacte est un HMM à émissions autorégressives (markoviennes conditionnelles) dont la factorisation en chaîne p(z_1|r_1) prod p(z_{n+1}|z_n, r_{n+1}) justifie le forward-backward utilisé. Les récursions elles-mêmes (eq:forward/eq:backward, App D l.37-47, posteriors l.52-58, identité sum_k xi_n(j,k)=gamma_n(j) l.59-60, eq:log_mvn l.19-24) ont été revérifiées : correctes et conformes au code (_forward/_backward/_compute_xi/_log_mvn_batch). (2) L.26-28 : « applying the AB projection ... to the converged (C_k, D_k, Delta_k, Sigma_V,k) blocks » — dans le cas supervisé il n'y a pas d'itération, « converged » est un reliquat du contexte EM ; l'ordre clamp-SPD puis projection AB d'Alg. 1 (l.56-59) correspond bien au code (supervised.py l.249-261).

**Preuve :** 05_estimation.tex l.75-77, l.26-28 ; App D l.37-60 vérifiés ligne à ligne contre semi_supervised.py l.155-212.

**Suggestion :** Remplacer par « an HMM with autoregressive (conditionally Markov) Gaussian emissions » et par « the estimated blocks » dans le cas supervisé.

**Ajustement de sévérité (vérificateurs) :** None — [info] is the right level; this is prose polish with no effect on correctness of derivations or code.

## Annexes A et E

_Audit des annexes A (renversement du temps, 33 lignes) et E (forme de Joseph, 86 lignes) avec redérivation manuelle complète, lecture des passages référents (02_model_h5.tex l.46-52, 03_filtering.tex l.19/206/256-283), vérification croisée avec prg/filter/gss_filter.py (l.404-424) et tests numériques numpy des identités matricielles. Bilan : les deux annexes sont référencées (non orphelines) et leurs RÉSULTATS finaux sont corrects — la collapse (H5) de l'annexe A est vraie (sans besoin de stationnarité, correctement évitée), et l'équivalence short/Joseph au gain optimal de l'annexe E est exacte (vérifiée à 9e-16 près) et fidèlement implémentée dans le code. Deux défauts sérieux de rédaction mathématique : (1) la preuve de l'annexe A attribue sa factorisation rétrograde à « chain rule + (H1) » alors qu'elle requiert (H5bis) à chaque pas, et localise l'usage de (H5) sur le mauvais facteur — telle qu'écrite, la preuve est incomplète voire circulaire selon la lecture ; (2) l'annexe E énonce la forme de Joseph « for any matrix K » avec pour membre de gauche Var[X|·], égalité fausse pour K non optimal (écart numérique ≈9.6 sur un exemple ; l'identité vraie pour tout K porte sur la covariance d'erreur de l'estimateur de gain K). S'y ajoutent des incohérences de conditionnement (blocs de P_{n+1}(k) conditionnés sur y_{1:n} vs gaussiennité marginale invoquée), un renvoi trompeur de la section 2 vers l'annexe A pour l'équivalence H5/H5bis jamais prouvée, et des points mineurs (PSD vs SPD, inversibilité de Σ_XX, citations Joseph, flag Joseph limité au mode h5_exact dans le code). Aucune de ces erreurs ne se propage au filtre : le code n'utilise que le gain optimal, où tout est exact._

### ✅ [HIGH] Preuve A : la factorisation rétrograde n'est PAS obtenue par « chain rule + (H1) » seuls — (H5bis) est utilisé au mauvais endroit

`paper/appendix/A_time_reversal.tex:13-22 (factorisation display l.15-20 et phrase l.21-22)` — statut : confirmed (2 vote(s)) — catégorie : derivation-gap

Le texte (l.13-14) affirme que « Repeated application of the chain rule and (H1) yields » la factorisation affichée. Redérivation : la chaîne (X_n, W_n) est Markov (H1), donc p(x_{1:N}, w_{1:N}) = p(x_N, w_N) ∏_m p(x_m, w_m | x_{m+1}, w_{m+1}). Chaque facteur se scinde en p(w_m | x_{m+1}, w_{m+1}) · p(x_m | w_m, x_{m+1}, w_{m+1}). Pour obtenir le premier crochet affiché p(w_m | w_{m+1}) (sans x_{m+1} dans le conditionnement), il faut p(w_m | x_{m+1}, w_{m+1}) = p(w_m | w_{m+1}), c'est-à-dire exactement (H5bis), appliqué à CHAQUE m = 1..n. La chain rule + (H1) seuls donnent x_{m+1} dans le conditionnement du premier crochet. De plus, la phrase l.21-22 (« The factor p(x_{n+1}|w_{n+1}) at the end of the second bracket is the time-reversed counterpart of (H5) ») est une mauvaise attribution : ce facteur terminal vient de la simple chain rule p(x_N, w_N) = p(w_N) p(x_N|w_N) et ne requiert AUCUNE hypothèse ; (H5bis) sert dans le premier crochet, pas là. Sous une autre lecture (factoriser d'abord p(w_{1:N}) puis p(x_{1:N}|w_{1:N})), réduire p(x_N|w_{1:N}) à p(x_N|w_N) serait circulaire (c'est la conclusion à prouver). Le RÉSULTAT est néanmoins correct : j'ai revérifié toute la dérivation avec (H5bis) inséré aux n endroits requis — la marginalisation télescopique sur x_1..x_n (l.23-25), le quotient par p(w_{1:n+1}) (l.31-33) et la conclusion sont justes.

**Preuve :** l.13-14 : « Repeated application of the chain rule and (H1) yields » ; le premier crochet affiché l.17 est p(w_1|w_2)…p(w_n|w_{n+1}) p(w_{n+1}) alors que H1 seul donne p(w_m | x_{m+1}, w_{m+1}) ; l.21-22 attribue (H5) au facteur p(x_{n+1}|w_{n+1}) qui est pur chain rule. Vérif cas N=2 : p(x_1,x_2,w_1,w_2) = p(w_2)p(x_2|w_2) p(w_1|x_2,w_2) p(x_1|w_1,x_2,w_2) ; la forme affichée exige p(w_1|x_2,w_2)=p(w_1|w_2) = (H5bis).

**Suggestion :** Réécrire : « Repeated application of the chain rule, (H1) and (H5bis) yields », et déplacer la justification : chaque facteur p(w_m|w_{m+1}) du premier crochet utilise p(w_m | x_{m+1}, w_{m+1}) = p(w_m | w_{m+1}), i.e. (H5bis) appliqué au pas m. Supprimer ou corriger la phrase l.21-22 (le facteur terminal est trivial).

**Ajustement de sévérité (vérificateurs) :** Keep high under a proof-correctness lens: the proof of the paper's central enabling lemma, as written, is not valid (insufficient justification of the key display plus a mathematically wrong attribution sentence), and a referee checking it would be blocked or suspect circularity. Downgrade to medium only if the audit reserves high for false results: the theorem itself is true, the filter/code are unaffected, and the fix is a two-line local rewording. | Keep high or downgrade to medium: the proof of the paper's central collapse property (eq:H5_collapse_mean/var) is invalid as written (needed hypothesis never invoked where used, invoked where not needed), which a referee would flag; but the stated result is true and the fix is a local two-line editorial change with no downstream impact on the paper's results or the code.

### ✅ [HIGH] Énoncé faux : « for any matrix K » avec membre de gauche Var[X_{n+1} | r_{n+1}=k, y_{n+1}]

`paper/appendix/E_joseph.tex:53-62 (eq:joseph_proof)` — statut : confirmed (2 vote(s)) — catégorie : math-error

eq:joseph_proof écrit Var[X_{n+1} | r_{n+1}=k, y_{n+1}] = (I−KH) Σ_XX (I−KH)ᵀ + K R Kᵀ en précisant « for any matrix K (not only the optimal Kalman gain) ». C'est mathématiquement faux tel qu'écrit : le membre de gauche est une quantité fixe (= Σ_XX − Σ_XY Σ_YY^{-1} Σ_YX, indépendante de K) alors que le membre de droite dépend de K ; l'égalité ne tient QUE pour le gain optimal K = Σ_XY S^{-1}. La forme de Joseph correcte « pour tout K » a pour membre de gauche la covariance d'erreur de l'estimateur affine x̂ = μ_X + K(y − μ_Y), i.e. E[(X−x̂)(X−x̂)ᵀ | r_{n+1}=k] = Σ_XX − KΣ_YX − Σ_XY Kᵀ + KΣ_YY Kᵀ, qui coïncide avec Var(X|·) seulement au gain optimal. Vérifié numériquement (q=3, s=2, Σ SPD aléatoire) : pour K optimal, |Joseph − short| = 8.9e-16 et |Joseph − Var(X|Y)| = 8.9e-16 ; pour K arbitraire, |Joseph(K) − Var(X|Y)| ≈ 9.59 mais |Joseph(K) − errcov(K)| = 3.6e-15. Impact contenu : le filtre (§3, eq:joseph) et le code n'emploient que le gain optimal, donc aucune propagation d'erreur ; mais l'équation affichée, telle quelle, est fausse — c'est précisément le genre de point qu'un rapporteur relèvera dans un papier dont l'argument central est l'exactitude. La sous-section 3 (l.64-75, équivalence avec la forme courte au gain optimal) est, elle, correcte : redérivée à la main — (I−KH)Σ_XX(I−KH)ᵀ + KRKᵀ = Σ_XX − KΣ_YX − Σ_XY Kᵀ + KΣ_YY Kᵀ via HΣ_XX = Σ_YX, puis = Σ_XX − KSKᵀ pour K = Σ_XY S^{-1}.

**Preuve :** E_joseph.tex l.54-55 : « gives, for any matrix K (not only the optimal Kalman gain) » ; l.56-60 : LHS = \Vark{X_{n+1}}{r_{n+1} = k, y_{n+1}}. Test numpy : K aléatoire ⇒ écart max 9.59 entre Joseph(K) et Var(X|Y) ; égalité (3.6e-15) entre Joseph(K) et la covariance d'erreur de l'estimateur de gain K.

**Suggestion :** Remplacer le LHS par E[(X_{n+1} − x̂)(X_{n+1} − x̂)ᵀ | r_{n+1}=k] avec x̂ := μ_X + K(y_{n+1} − μ_Y) et préciser « which reduces to Var[X_{n+1}|·] when K is the optimal gain » ; ou supprimer « for any matrix K » et énoncer l'identité directement au gain optimal.

**Ajustement de sévérité (vérificateurs) :** Lower from high to medium. The error is mathematically real and sits in a displayed equation of a paper whose selling point is exactness, so a referee could flag it — but it is confined to a parenthetical claim plus a mislabeled LHS in an appendix remark, the equation is true at the gain actually used everywhere (paper and code), nothing propagates, and the fix is a one-line rewording.

### ✅ [MEDIUM] L'équivalence (H5) ⟺ (H5bis) est annoncée comme prouvée dans l'Appendice A, qui ne la prouve pas

`paper/sections/02_model_h5.tex:46-52 (eq:H5bis) renvoyant à app:time_reversal` — statut : confirmed (2 vote(s)) — catégorie : claim-overreach

02_model_h5.tex l.46-47 : « which is equivalent (after time reversal, see Appendix~app:time_reversal) to (eq:H5bis) ». L'Appendice A ne démontre nulle part cette équivalence : il l'UTILISE (« cf. eq:H5bis », l.22) comme ingrédient pour prouver la collapse des moments. L'équivalence est vraie et élémentaire — (H5) dit X_{n+1} ⊥ (R_n,Y_n) | (R_{n+1},Y_{n+1}) et (H5bis) est la même indépendance conditionnelle lue dans l'autre sens (symétrie de l'indépendance conditionnelle / Bayes) — mais elle n'est prouvée nulle part dans le papier et le renvoi est trompeur (lecteur cherchera en vain dans A).

**Preuve :** 02_model_h5.tex l.46-47 vs A_time_reversal.tex : aucune dérivation Bayes p(w_n|w_{n+1},x_{n+1}) = p(w_n|w_{n+1}) ⟺ p(x_{n+1}|w_{n+1},w_n) = p(x_{n+1}|w_{n+1}) n'y figure.

**Suggestion :** Soit ajouter 2 lignes dans l'Appendice A (Bayes : les deux énoncés expriment X_{n+1} ⊥ W_n | W_{n+1}), soit remplacer le renvoi par « (by symmetry of conditional independence) » sans citer l'appendice.

**Ajustement de sévérité (vérificateurs) :** Keep medium (or low-medium). Not a mathematical error — the equivalence is true — but the gap is load-bearing: Appendix A's proof of the collapse relies on (H5bis), which is never derived from the assumed (H5), and the cross-references form a closed loop (Sec. 2 points to App. A, App. A points back to eq:H5bis). Fix is trivial (two lines). | Lower from medium to low-medium. Purely expository: the claimed equivalence is true and trivially provable in two lines, the proposed fixes are correct, and nothing downstream (results, constraint, code) is affected. The one aggravating factor keeping it above pure-typo level is the circular cross-reference (Sec. 2 ↔ Appendix A) inside the proof chain of the central collapse result, plus the related mislabel in 04_constraint.tex:21 (cites eq:H5bis while Appendix B actually argues from eq:H5), which the fix should also touch.

### ✅ [MEDIUM] Incohérence de conditionnement : « Z_{n+1}|r_{n+1}=k jointly Gaussian » vs blocs de P_{n+1}(k) conditionnés sur y_{1:n} (mélange gaussien en mode time-varying)

`paper/appendix/E_joseph.tex:13-19 et 23-30 (eq:Y_given_X) ; cf. 03_filtering.tex l.34-39 (eq:regime_moments) et l.119-140` — statut : confirmed (2 vote(s)) — catégorie : derivation-gap

L'appendice (l.23) justifie eq:Y_given_X par « Since Z_{n+1} | r_{n+1}=k is jointly Gaussian », mais construit H et R à partir des blocs de P_{n+1}(k), que §3 définit comme moments conditionnés sur (r_{n+1}=k, y_{1:n}) (eq:regime_moments l.37 et propagation eq:var_Z, conditionnée sur y_{1:n}). Or en mode time-varying, p(z_{n+1} | r_{n+1}=k, y_{1:n}) est un MÉLANGE gaussien sur r_n (mixture des composantes j pondérées par eq:reverse_P), pas une gaussienne unique : la décomposition conditionnelle Y|X (eq:Y_given_X) avec V ⊥ X et V gaussien n'est alors qu'approchée au sens des deux premiers moments. Le LHS d'eq:joseph_proof omet aussi y_{1:n} (valide seulement via la collapse (H5), eq:H5_collapse_var, non invoquée ici). En mode h5-exact/stationnaire — le seul où le code applique la forme de Joseph — Z|r=k est exactement gaussien (Remark rem:marginal_form, 02 l.84-91) et tout l'argument est exact. Point important : l'équivalence algébrique short/Joseph (sous-section 3) est une identité matricielle valable pour tout Σ symétrique avec Σ_XX inversible, indépendamment de toute gaussiennité — donc l'usage en implémentation est sûr quel que soit le mode ; seul l'habillage probabiliste des sous-sections 1-2 est trop rapide.

**Preuve :** E_joseph.tex l.15-16 : « Let Σ_XX, Σ_XY, Σ_YY be the blocks of P_{n+1}(k) » ; l.23 : conditionnement sur r_{n+1}=k seul ; 03_filtering.tex l.37 : P_n(k) := E[Z_n Z_nᵀ | r_n=k, y_{1:n}] ; l.107-108 : P_{n+1}(k) mélange sur j via p(r_n=j | r_{n+1}=k, y_{1:n}).

**Suggestion :** Soit dériver dans le cadre stationnaire (Z|r=k exactement gaussien, cohérent avec l'usage code), soit reformuler les sous-sections 1-2 comme une construction au sens des deux premiers moments (H, R définis algébriquement depuis Σ) et souligner que l'équivalence de la sous-section 3 est purement matricielle.

**Ajustement de sévérité (vérificateurs) :** Downgrade medium -> low (or low-medium). Every displayed equation in Appendix E is correct under the paper's standing assumptions (H1-H3+H5); the gap is in the probabilistic dressing of sub-sections 1-2 (Gaussianity claimed for a mixture, three inconsistent conditionings, missing invocation of eq:H5_collapse_var). No result, recommendation, or code path is invalidated — unlike the confirmed S_jk inconsistency in section 3, nothing here propagates. Fix is one or two sentences: cite the H5 collapse / rem:marginal_form, or recast H,R as an algebraic two-moment construction and note the sub-section-3 identity is purely matricial. | Downgrade medium -> low (low-medium at most). The final formula eq:joseph is algebraically valid in all modes, the main-text remark defines H and R purely algebraically, and the code only uses the Joseph form in the stationary h5_exact mode where the probabilistic argument is exact. The defect is confined to the justification narrative of appendix E subsections 1-2 plus one uncited use of the H5 collapse; the fix is a few sentences. Medium would only be defensible on the grounds that a paper whose central claim is exactness should not rest a derivation on a false "jointly Gaussian" premise, but nothing downstream is affected.

### ✅ [LOW] Conditionnements différents : la preuve donne p(x_{n+1}|r_{1:n+1}, y_{1:n+1}), les équations cibles conditionnent sur r_{n+1} seul

`paper/appendix/A_time_reversal.tex:7-11 (énoncé) vs sections/03_filtering.tex eq:H5_collapse_mean/var (l.209-216)` — statut : confirmed (1 vote(s)) — catégorie : derivation-gap

L'appendice prouve p(x_{n+1} | w_{1:n+1}) = p(x_{n+1} | w_{n+1}) avec w_m = (r_m, y_m), donc conditionnement sur TOUS les régimes r_{1:n+1}. Les équations eq:H5_collapse_mean/var (03_filtering.tex l.210-215) conditionnent sur (r_{n+1}, y_{1:n+1}) seulement — les r_{1:n} sont cachés. Le passage de l'un à l'autre requiert une marginalisation sur r_{1:n} : p(x_{n+1}|r_{n+1}, y_{1:n+1}) = Σ_{r_{1:n}} p(r_{1:n}|r_{n+1}, y_{1:n+1}) p(x_{n+1}|r_{1:n+1}, y_{1:n+1}) = p(x_{n+1}|r_{n+1}, y_{n+1}) car le terme interne est constant en r_{1:n}. Étape facile mais non mentionnée : « follow immediately by taking conditional expectations » (l.10-11) saute ce pas, et une espérance conditionnelle directe du résultat prouvé donnerait E[X_{n+1}|r_{1:n+1}, y_{1:n+1}], pas E[X_{n+1}|r_{n+1}, y_{1:n+1}].

**Preuve :** A_time_reversal.tex l.8 : p(x_{n+1}|w_{1:n+1}) = p(x_{n+1}|w_{n+1}) ; 03_filtering.tex l.210-211 : E[X_{n+1} | r_{n+1}, y_{1:n+1}] = E[X_{n+1} | r_{n+1}, y_{n+1}].

**Suggestion :** Ajouter une phrase : « marginalising the identity over r_{1:n} given (r_{n+1}, y_{1:n+1}) yields p(x_{n+1}|r_{n+1}, y_{1:n+1}) = p(x_{n+1}|r_{n+1}, y_{n+1}), whence (eq:H5_collapse_mean)-(eq:H5_collapse_var). »

**Ajustement de sévérité (vérificateurs) :** Keep at low. The conclusion of the paper is mathematically correct; only a one-line marginalisation/tower-property justification is missing in the appendix. Fix is a single added sentence.

### ✅ [LOW] Blocs « de P_{n+1}(k) » sans le caveat de centrage ; inversibilité de Σ_XX non énoncée ; « SPD » au lieu de PSD

`paper/appendix/E_joseph.tex:14-16 ; 32-37 (eq:HR_def) ; 79-86` — statut : confirmed (1 vote(s)) — catégorie : notation

Trois points mineurs. (i) l.15-16 : « Let Σ_XX, Σ_XY, Σ_YY be the blocks of P_{n+1}(k) » — P_{n+1}(k) est le moment d'ordre 2 NON centré E[ZZᵀ|·] (03 l.37) ; les blocs utilisés sont ceux de la covariance centrée Σ_{n+1}(k) = P_{n+1}(k) − μμᵀ. §3 signale cet abus (l.137-140 « by slight abuse of notation ») mais l'appendice le répète sans le caveat. (ii) H := Σ_YX Σ_XX^{-1} (l.33, et §3 l.273) suppose Σ_XX inversible — hypothèse jamais énoncée (la forme courte ne requiert que Σ_YY ≻ 0). (iii) l.83-84 : « sum of two positive semi-definite matrices and is therefore symmetric and SPD by construction » — une somme de PSD est PSD, pas SPD (définie positive) ; en outre R = Σ_YY − HΣ_XX Hᵀ calculé en précision finie peut être légèrement indéfini (le code applique d'ailleurs _psd_floor à R, gss_filter.py l.418).

**Preuve :** E_joseph.tex l.15-16, l.33, l.83-84 ; 03_filtering.tex l.124 et l.137-140 ; prg/filter/gss_filter.py l.418 : R = _psd_floor(_sym(...)).

**Suggestion :** (i) ajouter « (centred covariance blocks, cf. Sec. 3) » ; (ii) ajouter « assuming Σ_XX ≻ 0 » ; (iii) remplacer « SPD » par « PSD ».

**Ajustement de sévérité (vérificateurs) :** Keep low — three wording/precision defects confined to an appendix, none affecting the validity of results (the §3 statements are correct); fixes are one-line edits.

### ✅ [INFO] Distribution stationnaire NON requise (correct) ; (H2)-(H3) invoqués mais inutilisés ; wording « collapses into the constant 1 » imprécis

`paper/appendix/A_time_reversal.tex:7 (hypothèses) et 13-33 (preuve)` — statut : confirmed (1 vote(s)) — catégorie : notation

Réponse à la question posée : la preuve n'a PAS besoin de stationnarité ni de la distribution stationnaire — elle emploie les conditionnelles rétrogrades exactes p(w_m|w_{m+1}), p(x_m|w_m,w_{m+1},x_{m+1}) du processus réel, valables même si la chaîne retournée est inhomogène en temps ; aucun théorème de Bayes sur une loi stationnaire n'est nécessaire ni invoqué, et l'appendice ne prétend jamais que les noyaux retournés sont homogènes. C'est correct. Deux points mineurs : (i) la preuve n'utilise que (H1) et (H5/H5bis) ; (H2) (gaussiennité) et (H3) sont annoncés l.7 mais ne servent pas (hypothèses surabondantes, sans danger) ; (ii) l.23-25 « Marginalising … collapses the second bracket into the constant 1 » : le second crochet contient aussi p(x_{n+1}|w_{n+1}) qui survit (il n'est pas intégré) — la multline affichée l.26-30 est correcte, seul le wording est approximatif.

**Preuve :** Aucune occurrence de « stationary » dans A_time_reversal.tex ; les facteurs p(w_m|w_{m+1}) sont des densités conditionnelles génériques, pas des noyaux homogènes. l.7 : « under (H1)--(H3) and (H5) » ; (H2),(H3) n'apparaissent dans aucun pas de la preuve.

**Suggestion :** Optionnel : écrire « under (H1) and (H5) » (ou garder H1-H3+H5 par cohérence avec le papier), et préciser « collapses the second bracket into p(x_{n+1}|w_{n+1}) ».

**Ajustement de sévérité (vérificateurs) :** None — [info] is the right level. This is a positive verification (no stationarity needed, proof correct) plus two cosmetic nits (surplus hypotheses in the statement, one imprecise sentence whose displayed equation is nonetheless correct). The proposed optional fixes are accurate; if anything, prefer the second fix ('collapses the second bracket into p(x_{n+1}|w_{n+1})') over changing l.7, since (H2) implicitly guarantees the densities/second moments used by the notation.

### ✅ [INFO] Cohérence papier/code : formules conformes, mais le flag Joseph n'existe qu'en mode h5_exact (et le défaut du code est la forme courte, pas la « recommandée »)

`paper/appendix/E_joseph.tex:77-86 et 03_filtering.tex l.256-283 (rem:joseph) vs prg/filter/gss_filter.py l.211-224, 404-424` — statut : confirmed (1 vote(s)) — catégorie : claim-overreach

Vérification code : gss_filter.py implémente exactement les formules de l'appendice — H = Σ_YX Σ_XX^{-1} (via _safe_solve(S_XX.T, S_XY).T, l.417), R = Σ_YY − HΣ_XX Hᵀ (l.418), Joseph (I−KH)Σ_XX(I−KH)ᵀ + KRKᵀ (l.420), forme courte Σ_XX − KΣ_YY Kᵀ (l.423) avec K = Σ_XY Σ_YY^{-1} (l.404-406), conforme à eq:HR_def, eq:joseph_proof, eq:kalman_var, eq:gain_k. L'équivalence des deux formes au gain optimal est correcte (redérivée + vérifiée numériquement, écart 8.9e-16). Deux écarts papier/code : (i) la Remark rem:joseph (§3) présente le switch short/Joseph dans le contexte de la mise à jour PAR PAS (eq:kalman_var « re-evaluated at every step », l.252-254 ; coût « per regime per step », App. E l.85-86), alors que le code n'offre le flag joseph qu'en mode h5_exact stationnaire (covariance constante précalculée) et l'ignore avec warning en imm_general (l.218-221) — aucune option Joseph dans le chemin time-varying ; (ii) App. E l.84-85 déclare Joseph « the recommended implementation », mais le défaut du code est joseph=False (l.205).

**Preuve :** gss_filter.py l.205 : « joseph: bool = False » ; l.218-221 : « joseph=True has no effect in mode='imm_general' » ; docstring l.77-80 : « Both forms give the same constant covariance under stationarity » ; 03_filtering.tex l.280-281 : « The implementation switches between the short and Joseph forms via a single boolean flag ».

**Suggestion :** Préciser dans rem:joseph que, dans l'implémentation, la forme de Joseph s'applique au précalcul stationnaire (mode h5-exact) ; aligner « recommended implementation » avec le défaut du code (ou justifier le défaut short).

**Ajustement de sévérité (vérificateurs) :** None — [info] is correctly calibrated: it is a paper/code documentation-coherence issue (overclaiming Remark + recommended-vs-default mismatch), not a mathematical or correctness error in either the paper's derivations or the code.

### ✅ [INFO] Citations de la forme de Joseph à harmoniser : Kalman 1960 ne contient pas la forme de Joseph

`paper/appendix/E_joseph.tex:53-55 (cite anderson_optimal_1979 Sec. 5.3) et 03_filtering.tex l.260 (cite kalman_new_1960, bucy_filtering_1968)` — statut : confirmed (1 vote(s)) — catégorie : claim-overreach

§3 (rem:joseph, l.260) attribue la forme de Joseph à \cite{kalman_new_1960,bucy_filtering_1968} ; l'article de Kalman (1960) ne contient pas la forme de Joseph, due à Bucy & Joseph (1968) — la première citation est donc discutable. L'appendice cite, lui, Anderson & Moore (1979), Sec. 5.3 ; le numéro de section est à vérifier (la discussion de la forme stabilisée y figure plutôt dans le chapitre sur les aspects computationnels selon les éditions). Les deux passages citent des sources différentes pour la même identité.

**Preuve :** 03_filtering.tex l.260 : « Joseph form~\cite{kalman_new_1960,bucy_filtering_1968} » ; E_joseph.tex l.53-54 : « The standard Joseph identity~\cite[Sec.~5.3]{anderson_optimal_1979} » ; paper.bib l.186, 196, 205 (les trois entrées existent).

**Suggestion :** Citer uniformément Bucy & Joseph (1968) (éventuellement + Anderson & Moore avec le bon numéro de chapitre) ; retirer kalman_new_1960 de la citation de la forme de Joseph.

**Ajustement de sévérité (vérificateurs) :** Keep [info] (citation hygiene, no impact on math — the Joseph identity and its proof in the paper are correct). One refinement: the finding hedged on the Anderson & Moore section number, but it is now verified incorrect (Sec. 5.3 = "The Innovations Sequence"; the book never names "Joseph"), so the correctif should also replace or drop \cite[Sec.~5.3]{anderson_optimal_1979}, not just harmonize §3.

### ✅ [INFO] Aucune annexe orpheline : A référencée 3 fois, E référencée 1 fois, toutes deux incluses dans paper.tex

`paper/appendix/A_time_reversal.tex + paper/appendix/E_joseph.tex:Références : 02_model_h5.tex:47, 03_filtering.tex:19, 03_filtering.tex:206 (app:time_reversal) ; 03_filtering.tex:282 (app:joseph)` — statut : confirmed (1 vote(s)) — catégorie : notation

Annexe A (label app:time_reversal) : référencée par 02_model_h5.tex l.47 (équivalence H5/H5bis — renvoi imprécis, cf. finding dédié), 03_filtering.tex l.19 (exactitude de l'IMM) et l.206 (preuve de la collapse eq:H5_collapse_mean/var — usage principal, correct). Annexe E (label app:joseph) : référencée par 03_filtering.tex l.282 (dérivation de eq:joseph dans rem:joseph — usage cohérent). Les deux fichiers sont inclus dans paper.tex (l.56 et l.60). Tous les labels d'équations cités par les annexes (eq:H5_collapse_mean/var, eq:H5bis, eq:joseph, eq:kalman_var, rem:joseph) existent dans les sections principales. Ni A ni E n'est orpheline.

**Preuve :** grep \ref{app:time_reversal} → 3 hits dans sections/ ; \ref{app:joseph} → 1 hit ; paper.tex l.56-60 inclut appendix/A et appendix/E.

**Ajustement de sévérité (vérificateurs) :** none — [info] is appropriate; this is an accurate positive sanity-check (no orphan appendices), not a defect

## Cohérence des notations (tout le papier)

_Audit de cohérence des notations sur l'intégralité du papier (macros.tex, sections 00-08, appendices A-E, diagramme TikZ et 9 tables générées), répertoire /Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper. Méthode : lecture exhaustive, re-dérivation manuelle des identités citées (blocs de F·P·F^T, formes compactes P/Q/R/M, Joseph, BIC), table croisée complète des \label/\ref/\eqref et des clés \cite, comptage d'usage de chaque macro, vérification dimensionnelle des équations des appendices. Bonne santé sur le plan mécanique : aucun \ref/\eqref cassé, aucune clé de citation manquante, comptage BIC d_H5=17 correct. En revanche la couche notationnelle présente des défauts sérieux : le symbole P porte cinq sens (dont deux à sept lignes d'écart en Sec. 6 et deux dans le même appendice B), les macros \Vark/\Covk impriment des moments non centrés là où des covariances centrées sont requises (rendant eq:mix_var, eq:cov_Z_Z et plusieurs équations de l'appendice B littéralement fausses dès que le biais — pourtant « first-class » — est non nul), et la définition des moments de régime (eq:regime_moments) est incompatible avec leurs trois usages (initialisation a priori, écrasement a posteriori, formules de paire qui deviennent dégénérées Σ_YY=0). Le Step 4 de l'appendice B est invalide tel qu'imprimé (identités Q_A^T=Q fausses, équation intermédiaire dimensionnellement incohérente pour q≠s, résultat final néanmoins correct), et l'« équivalence » centrale de l'appendice C ainsi que la condition de nécessité « K·s ≥ q+s » (annoncée en intro mais absente de l'appendice cité) ne sont pas établies. Enfin, la confusion Σ_{YY,n+1}(k) / Σ_{YY,n+1}(k,j) entre Sec. 3 et l'appendice B est la trace notationnelle exacte de l'erreur (S_jk) confirmée numériquement côté code, alors que la section présente la récursion comme exacte sans dérivation de cette équation._

### ✅ [CRITICAL] eq:(S_jk): conditionnements incohérents, présenté comme exact sans dérivation

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/03_filtering.tex:170-187 (eq:S_jk, eq:M_tilde); cf. appendix/B_h5_derivation.tex:62-70` — statut : confirmed (2 vote(s)) — catégorie : math-error

S_{n+1}^{(j,k)} combine la variance marginale Σ_{YY,n+1}(k) (conditionnée à r_{n+1}=k seul, donc mélangée sur r_n) avec des cross-covariances pair-conditionnelles Cov(Y_{n+1},Y_n | r_n=j, r_{n+1}=k). L'appendice B (eq:Omega, l.62-70) montre que l'objet correct du conditionnement gaussien est la variance PAIR-conditionnelle Σ_{YY,n+1}(k,j) — un symbole quasi identique à Σ_{YY,n+1}(k), ce qui est vraisemblablement l'origine de l'erreur. La section présente l'ensemble de la récursion comme exacte (l.14-22 « this IMM is exact », l.43-46 « no Gaussian-collapse approximation is made at any stage ») et eq:S_jk est asserté sans dérivation ni renvoi : c'est donc une erreur de dérivation du papier, pas une approximation assumée. Cohérent avec la divergence numérique confirmée du mode imm_general.

**Preuve :** l.170-174: « S_{n+1}^{(j,k)} = \Sigma_{YY,n+1}(k) - \Covk{Y_{n+1}}{Y_n}{j, k, \yn}\,\Sigma_{YY,n}(j)^{-1} \times\Covk{Y_n}{Y_{n+1}}{j, k, \yn} » vs B l.70: « \Sigma_{YY,n+1}(k,j) := \Vark{Y_{n+1}}{r_n=j, r_{n+1}=k} »

**Suggestion :** Remplacer Σ_{YY,n+1}(k) par le bloc YY de F_k Σ_n(j) F_k^T + Σ_W(k) (variance pair-conditionnelle), et harmoniser le symbole avec Σ_{YY,n+1}(k,j) de l'appendice B (correctif déjà validé côté code).

### ✅ [HIGH] Moments de régime (μ_n(k), P_n(k)): trois conventions de conditionnement incompatibles

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/03_filtering.tex:35-38, 166-183, 285-302, 361-363` — statut : confirmed (2 vote(s)) — catégorie : notation

eq:regime_moments (l.35-38) définit μ_n(k) := E[Z_n | r_n=k, y_{1:n}] et P_n(k) := E[Z_n Z_n^T | r_n=k, y_{1:n}]. Comme Y_n est observé dans y_{1:n}, cette définition implique μ_{Y,n}(j) = y_n et Σ_{YY,n}(j) = 0 : le terme correctif de eq:y_pred_jk (y_n − μ_{Y,n}(j)) s'annule identiquement et Σ_{YY,n}(j)^{-1} dans eq:M_tilde/eq:S_jk est l'inverse d'une matrice nulle. Or l'initialisation (l.361-363) fournit des moments A PRIORI non conditionnés (μ_1(k)=μ_{z0}(k)), tandis que rem:posterior_moments (l.285-302) impose l'écrasement par les moments A POSTERIORI. Les formules du step (II) ne sont cohérentes qu'avec des moments PRÉDITS E[Z_n | r_n=j, y_{1:n-1}], jamais définis.

**Preuve :** l.37: « P_n(k) := \Ek{Z_n Z_n\tp}{r_n = k, \yn} » ; l.168: « \widetilde M_{n+1}^{(j,k)}\,(y_n - \mu_{Y,n}(j)) » — nul si μ_n(j) est conditionné à y_{1:n} ; l.181: « \Sigma_{YY,n}(j)^{-1} » — singulier sous la même convention.

**Suggestion :** Introduire explicitement deux jeux de notations (moments prédits μ_{n|n-1}(k) vs a posteriori μ_{n|n}(k)) et réécrire les steps (I)-(II) avec le bon jeu à chaque endroit.

### ✅ [HIGH] Symbole P surchargé avec cinq sens différents (idem R, et trois notations pour Σ_W)

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/04_constraint.tex:04:29; 03:87,156; 05:9,25; 06:18,25; B:16-22,110` — statut : confirmed (2 vote(s)) — catégorie : notation

P désigne : (1) la matrice de transition P_{j,k} (03 l.87,156 ; 05 l.9,25 ; 06 l.25) ; (2) la matrice auxiliaire P := Δ^T C^T + Σ_V D^T (04 l.29) ; (3) la covariance de bruit P(j) (App B l.16-22) ; (4) le second moment de régime P_n(k) (03 l.37) ; (5) la covariance filtrée P_{n|n}^{(k)} (03 l.32). En 06, les sens (2) et (1) cohabitent à 7 lignes d'écart (l.18 « PM^{-1}W » vs l.25 « P = [0.97 ...] ») ; dans l'App B, P(j) côtoie « (= R = P^T) » l.110 qui renvoie au P auxiliaire. De même R = CΔ + DΣ_V (04 l.34) collisionne avec le processus R_n et avec R_{n+1}^{(k)} (bruit effectif de Joseph, 03 l.275, E l.35). Enfin la covariance jointe du bruit a trois notations : Σ_{W,k} (05/06/D), P(j) (App B), Σ(r) (04 l.56-58, App C).

**Preuve :** 06 l.18: « \|\Delta^T A^T + \Sigma_V B^T - PM^{-1}W\|_F » puis l.25: « P = \begin{bmatrix}0.97 & 0.03 \\ 0.02 & 0.98\end{bmatrix} » ; B l.110: « = R = P^T » huit lignes sous la définition « P(j) := \Vark{W_n}{r_n = j} ».

**Suggestion :** Renommer les auxiliaires de 04 (p.ex. P→Π_1 ou N_1, R→P^T explicite), unifier la covariance bruit sous Σ_W(r) partout, et réserver P aux probabilités de transition.

**Ajustement de sévérité (vérificateurs) :** high → medium

### ✅ [HIGH] \Vark/\Covk impriment des moments NON centrés mais sont utilisés comme covariances centrées

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/macros.tex:macros:23-26; 02:152-157; 03:32,103-105; B:62-70,119-127; E:57` — statut : confirmed (2 vote(s)) — catégorie : notation

macros.tex définit \Vark{A}{c} → E[A A^T | c] et \Covk{A}{B}{c} → E[A B^T | c] (moments non centrés). Plusieurs équations les emploient comme covariances centrées, devenant littéralement fausses telles qu'imprimées : (i) eq:mix_var (02 l.152-157) est la loi de variance totale, valide pour Var mais pas pour E[XX^T|·] ; (ii) P_{n|n}^{(k)} := \Vark{X_n}{r_n=k,\yn} (03 l.32) doit être la covariance de Kalman alors que P_n(k) trois lignes plus bas est bien le moment non centré ; (iii) eq:cov_Z_Z (03 l.103-105) : LHS = E[Z_{n+1}Z_n^T|·] mais RHS = F_k(P_n(j)−μμ^T), le membre centré — faux dès que b_k≠0 ; (iv) Ω_{jk} (B l.62-70) et eq:CXY_n1 (B l.119-127) : avec biais, E[X_{n+1}Y_{n+1}^T] ≠ Cov, alors que le papier traite le biais en « first-class » ; (v) eq:joseph_proof (E l.57). À l'inverse, 03 l.337 centre explicitement (\Vark{Z_1 − μ_{z0}(k)}{r_1=k}), prouvant que la macro est bien non centrée.

**Preuve :** macros l.26: « \newcommand{\Vark}[2]{\Ek{#1\,#1\tp}{#2}} » ; 03 l.103-105: « \Covk{Z_{n+1}}{Z_n}{r_n = j, r_{n+1} = k, \yn} = F_k\,(P_n(j) - \mu_n(j)\mu_n(j)\tp) » — or E[Z_{n+1}Z_n^T|·] = F_k P_n(j) + b_k μ_n(j)^T.

**Suggestion :** Ajouter des macros Cov/Var centrées (p.ex. \CCovk → Cov(A,B|c)) et les utiliser partout où l'objet est une covariance ; réserver E[··^T] aux seconds moments P_n(k).

**Ajustement de sévérité (vérificateurs) :** Keep high within the notation-consistency dimension: multiple equations in 02, 03 and E are false as literally typeset. Note however that the appendix-B sub-claims should be dropped (covered by the declared zero-mean convention at B Step 1), and no downstream result or code is actually wrong — if the dimension weighted only substantive impact, medium would also be defensible.

### ✅ [HIGH] Step 4 de la dérivation H5 : justifications fausses et équation intermédiaire dimensionnellement invalide

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/appendix/B_h5_derivation.tex:136-152 (Step 4)` — statut : confirmed (2 vote(s)) — catégorie : derivation-gap

(a) l.145-146 affirme « Q_A C^T + Q_B D^T + Σ_V is exactly [F·P(r)·F^T + Σ_W]_{YY} = M » — contradiction directe avec l.139 où T = Q_A C^T + Q_B D^T + Δ (c'est le bloc XY, pas YY ; le bloc YY est QC^T + RD^T + Σ_V). (b) l.151-152 : « Q_A^T = CΣ_U + DΔ^T = Q » et « Q_B^T = R » sont faux : Q_A^T = Σ_U A^T + ΔB^T et Q_B^T = Δ^T A^T + Σ_V B^T (le membre de gauche de la contrainte !). (c) L'équation non numérotée l.147-150 « T M^{-1} R = Q A^T M^{-1} R + ... » est dimensionnellement invalide pour q≠s (QA^T est s×q, inmultipliable par M^{-1} s×s). Le résultat final eq:h5_compact_app est néanmoins correct : il s'obtient en transposant eq:h5_first (T^T = QA^T + RB^T + Δ^T, R^T = P, M symétrique).

**Preuve :** l.151: « where we used \(Q_A\tp = C\Sigma_U + D\Delta\tp = Q\) (\(Q_A = Q^T\)) » — or Q_A := AΣ_U + BΔ^T (l.140), donc Q_A^T = Σ_U A^T + ΔB^T ≠ Q^T sauf si A=C, B=D.

**Suggestion :** Remplacer tout le Step 4 par : transposer eq:h5_first (AΔ + BΣ_V = T M^{-1} R) en utilisant T^T = QA^T + RB^T + Δ^T, R^T = P et M = M^T, ce qui donne directement eq:h5_compact_app.

**Ajustement de sévérité (vérificateurs) :** none — « high » est correctement calibrée

### ✅ [HIGH] « Équivalence » eq:H5_in_X non établie + condition de nécessité « K·s ≥ q+s » fantôme et revendications incohérentes

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/appendix/C_projections.tex:C:31-39; 01:84-85; 04:67-68; abstract:33-34` — statut : confirmed (2 vote(s)) — catégorie : claim-overreach

(a) C l.31-39 : eq:H5_in_X (X(r) = H^T M^{-1}(HΣX + Δ^T), (q+s)×q équations) est annoncée « equivalent » à eq:H5_compact (s×q équations) via « multiplying on the left by M(r) and re-arranging » — la forme compacte n'est que la projection de H5_in_X sur la ligne-bloc [Δ^T, Σ_V] ; H5_in_X est strictement plus forte, et l'argument de nécessité de l'appendice repose sur ce passage non justifié. (b) L'intro (01 l.84-85) annonce « generically necessary when K·s ≥ q+s ... (Appendix C) » : cette condition n'apparaît nulle part dans l'appendice C (ni ailleurs), et on ne voit pas pourquoi K (nombre de régimes) interviendrait dans une contrainte par-régime. (c) Quatre formulations incompatibles : abstract l.33-34 « sufficient — and generically necessary » (sans condition) ; 04 l.67-68 « the unique closed-form solution » ; 08 l.17-19 « generically necessary in all configurations considered ». De plus le quantificateur « for all Σ(r) ≻ 0 » (C l.45-53, 04 l.62-65) traite Σ(r) comme libre alors que c'est un paramètre fixé du modèle, ce que le texte ne discute pas.

**Preuve :** 01 l.84: « generically necessary when \(K \cdot s \geq q + s\) ... (Appendix~\ref{app:AB_constraint}) » — grep de « q + s » dans C_projections.tex : uniquement des dimensions de matrices, aucune condition de nécessité.

**Suggestion :** Soit prouver l'équivalence (ou affaiblir en « sufficient condition »), soit énoncer et démontrer dans C la condition exacte de nécessité générique ; aligner abstract/01/04/08 sur la même formulation.

**Ajustement de sévérité (vérificateurs) :** none — high est correctement calibré (pas critical : la suffisance est correctement démontrée et le filtre exact n'en dépend pas ; pas medium : la nécessité générique est revendiquée en abstract/intro/conclusion avec une preuve cassée et une condition fantôme)

### ✅ [MEDIUM] Le problème de filtrage est défini avec y_{1:N} (lissage) au lieu de y_{1:n}

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/02_model_h5.tex:144` — statut : confirmed (2 vote(s)) — catégorie : typo-math

« The objective is to compute, for every n = 1,…,N, the posterior conditional mean \hat X_n := \Ek{X_n}{\yN} » — \yN = y_{1:N} définit le lisseur, pas le filtre. Tout le reste du paragraphe (eq:mix_mean/mix_var, propagation de p(r_{n+1}|y_{1:n+1})) et tout le papier sont en filtrage. C'est l'unique occurrence de la macro \yN dans le corps du papier, ce qui confirme la coquille (\yn attendu).

**Preuve :** l.144: « \(\hat X_n := \Ek{X_n}{\yN}\) » ; grep : \yN utilisé 1 fois dans tout le papier, \yn 21 fois, \ynp 14 fois.

**Suggestion :** Remplacer \yN par \yn.

**Ajustement de sévérité (vérificateurs) :** medium est defendable (coquille dans l'enonce formel du probleme central); low serait aussi acceptable car le contexte rend l'intention evidente et le correctif est trivial. Garder medium.

### ✅ [MEDIUM] \yn[1:n+1] s'imprime « y_{1:n}[1:n+1] »

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/03_filtering.tex:301` — statut : confirmed (2 vote(s)) — catégorie : typo-math

La macro \yn ne prend pas d'argument ; « the full history \(\yn[1:n+1]\) » produit le rendu cassé « y_{1:n}[1 : n+1] » dans le PDF.

**Preuve :** l.301: « conditions on the full history \(\yn[1:n+1]\). »

**Suggestion :** Remplacer par \ynp (= y_{1:n+1}).

**Ajustement de sévérité (vérificateurs) :** low | low

### ✅ [MEDIUM] Résidu H5 « PM^{-1}W » : W jamais défini, et W surchargé (bruit / poids / résidu)

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/06_experiments.tex:17-18; appendix/D_baum_welch.tex:75-82` — statut : confirmed (2 vote(s)) — catégorie : notation

06 l.18 : le résidu H5 est ‖Δ^TA^T + Σ_VB^T − PM^{-1}W‖_F où W n'est défini nulle part — d'après eq:H5_compact ce devrait être (QA^T + RB^T + Δ^T). Par ailleurs W désigne le bruit W_{n+1} (02 l.60-76) et W_k la matrice de poids diag(γ_2(k),…,γ_N(k)) (D l.75-77), cette dernière à cinq lignes de Σ_{W,k} (D l.82) qui utilise W dans son troisième sens.

**Preuve :** 06 l.18: « \|\Delta^T A^T + \Sigma_V B^T - PM^{-1}W\|_F < 10^{-10} » — aucun « W := » dans 04, 06 ni C.

**Suggestion :** Écrire le résidu en toutes lettres ‖Δ^TA^T + Σ_VB^T − PM^{-1}(QA^T + RB^T + Δ^T)‖_F ; renommer la matrice de poids (p.ex. G_k ou D_γ(k)).

**Ajustement de sévérité (vérificateurs) :** medium confirmée (garder medium ; le desc surestime légèrement l'ambiguïté — l'équation désambiguïse les rôles — mais la collision de symboles non annoncée dans la section du test H5 central justifie medium plutôt que low)

### ✅ [MEDIUM] b^{(k)} = coefficient de régression sur Y_n en Sec. 7, mais = vecteur de biais en Sec. 6 ; et la même quantité est appelée b^{(k)} puis B^{(k)}/B(k)

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/07_real_data.tex:82-94; cf. 06:42,106-108` — statut : confirmed (2 vote(s)) — catégorie : notation

La régression du test F (07 l.82) s'écrit « X_{n+1} = a^{(k)} X_n + b^{(k)} Y_n + c^{(k)} + ε_n » : b^{(k)} y est l'analogue du bloc B et c^{(k)} le biais. Or en 06 (l.42, 106-108) b^{(k)} est le vecteur de biais ([0.10, 0.05]^T etc.). Pire, dans la même sous-section, le texte (l.92-94 « the unconstrained estimate of B^{(k)} ») et la table tab_enso_h5_test (colonne « B(k) ») désignent ce même coefficient par B majuscule. Un lecteur ne peut pas savoir si le test porte sur le biais ou sur le bloc B.

**Preuve :** 07 l.82: « b^{(k)} Y_n + c^{(k)} » vs 06 l.42: « b^{(1)} = [0.10,\, 0.05]^T » (biais) vs 07 l.92: « estimate of \(B^{(k)}\) ».

**Suggestion :** Écrire la régression avec les symboles du modèle : X_{n+1} = A^{(k)} X_n + B^{(k)} Y_n + b_X^{(k)} + ε_n.

**Ajustement de sévérité (vérificateurs) :** low

### ❓ [MEDIUM] App E : Σ_XX, Σ_XY, Σ_YY « blocks of P_{n+1}(k) » contredit la convention centrée de Sec. 3

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/appendix/E_joseph.tex:13-19; cf. 03:126-140` — statut : uncertain (2 vote(s)) — catégorie : notation

03 l.137-140 précise que les blocs Σ désignent, « by slight abuse of notation », les blocs de la covariance CENTRÉE Σ_{n+1}(k) = P_{n+1}(k) − μ_{n+1}(k)μ_{n+1}(k)^T. L'appendice E (l.15-16) déclare au contraire « Let Σ_XX, Σ_XY, Σ_YY be the blocks of P_{n+1}(k) » : pris au pied de la lettre, S = Σ_YY serait un moment non centré et le gain K = Σ_XY Σ_YY^{-1} faux. Accessoirement K (gain, E l.19,55) et K (nombre de régimes, partout) cohabitent sans indice.

**Preuve :** E l.15-16: « \(\Sigma_{XX}, \Sigma_{XY}, \Sigma_{YY}\) be the blocks of \(P_{n+1}(k)\) » vs 03 l.137-139: « the \(\Sigma\)-blocks refer ... to the corresponding blocks of the \emph{centred} covariance \(\Sigma_{n+1}(k)\) ».

**Suggestion :** Écrire « blocks of the centred covariance Σ_{n+1}(k) = P_{n+1}(k) − μμ^T » dans E, comme en Sec. 3.

**Ajustement de sévérité (vérificateurs) :** n/a (isReal=false) — au pire trivial/polish, pas medium

### ❓ [MEDIUM] « M ≻ 0 whenever Σ_U, Σ_V ≻ 0 » est faux ; il faut la positivité JOINTE

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/04_constraint.tex:38-40` — statut : uncertain (2 vote(s)) — catégorie : math-error

M = [C D] Σ(r) [C D]^T + Σ_V où Σ(r) est la covariance jointe. Contre-exemple scalaire : Σ_U = Σ_V = 1, Δ = −10, C = D = 1 donne M = 1+1+1−20 = −17 < 0 alors que Σ_U, Σ_V ≻ 0. La condition correcte est Σ(r) ⪰ 0 (l'« hypothèse de positivité physique » que le papier invoque par ailleurs) plus Σ_V ≻ 0, qui donne M ⪰ Σ_V ≻ 0.

**Preuve :** l.38-40: « \(M = M^T \succ 0\) whenever \(\Sigma_U, \Sigma_V \succ 0\) ».

**Suggestion :** Remplacer par : « M = [C D]Σ(r)[C D]^T + Σ_V ≻ 0 dès que Σ(r) ⪰ 0 et Σ_V ≻ 0 ».

**Ajustement de sévérité (vérificateurs) :** n/a (isReal=false) ; si conservé malgré tout comme nit de rédaction, sévérité low au maximum

### ✅ [MEDIUM] Intercept de eq:mean_X_linear écrit b_{X,k} (faux si b_{Y,k}≠0) alors que la remarque invoque un « intercept α » inexistant

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/appendix/B_h5_derivation.tex:51-58 et 165-173` — statut : confirmed (2 vote(s)) — catégorie : notation

eq:mean_X_linear (l.52-53) écrit E[X_{n+1}|j,k,y_n,y_{n+1}] = b_{X,k} + β_1 y_n + β_2 y_{n+1}. Sous la convention loi-bruit (Z_n centré), E[X_{n+1}] = b_{X,k} mais E[Y_{n+1}] = b_{Y,k} ≠ 0, donc l'intercept vaut b_{X,k} − β_2 b_{Y,k} ≠ b_{X,k}. La remarque finale (l.168) parle de « the intercept \alpha » — symbole qui n'apparaît dans aucune équation de l'appendice. Texte et équation se contredisent ; l'argument d'invariance au biais reste correct (β_1, β_2 ne dépendent que des covariances) mais tel qu'imprimé l'intercept est faux.

**Preuve :** l.53: « = b_{X,k} + \beta_1\,y_n + \beta_2\,y_{n+1} » vs l.168: « it enters the mean in \eqref{eq:mean_X_linear} through the intercept \(\alpha\) ».

**Suggestion :** Écrire l'intercept générique α dans eq:mean_X_linear (avec α = b_{X,k} − β_2 b_{Y,k} si besoin) et garder la remarque telle quelle.

**Ajustement de sévérité (vérificateurs) :** none — medium is correctly calibrated

### ✅ [MEDIUM] Σ_{YY,n+1}(k,j) : ordre des arguments inversé par rapport à toutes les autres quantités de paire

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/appendix/B_h5_derivation.tex:62-70` — statut : confirmed (2 vote(s)) — catégorie : notation

Toutes les quantités de paire du papier s'indexent (source j, cible k) : P_{j,k}, Λ_{n+1}(j,k), S_{n+1}^{(j,k)}, Γ_{jk}, ŷ^{(j,k)}, ξ_n(j,k). Seul Σ_{YY,n+1}(k,j) inverse l'ordre alors que sa définition conditionne sur (r_n=j, r_{n+1}=k). De plus le symbole recycle la notation temporelle de filtre Σ_{YY,n+1}(k) (03 l.236) pour un objet stationnaire pair-conditionnel — c'est précisément la paire de symboles confondue dans eq:S_jk (voir finding sur (S_jk)).

**Preuve :** l.70: « \(\Sigma_{YY,n+1}(k,j) := \Vark{Y_{n+1}}{r_n=j, r_{n+1}=k}\) » — ordre (k,j) vs conditionnement (j,k).

**Suggestion :** Noter Σ_{YY}^{(j,k)} (ou (j,k) partout) et distinguer typographiquement la quantité stationnaire de la quantité de filtre.

### ✅ [MEDIUM] « Sufficient and necessary to enforce ... in the special case j = k » : la suffisance n'est pas démontrée

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/appendix/B_h5_derivation.tex:100-103` — statut : confirmed (2 vote(s)) — catégorie : derivation-gap

Le passage du cas j=k au cas général (tous j) est asserté (« Indeed » suivi du seul calcul j=k). La nécessité est triviale (j=k est une instance), mais la suffisance — l'équation (eq:h5_beta1) au seul point j=k implique toutes les paires (j,k) — n'est établie nulle part dans l'appendice ; elle est vraie sous la paramétrisation AB (T_{jk}M_{jk}^{-1} = Δ_kΣ_{V,k}^{-1} indépendant de j) mais c'est précisément ce qui est en cours de dérivation. Tel quel le « sufficient » est circulaire.

**Preuve :** l.100-102: « For this to hold for \emph{all} source regimes \(j\), it is sufficient and necessary to enforce~\eqref{eq:h5_beta1} in the special case \(j = k\) ».

**Suggestion :** Soit ne déduire que la nécessité ici et prouver la suffisance pour tout (j,k) après obtention de AB (le calcul T_{jk}M_{jk}^{-1} = Δ_kΣ_{V,k}^{-1} tient en trois lignes), soit renvoyer à la Prop. 2/App. C.

### ✅ [MEDIUM] X(r), H(r), Z(r) : trois symboles de l'appendice C qui collisionnent avec l'état X_n, l'opérateur H de Joseph et l'état augmenté Z_n

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/appendix/C_projections.tex:13-20, 48` — statut : confirmed (2 vote(s)) — catégorie : notation

L'appendice C définit X(r) := [A(r)^T; B(r)^T] (matrice de paramètres), H(r) := [C(r), D(r)] et Z(r) ∈ R^{s×q}, alors que dans tout le papier X_n est l'état caché, Z_n = [X_n^T, Y_n^T]^T l'état augmenté, et H^{(k)} := Σ_YX Σ_XX^{-1} l'opérateur d'observation effectif (03 l.273, App E) — un objet différent jouant pourtant un rôle analogue d'« observation ». Le titre de sous-section « Reformulation as a linear identity in X(r) » est particulièrement trompeur.

**Preuve :** C l.15: « X(r) := \begin{bmatrix} A(r)^T \\ B(r)^T \end{bmatrix} » ; l.18: « H(r) := [\,C(r),\; D(r)\,] » ; l.48: « \(Z(r) \in \mathbb{R}^{s \times q}\) ».

**Suggestion :** Utiliser des lettres calligraphiques ou grecques (p.ex. 𝒳(r), ℋ(r), Ζ→Ψ(r)) pour les objets purement algébriques de l'appendice.

**Ajustement de sévérité (vérificateurs) :** low | low

### ✅ [MEDIUM] Le test F de B=0 est présenté comme test de (H5), mais rejeter B=0 ne rejette pas (H5)

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/07_real_data.tex:75-77, 98-100` — statut : confirmed (2 vote(s)) — catégorie : claim-overreach

Le texte qualifie le test de « simple necessary check » et conclut sur le S&P500 que « the same test rejects (H5) at p < 10^{-4} ». Or (H5) n'impose pas B=0 mais B = ΔΣ_V^{-1}D (eq:AB) : B=0 n'est qu'un sous-cas (« the strongest form »). Ne pas rejeter B=0 est bien compatible avec (H5), mais rejeter B=0 ne rejette nullement (H5) — la phrase sur les données financières surinterprète le test. Le test cohérent serait B − Δ̂Σ̂_V^{-1}D̂ = 0.

**Preuve :** l.75-76: « A simple necessary check is whether \(B^{(k)}\) is statistically distinguishable from zero » ; l.99-100: « where the same test rejects (H5) at \(p < 10^{-4}\) for the dominant regime ».

**Suggestion :** Reformuler : le test rejette le sous-modèle B=0, pas (H5) ; ou tester directement le résidu AB.

### ✅ [MEDIUM] Désaccords texte ↔ tables générées (valeurs et colonnes citées absentes)

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/06_experiments.tex:06:199,216,228,280-282,350-352; 07:191` — statut : confirmed (2 vote(s)) — catégorie : notation

(1) 06 l.228 : « machine precision (≈ 10^{-17}) » vs tab_supervised_M1 : « 10^{-18} ». (2) 06 l.199 : « +5.7% above 1 for IMM » vs NEES = 1.050 dans tab_filter_M2M3 (= +5.0%). (3) 06 l.216 : protocole N_train ∈ {200, 500, 1000, 2000} mais la table ne montre que {200, 500, 2000}. (4) 06 l.280-282 : « ε̂_b = 0.2189 in both columns » — aucune colonne ε̂_b dans tab_supervised_M1 ; symbole ε̂_b jamais défini. (5) 06 l.350-352 : « ‖F̂−F‖_F/‖F‖_F = 0.06 vs 0.14 » attribué au contexte de tab:em_basin qui ne contient que basin rate et n_iter. (6) 07 l.191 : « accuracy reaches 53% » vs tab_enso_em : V1 = 0.544, V0 = 0.522 — 53% ne correspond à aucune ligne.

**Preuve :** tab_supervised_M1: « AB constraint & ... & $10^{-18}$ » vs 06 l.227-228: « (\approx 10^{-17}) » ; tab_filter_M2M3: « M2 ... 1.050 » vs 06 l.199: « (+5.7\%) ».

**Suggestion :** Régénérer texte et tables depuis la même source de résultats ; ajouter les colonnes citées (ε̂_b, erreurs paramètres) ou supprimer les renvois.

**Ajustement de sévérité (vérificateurs) :** medium (inchangée) — agrégat justifié de 5 désaccords quantitatifs avérés + 1 point présentationnel plus faible (item 0.06/0.14, valeurs non attribuées explicitement à la table)

### ✅ [MEDIUM] rem:h5exact : S^{(k)} (indexé k) annoncé « matching » Γ_{jk} (indexé paire) ; π_0 vs π_∞ ; π_0 libre ou dérivée ?

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/03_filtering.tex:48-64; cf. 06:44, 03:335, 06:376-378` — statut : confirmed (2 vote(s)) — catégorie : notation

(1) l.56-59 : « the Kalman gain K^{(k)} and the innovation covariance S^{(k)} are also constant, matching the explicit expressions Γ_{jk} and μ_{jk}(·) » — une quantité indexée par k seul ne peut coïncider avec une quantité de paire : Γ_{jk} dépend de j via Σ_{U,j}, Δ_j, Σ_{V,j} (eq:Gamma_jk). (2) La distribution stationnaire des régimes est notée π_0(k) en 03 l.335 (« the stationary distribution of the transition kernel ») et π_∞(k) en 06 l.44. (3) Incohérence de statut : 03 déclare π_0 dérivée de P, mais 05 (l.105) l'estime librement (π̂_0 = γ_1) et la formule BIC (06 l.376-378) lui compte K−1 paramètres libres.

**Preuve :** 03 l.57-59: « matching the explicit expressions \(\Gamma_{jk}\) and \(\mu_{jk}(\cdot)\) of Proposition~\ref{prop:markov} » ; 06 l.44: « \(\pi_\infty(1) \approx 0.40\) » vs 03 l.335: « \(\pi_0(k) := \pu{r_1 = k}\) (the stationary distribution...) ».

**Suggestion :** Préciser dans rem:h5exact la relation exacte (p.ex. S^{(k)} = Σ_j p(j|k)Γ_{jk} + spread, ou la condition d'égalité) ; unifier π_∞ ; trancher le statut de π_0 (libre vs stationnaire) et corriger d_H5 en conséquence.

### ✅ [LOW] Trois conventions d'indexation de régime (A_r / A(k) / A^{(k)}) + alphabet de régimes {1..K} vs {0,1,2} + indexation Python

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/01_introduction.tex:abstract:30-31; 01:126-128; 02:65-67; 04:56-58; 06:28; 07:34-38; 05:146,187` — statut : confirmed (1 vote(s)) — catégorie : notation

Le paragraphe Notation (01 l.126-128) annonce l'indice : A_r, Σ_{V,r} — convention suivie en 02-05. Mais l'abstract (l.30-31) et 04 l.56-58 utilisent la forme fonctionnelle A(k), Σ_V(k), Δ(r) ; 06 (l.28 et suiv.) et 07 passent aux exposants A^{(k)}, C^{(k)}, b^{(k)}, qui collisionnent avec les exposants de mode du filtre (S^{(k)}, K^{(k)}, x̂^{(k)}). Par ailleurs 07 l.34-38 prend R_n ∈ {0,1,2} alors que 02 l.13 pose Ω = {1,…,K} ; et 05 l.146/187 trie les régimes par « A_k[0,0] » (indexation logicielle base 0) au lieu de (A_k)_{11}.

**Preuve :** 01 l.127: « (e.g.\ \(A_r\), \(\Sigma_{V,r}\)) » vs 06 l.28: « (superscripts \((1)\) and \((2)\)) » vs abstract l.30: « \(A(k) = \Delta(k)\,\Sigma_V(k)^{-1}\,C(k)\) » ; 07 l.34: « \(R_n \in \{0, 1, 2\}\) ».

**Suggestion :** Tout ramener à la convention indice annoncée (A_k), renuméroter les régimes ENSO 1..3, écrire (A_k)_{11}.

**Ajustement de sévérité (vérificateurs) :** none — « low » est approprié

### ✅ [LOW] Macros mortes et macros contournées (mêmes quantités tapées à la main)

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/macros.tex:macros:17-80; 05:49,113,183; 07:211; D:75; 03:275` — statut : confirmed (1 vote(s)) — catégorie : notation

Jamais utilisées : \hlmath, \Eu, \VarInvk, \EMj, \rnn, \inv (les inverses sont tapés ^{-1} partout), \tr. Définies mais contournées : \argmax/\argmin (DeclareMathOperator avec limites correctes) vs \arg\min_Θ / \arg\max tapés à la main en 05 l.49,113,183 et 07 l.211 ; \diag vs \mathrm{diag} en D l.75. Transposition hétérogène : \tp (= ^{T}) en 02/03/E mais ^T manuel dans 04 (16×), 05 (6×), C (27×), D (3×), plus la forme hybride « H_{n+1}^{(k),T} » en 03 l.275 — rendu visuellement incohérent (^{T} vs ^T).

**Preuve :** grep : \Eu 0, \VarInvk 0, \EMj 0, \rnn 0, \inv 0, \argmax/\argmin 0 dans le corps ; 05 l.113: « \arg\min_\Theta » ; C : 27 occurrences de ^T manuel.

**Suggestion :** Supprimer les macros mortes, faire un sed global ^T→\tp et \arg\min→\argmin avant soumission.

### ✅ [LOW] Flottants jamais référencés et symbole de figure jamais défini

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/04_constraint.tex:04:100-136; 07:117; figures/recursion_diagram.tex:31` — statut : confirmed (1 vote(s)) — catégorie : notation

(1) Table tab:h5_special_cases (04 l.100-136) : aucun \ref dans le texte (le paragraphe l.94-98 ne la cite pas) — IEEE exige que tout flottant soit appelé. (2) Table tab:enso_filter (07, \input l.117) : jamais référencée par \ref ; le texte enchaîne directement sur les valeurs. (3) La boîte (II) du diagramme de récursion utilise ν^{(j,k)} alors que le texte ne définit que ν^{(k)} (03 l.232) — l'innovation de paire n'est jamais nommée.

**Preuve :** grep « \ref{tab:h5_special_cases} » et « \ref{tab:enso_filter} » : 0 occurrence hors \label ; recursion_diagram.tex l.31: « \(\nu^{(j,k)},\ S^{(j,k)},... \)».

**Suggestion :** Ajouter « Table~\ref{...} summarises... » aux deux endroits ; définir ν^{(j,k)} := y_{n+1} − ŷ_{n+1|n}^{(j,k)} en 03 ou utiliser ŷ^{(j,k)} dans la figure.

**Ajustement de sévérité (vérificateurs) :** none — low est approprié

### ✅ [LOW] Divers : remarque tautologique, « Set N = n+1 » inutilisé, U/V jamais définis, |L_Σ|_diag non défini, bornes de sommes implicites

`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/paper/sections/02_model_h5.tex:02:84-91; A:13; B:119-127; D:19-26; 05:103-106; D:83` — statut : confirmed (1 vote(s)) — catégorie : notation

(1) rem:marginal_form (02 l.84-91) : l'équation affichée p(x_n,r_n,y_n) = p(r_n)p(x_n,y_n|r_n) est la chain rule, vraie sans aucune hypothèse — le contenu annoncé (indépendance en n des marginales) n'est pas dans la formule. (2) App A l.13 : « Set N = n+1 » recycle N (longueur de trajectoire) et n'est plus jamais utilisé dans la preuve. (3) App B l.121 : U_{n+1}, V_{n+1} (sous-blocs X/Y du bruit W) utilisés sans avoir jamais été introduits — 02 ne définit que W_{n+1} et ses blocs de covariance. (4) App D l.21 : notation 2log|L_Σ|_{diag} non définie (somme des logs des éléments diagonaux). (5) Bornes de sommes omises : eq:M_step_P (05 l.103, Σ_n sans bornes — numérateur et dénominateur courent sur n=1..N−1) et dénominateur de eq:Sigma_W_weighted (D l.83). (6) eq:sqrt_w (D l.69-71) définit des quantités tildées jamais réutilisées (la solution est donnée via W_k).

**Preuve :** 02 l.88: « \(\pu{x_n, r_n, y_n} = \pu{r_n}\,\pk{x_n, y_n}{r_n}\) » (tautologie) ; B l.121: « \Covk{(AX_n+BY_n+U_{n+1})}{(CX_n+DY_n+V_{n+1})}{r,r} » sans définition préalable de U, V.

**Suggestion :** Écrire la stationnarité comme p(x_n,y_n|r_n=k) = N(m_k, S_k) indépendant de n ; supprimer « Set N = n+1 » ; définir W = [U^T, V^T]^T en 02 ; expliciter log det Σ = 2Σ_i log(L_Σ)_{ii} et les bornes n=1..N−1.

---
_Généré automatiquement ; raisonnements complets des vérificateurs dans `raw/03-paper-math-extracted.json`._