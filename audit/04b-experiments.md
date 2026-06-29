# Vague 4b — Chiffres / résultats du papier ↔ code & données

Workflow `audit-4b-find` (run `wf_397fc904-434`) + vérif `verify-batch` (runs `wf_1ed49426-fda`, `wf_03c227fc-29b`). 3 finders + vérification adversariale (2 lentilles pour high/medium, 1 pour low/info). Détails : `raw/04b-extracted.json`.

**Bilan : 18 trouvailles — 14 confirmées** (0 critical, 6 high, 6 medium, 1 low, 1 info), 1 incertaines, 3 réfutées.

## Trouvailles majeures (critical + high confirmées)

- **[HIGH] Stale legacy results/enso/e2_table.tex contradicts the published ENSO filter table** — `results/enso/e2_table.tex:5-8`
- **[HIGH] Stale results/enso/e3_table.tex uses an obsolete EM-variant design (tau=A/B) absent from the paper** — `results/enso/e3_table.tex:5-9`
- **[HIGH] ENSO filter table calls the test column "joint test log-likelihood" but it is the Y-only predictive density** — `prg/experiments/run_real_data.py:385 (emit_e2_tex caption) → rendered in paper/figures/generated/tab_enso_filter.tex line 2, included by paper/sections/07_real_data.tex:117`
- **[HIGH] Hamilton MS-AR MSE-on-X uses smoothed (acausal) regime probabilities — peeks at future test data** — `scripts/baselines/hamilton_msar.py:114 (res_full.predict()) ; metric used at 115; reported via scripts/e3_add_hamilton.py:97-101 into results/e3/table3.tex`
- **[HIGH] Regime ground-truth labels are derived from a CENTERED (non-causal) smooth of the observation Y — undocumented look-ahead leakage** — `data/real/enso_sst.csv (build_enso_csv.py:99-107) + paper/sections/07_real_data.tex:28,35-38,197-198:build: 99-107; paper: 28,197-198`
- **[HIGH] Committed results/enso/ contains stale, contradictory E3 artifacts produced by the S&P/VIX script, not by run_real_data.py** — `results/enso/e3_summary.json + results/enso/e3_table.tex:e3_summary.json:1-86; e3_table.tex:1-15`

## Tableaux/chiffres §6-7 ↔ fichiers générés

_Compared every printed numeric in paper/sections/06_experiments.tex and 07_real_data.tex against (a) the eight \input-ed generated tables in paper/figures/generated/*.tex, (b) the source CSVs in data/experiments/ and the ENSO data in data/real/enso_sst.csv, (c) the JSON/CSV/tex artifacts in results/{e1,e2,e3,enso}, and (d) the generators fill_placeholders.py, make_figures.py, run_real_data.py. Recomputed M1/M2/M3 filter benchmark (RMSE/NEES/LB/CPU), the CPU ratio (1.372x ~ '1.4x'), BIC K=1..4 (1770.8/1846.8/1938.1/2044.5), supervised rel-F errors at N=200/500/2000, EM basin rates and n_iter, the ENSO H5-test, the ENSO filter and EM tables, ENSO descriptive stats (248/423/243 counts, 27/46/27%, 105 transitions, 732 train / 182 test), and the 0.04 nats/obs penalty — ALL of these match the paper and their generated tables; they are NOT reported as findings. I excluded the previously-acquired items (the ~10^-17 vs 10^-18 residual, NEES +5.7% vs 1.050, N_train {1000} omission, eps_b=0.219, '53%' vs 0.544/0.522, IMM-approx labeling, BIC plug-in dof, EM protocol 50/5 vs 100/10, LB min-p convention, M2 P matrix, H5 tolerance). NEW findings are confined to stale/orphaned reproducibility artifacts in results/enso (e2_table.tex, e3_table.tex, e3_summary.json) that contradict the canonical JSONs and the paper, plus the orphan tab_bic.tex carrying unfilled \ph{XX\%} placeholders, and one minor average-run-length rounding (8.7 vs 8.62). The paper's compiled text and \input-ed tables are themselves numerically self-consistent. Absolute paths: /Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/{paper/sections/06_experiments.tex, paper/sections/07_real_data.tex, paper/figures/generated/tab_bic.tex, results/enso/e2_table.tex, results/enso/e3_table.tex, results/enso/e3_summary.json}._

### ✅ [HIGH] Stale legacy results/enso/e2_table.tex contradicts the published ENSO filter table

`results/enso/e2_table.tex:5-8` — statut : confirmed (2 vote(s)) — catégorie : stale-artifact-contradiction

results/enso/e2_table.tex is a leftover from a superseded pipeline run (mtime May 2 17:02) and reports ENSO filter numbers that flatly contradict both the canonical JSON (results/enso/e2_table.json, May 7 05:50) and the table actually compiled into the paper (paper/figures/generated/tab_enso_filter.tex). The current run_real_data.py no longer writes results/enso/e2_table.tex (it writes e2_table.json + paper/figures/generated/tab_enso_filter.tex, see run_real_data.py:499-503), so this file is orphaned and misleading for reproducibility. The H5-exact log-lik / NLL / MSE and the IMM(H5-fit) row differ entirely.

**Preuve :** results/enso/e2_table.tex:5 'H5-exact (H5 fit) & -20.56 & +0.1130 & 0.6875'; line 7 'IMM-general (H5 fit) & -31.55 & +0.1733 & 0.8065'. CANONICAL results/enso/e2_table.json: h5_exact_h5fit log_lik=-30.0877, nll=0.16532, mse=0.64367; imm_general_h5fit log_lik=-37.392, nll=0.20545, mse=0.64344. PAPER tab_enso_filter.tex: 'H5-exact (H5 fit) & -30.09 & +0.1653 & 0.6437' and 'IMM-approx (H5 fit) & -37.39 & +0.2055 & 0.6434'. The .tex stale file is internally inconsistent with the paper.

**Suggestion :** Delete results/enso/e2_table.tex (and e3_table.tex, e3_summary.json below) or regenerate them, since the live pipeline no longer produces them; keep only the JSON + paper/figures/generated/*.tex that match the paper. The paper itself is correct.

**Ajustement de sévérité (vérificateurs) :** high -> medium. The contradiction is real and verifiable, but the file is a fully orphaned, non-compiled artifact with zero impact on any published claim or the PDF (paper numbers are correct). This is a reproducibility/repo-hygiene issue, not a correctness defect in the paper, so 'high' overstates it; 'medium' is the defensible calibration. | high -> low

### ✅ [HIGH] Stale results/enso/e3_table.tex uses an obsolete EM-variant design (tau=A/B) absent from the paper

`results/enso/e3_table.tex:5-9` — statut : confirmed (2 vote(s)) — catégorie : stale-artifact-contradiction

results/enso/e3_table.tex (mtime May 2 17:03) encodes a superseded experimental design with variants V0/V1 post-hoc tau=B / V2 post-hoc tau=A-dagger / V3 GEM tau=B, where V2 is failed/'---'. The published paper §7.4 and the canonical results/enso/e3_table.json + paper/figures/generated/tab_enso_em.tex use a different, current design: V0 unconstrained / V1 post-hoc AB / V2 GEM AB. The stale file would mislead a reader reproducing the experiment, and its rows do not correspond to any row in the paper.

**Preuve :** results/enso/e3_table.tex rows: 'V1 post-hoc $\tau=B$ & -437.4 & +0.1872 & 0.4089 ...', 'V2 post-hoc $\tau=A^\dagger$ & -437.4 & ---', 'V3 GEM $\tau=B$ & -441.4 ...'. PAPER tab_enso_em.tex (from results/enso/e3_table.json, May 7 05:50): 'V0 unconstrained & -437.4 & +0.1770 ...', 'V1 post-hoc AB & -437.4 & +0.1319 & 1.0703 ...', 'V2 GEM AB & -1482.1 & +0.1790 & 0.6486 ...'. The variant labels and values are mutually exclusive.

**Suggestion :** Remove or regenerate results/enso/e3_table.tex to match the current 3-variant (V0/V1/V2 AB) design used by run_real_data.py:emit_e3_tex and the paper.

**Ajustement de sévérité (vérificateurs) :** high -> medium | high -> medium

### ✅ [MEDIUM] Stale results/enso/e3_summary.json holds an old 4-variant EM run that disagrees with the canonical e3_table.json

`results/enso/e3_summary.json:1-60` — statut : confirmed (2 vote(s)) — catégorie : stale-artifact-contradiction

results/enso/e3_summary.json (mtime May 2 17:00) is a leftover from the obsolete EM design (variants V0_unconstrained, V1_posthoc_B, V2_posthoc_A, V3_GEM_B). It is not produced by the current run_real_data.py (which writes e3_table.json) and its numbers contradict the paper. In particular it contains a numerically blown-up entry (test MSE 3.8e+234) for V2_posthoc_A that no longer exists in the current pipeline.

**Preuve :** results/enso/e3_summary.json: variant 'V2_posthoc_A' test_LL=-58430.95, mse=3.8367e+234; variant 'V3_GEM_B' train_LL=-441.38. CANONICAL results/enso/e3_table.json has only V0_unconstrained (train_LL=-437.45), V1_posthoc_AB (train_LL=-437.45, test_MSE=1.0703), V2_GEM_AB (train_LL=-1482.06). No V2_posthoc_A / V3_GEM_B in current design.

**Suggestion :** Delete results/enso/e3_summary.json; it is a dead artifact from a prior design and does not feed the paper.

**Ajustement de sévérité (vérificateurs) :** Slightly over-rated. The file is a fully orphaned artifact: no code writes it and no LaTeX/section reads it, so it has zero impact on the compiled paper's correctness or build. The finding itself acknowledges this. As a pure reproducibility/repo-hygiene issue with no path into the paper, low is better calibrated than medium. Recommend downgrade medium -> low. | medium -> low

### ✅ [MEDIUM] Generated tab_bic.tex still contains unfilled \ph{XX\%} placeholders (orphan, never \input)

`paper/figures/generated/tab_bic.tex:1-4` — statut : confirmed (2 vote(s)) — catégorie : unfilled-placeholder

make_figures.py:make_tab_bic deliberately emits the selection-% column as the literal placeholder \ph{XX\%} (make_figures.py:334-339), because 'proper selection needs EM per K'. The resulting paper/figures/generated/tab_bic.tex contains four \ph{XX\%} entries that fill_placeholders.py does NOT resolve (it only fills \ph{...} inside 06_experiments.tex, not inside generated table files). The file is currently an orphan: it is the only generated *.tex that is never \input by any paper section, so it does not break compilation. But if anyone wires it in, the \ph macro (paper/macros.tex:86) renders a yellow highlighted '[XX%]' box in the PDF. The §6.6 BIC narrative instead only prints the K=2 value 1846.8 inline.

**Preuve :** paper/figures/generated/tab_bic.tex:1-4 each line ends '& \ph{XX\%} \\'. make_figures.py:338 emits r"\ph{XX\%}". No \input{figures/generated/tab_bic.tex} anywhere in paper/ (grep returns none). paper/macros.tex:86 \newcommand{\ph}[1]{\colorbox{yellow!40}{\small\textbf{[#1]}}}.

**Suggestion :** Either remove tab_bic.tex from the generated set, or have make_tab_bic write '---' / omit the selection column instead of \ph{XX\%}, so a stray \input cannot inject a visible placeholder into the PDF. The numeric BIC values (1770.8/1846.8/1938.1/2044.5) are themselves correct and reproducible from mc_results.csv.

**Ajustement de sévérité (vérificateurs) :** medium -> low | medium -> low

### ✅ [LOW] ENSO average run length stated as 8.7 months; data give 8.62 (914/106 runs)

`paper/sections/07_real_data.tex:41` — statut : confirmed (1 vote(s)) — catégorie : rounding-mismatch

The text reports an average run length of approximately 8.7 months alongside 105 regime transitions. With 914 months and 105 transitions there are 106 runs, giving 914/106 = 8.62 months, which rounds to 8.6, not 8.7. The 8.7 figure corresponds to dividing by the transition count (914/105 = 8.70) rather than by the number of runs (transitions+1). Minor, but the displayed value is off by one in the denominator.

**Preuve :** 07_real_data.tex:39-41 '105 regime transitions (average run length \approx 8.7 months)'. data/real/enso_sst.csv: 914 rows, 105 transitions, 106 runs => 914/106 = 8.62. Computed directly from the regime column.

**Suggestion :** Change 8.7 to 8.6 (= n_months / n_runs), or clarify the definition; the 105-transition count itself is correct.

**Ajustement de sévérité (vérificateurs) :** high -> low

## Baselines & équité des comparaisons

_Audited the comparison baselines and fairness for the §6/§7 experiments. Read scripts/baselines/hamilton_msar.py, scripts/baselines/kalman_single.py, scripts/e2_filter_comparison.py, scripts/e3_add_hamilton.py, scripts/e3_bw_em.py, scripts/e1_supervised_h5.py, scripts/labels.py, prg/experiments/run_real_data.py, prg/experiments/metrics.py, prg/learning/supervised.py, prg/filter/gss_filter.py (step/log_lik/init semantics), and paper/sections/06_experiments.tex + 07_real_data.tex, cross-checked against results/e1, results/e2, results/e3, results/enso JSON/tex and the installed statsmodels source. KEY TRACEABILITY RESULT (frames all severities): the Hamilton MS-AR baseline and the S&P500 filter (table2) / BW-EM (table3) comparisons exist in code but appear NOWHERE in the compiled paper — the paper's only S&P500 touchpoint is one prose sentence (§7:99-100) that IS correctly supported by e1_supervised_h5.py. The paper's actual baselines are ENSO-only: H5-exact, IMM-approx, Kalman K=1. NEW findings (not the listed acquis): (1) HIGH — the published ENSO filter table caption calls the test column 'joint test log-likelihood' but the filter log_lik is the Y-only predictive density log p(y_n|y_{1:n-1}), not joint (X,Y); X is never scored. (2) HIGH — Hamilton's MSE-on-X uses statsmodels predict() which defaults to SMOOTHED (acausal) regime probabilities, leaking future test data, vs the strictly causal GSS/Kalman filtered estimates (code-only, not in paper). (3) MEDIUM — Hamilton wrapper hardcodes converged=True and has a dead/misleadingly-documented 'seed' parameter. (4) MEDIUM — Hamilton's X-likelihood is rendered in the same NLL/obs column as the GSS/Kalman Y-likelihood (incommensurable). (5) INFO — scope/traceability of the above. (6) LOW — ENSO E2 supervised-vs-unsupervised param asymmetry, but it is conservative (disfavors the proposed method) and disclosed. The Kalman K=1 baseline itself is a correct full K=1 GSS reduction and is accurately described; the §6 M1/M2/M3 H5-exact vs IMM-approx comparison uses identical true params (fair). Did not re-report acquired items (RMSE/q normalization, IMM-approx=imm_general naming, BIC plug-in, EM protocol mismatch, LB min-p convention, M2 P absent, H5 tolerance)._

### ✅ [HIGH] ENSO filter table calls the test column "joint test log-likelihood" but it is the Y-only predictive density

`prg/experiments/run_real_data.py:385 (emit_e2_tex caption) → rendered in paper/figures/generated/tab_enso_filter.tex line 2, included by paper/sections/07_real_data.tex:117` — statut : confirmed (2 vote(s)) — catégorie : Mislabelled metric / unfair-looking comparison

The Table tab:enso_filter caption (generated by run_real_data.py:385) states: "$\log\hat L$ is the joint test log-likelihood". But the per-step log_lik returned by GSSFilter is log p(y_n | y_{1:n-1}) — the predictive density of Y (Niño 3.4) ONLY, with X integrated/never scored. This is explicitly documented in prg/filter/gss_filter.py:138 ('Incremental log-likelihood log p(y_n | y_{1:n-1})') and confirmed in _update_step_h5 (gss_filter.py:643, comment 'log p(y_{n+1} | y_{1:n})'). The Kalman K=1 baseline scores the same Y-only density (kalman_single.py:207-210), so the FILTER-vs-FILTER comparison is fair, but the column LABEL is wrong: X (Niño 1+2) contributes nothing to the reported log-lik. Note the EM TRAIN log-lik IS genuinely joint p(Z)=p(X,Y) (semi_supervised.py:19,175), so the confusion is between the joint EM train objective and the Y-marginal filter test score — the caption conflates them. This is distinct from the already-acted RMSE-normalization finding.

**Preuve :** prg/experiments/run_real_data.py:385: r" the OLS-fit is unconstrained. $\log\hat L$ is the joint test" / 386: r" log-likelihood ..."; vs prg/filter/gss_filter.py:138 'Incremental log-likelihood log p(y_n | y_{1:n-1})'; tab_enso_filter.tex line 2 in the compiled paper.

**Suggestion :** Change the caption to '$\log\hat L$ is the test (one-step) predictive log-likelihood of the observation $Y$ (Niño 3.4)'. Drop 'joint'. Optionally clarify that $X$ is not part of the scored likelihood (only the EM training objective is joint).

**Ajustement de sévérité (vérificateurs) :** high -> medium | high -> medium

### ✅ [HIGH] Hamilton MS-AR MSE-on-X uses smoothed (acausal) regime probabilities — peeks at future test data

`scripts/baselines/hamilton_msar.py:114 (res_full.predict()) ; metric used at 115; reported via scripts/e3_add_hamilton.py:97-101 into results/e3/table3.tex` — statut : confirmed (2 vote(s)) — catégorie : Baseline correctness / unfair information leak

HamiltonMSAR.predict_test computes the X prediction with `fitted_by_regime = np.asarray(res_full.predict())` (hamilton_msar.py:114) and reports `mse_x = SSE/N` over the test slice. In statsmodels, `MarkovSwitchingResults.predict()` defaults to `probabilities='smoothed'` (verified in the installed statsmodels markov_switching.py:1941 docstring 'Default is smoothed' and the body at lines 720-723: when probabilities is None it calls self.smooth(...) and uses smoothed_joint_probabilities). The smoother (Kim smoother) uses the ENTIRE future of the series to weight regimes, so Hamilton's per-step X prediction is acausal and benefits from look-ahead. By contrast every other method in the comparison (GSSFilter E_x is the causal filtered mean E[X_n|y_{1:n}], and SingleKalmanFilter likewise) is strictly causal. This makes Hamilton's MSE-on-X artificially low / non-comparable. Note: the regime classification uses pi_filt (filtered, causal) correctly (hamilton_msar.py:117), so only the MSE metric is leaked. This baseline feeds results/e3/table3.tex, NOT the compiled paper (see scope note), so impact on the published claims is currently nil — but if table3 is ever surfaced it would be an unfair comparison.

**Preuve :** hamilton_msar.py:114 `fitted_by_regime = np.asarray(res_full.predict())`; statsmodels markov_switching.py predict() docstring 'Default is smoothed' and body `if probabilities is None or probabilities == 'smoothed': results = self.smooth(...); probabilities = results.smoothed_joint_probabilities`.

**Suggestion :** Pass `probabilities='filtered'` (or 'predicted' for one-step-ahead) to res_full.predict() so the X prediction is causal and commensurable with the GSS/Kalman filtered estimates.

**Ajustement de sévérité (vérificateurs) :** Downgrade high -> low | Downgrade high -> medium. The bug is real and reproducible, but its own scope note is correct: results/e3/table3.tex is not \input anywhere in paper/ (grep confirms zero references), so no published claim is affected. Additionally the leak only makes Hamilton's MSE-on-X artificially LOW, yet the rendered value (1.1675) is still worse than the causal V0 baseline (0.9310), so even the unfair advantage does not flip any comparison. A defect in a non-published auxiliary baseline script is medium, not high, for a paper-correctness audit.

### ✅ [MEDIUM] Hamilton baseline hardcodes converged=True and silently ignores its `seed` argument

`scripts/e3_add_hamilton.py:104 ("converged": True) ; fit at scripts/baselines/hamilton_msar.py:48-63` — statut : confirmed (2 vote(s)) — catégorie : Convergence not verified / dead parameter

Two issues in the Hamilton baseline fitting: (1) e3_add_hamilton.py:104 sets `"converged": True` unconditionally; the wrapper never inspects `self._res.mle_retvals['converged']` from the BFGS optimizer. A non-converged BFGS fit would be reported as converged. (2) HamiltonMSAR.fit has a `seed: int = 42` parameter and a comment 'Deterministic optimisation: we pass a seed via em_algorithm init' (hamilton_msar.py:61), but the seed is never used anywhere — `mod.fit(em_iter=max_iter, disp=False)` passes no seed and `search_reps` defaults to 0 (no random start search). The fit is deterministic because start_params defaults to the model's fixed start_params, so the comment is misleading and the `seed` parameter is dead. Also note `em_iter=max_iter` (=500) only sets the number of EM warm-up steps to improve starting params; the actual optimizer is BFGS with maxiter=100 (statsmodels fit() defaults), not 500 EM iterations as one might infer.

**Preuve :** e3_add_hamilton.py:104 `"converged": True,`; hamilton_msar.py:48 `def fit(self, x_train, max_iter=500, seed=42)`, :61 comment, :62 `self._res = mod.fit(em_iter=max_iter, disp=False)` (seed unused); statsmodels fit signature `def fit(..., method='bfgs', maxiter=100, ..., em_iter=5, search_reps=0, ...)`.

**Suggestion :** Read convergence from self._res.mle_retvals.get('converged') and propagate it; remove the unused `seed` parameter and the misleading comment, or actually use search_reps with a seeded RNG if randomized restarts were intended.

**Ajustement de sévérité (vérificateurs) :** high -> low | medium -> low: both defects are real, but they live in scripts/baselines + scripts/e3_add_hamilton.py, which feed only results/e3/table3.tex — a table not \input by any paper section (verified by grep over paper/). No compiled/published claim is affected. The hardcoded converged=True is a latent correctness risk (could hide an optimizer failure) and the dead seed parameter is cosmetic, so low fits an audit scoped to the paper's integrity.

### ✅ [MEDIUM] Hamilton X-likelihood placed in same 'test NLL/obs' column as the GSS/Kalman Y-likelihood

`scripts/e3_add_hamilton.py:131-141 (table3.tex emitter) ; results/e3/table3.tex` — statut : confirmed (2 vote(s)) — catégorie : Incommensurable metric placed in shared column

In results/e3/table3.tex the Hamilton_MSAR row prints test NLL/obs = +1.2986 in the same column as the GSS variants' values (−0.146 … +0.360). But Hamilton models X directly (endog=x_train, hamilton_msar.py:51-58) so its llf_obs is log p(x_t|x_{1:t-1}) (a density of X), whereas V0–V3 report log p(y_t|y_{1:t-1}) (a density of Y). These are densities of DIFFERENT random variables on DIFFERENTLY-standardized scales and are not comparable; the positive vs negative sign difference makes Hamilton look strictly worse on a metric it should never be compared on. The code's own docstring (hamilton_msar.py:22-24) and the table comment (table3.tex line 2-3 / e3_add_hamilton.py:122-124) acknowledge non-comparability, yet the value is still rendered in the shared column inviting the invalid comparison. The train_log_lik is correctly blanked to 'n/a', but the test column is not. Scope: this table is NOT in the compiled paper, so no published claim is affected today.

**Preuve :** e3_add_hamilton.py:96 `"test_nll_per_obs": float(scores["nll_per_obs"])` placed at :138 in the same f-string column as V0-V3; hamilton_msar.py:121 `"nll_per_obs": -ll_test / N_test` where ll_test is on X; table3.tex line 'Hamilton_MSAR  & n/a & +1.2986 & ...'.

**Suggestion :** Render Hamilton's test NLL/obs as 'n/a' (like its train log-lik) or move it to a separate note, keeping only the genuinely comparable metrics (MSE on X — once the smoothing leak is fixed — and acc/ARI vs labels).

**Ajustement de sévérité (vérificateurs) :** medium -> low | medium -> low

### ❌ [LOW] ENSO E2 gives GSS filters supervised (label-using) params but Kalman K=1 unsupervised params; documented and conservative

`prg/experiments/run_real_data.py:185-217 (run_e2) vs 208 (kalman from full train)` — statut : refuted (1 vote(s)) — catégorie : Training-information asymmetry (conservative, not advantageous to paper method)

In run_e2 the GSS filters use fit_supervised(rs_tr, xs_tr, ys_tr) i.e. parameters estimated WITH the true regime labels R (supervised OLS per regime), while SingleKalmanFilter.from_regressed(xs_tr, ys_tr) (kalman_single.py:98) uses only (X,Y) with NO regime information. This is an information asymmetry, but it FAVORS the GSS filters (more info), and the paper still reports K=1 winning on both MSE and NLL/obs (07_real_data.tex:130-135). So the asymmetry does not artificially advantage the proposed method — if anything it makes the regime-aware GSS filters look worse, and the text honestly attributes K=1's win to 'modest regime heterogeneity'. Also note the n=1 prior treatment differs (GSS h5_exact uses stationary moments, gss_filter.py:554; Kalman uses sample mean/cov of training Z, kalman_single.py:137-144), a minor inconsistency over N=182 steps. Reported as info only; comparison is fair-to-conservative.

**Preuve :** run_real_data.py:190 fit_supervised(rs_tr,...) for GSS vs :208 SingleKalmanFilter.from_regressed(xs_tr, ys_tr) (no rs); 07_real_data.tex:130-135 acknowledges K=1 wins; kalman_single.py:137-144 sample-moment prior vs gss_filter.py:554 stationary prior.

**Suggestion :** No change needed for fairness, but a one-line caption note that K=1 is unsupervised (no regime labels) while the GSS fits are supervised would make the comparison fully transparent.

**Ajustement de sévérité (vérificateurs) :** N/A — trouvaille non retenue. Si on voulait la conserver, elle ne dépasse pas 'info' (suggestion purement optionnelle d'une note de légende), pas 'low'.

### ✅ [INFO] Hamilton MS-AR, S&P500 filter table (table2) and S&P500 BW-EM table (table3) exist in code but appear nowhere in the paper

`paper/sections/07_real_data.tex:99-100` — statut : confirmed (1 vote(s)) — catégorie : Scope / text-vs-code traceability

The task asks to audit the Hamilton MS-AR baseline and S&P500 baselines, but a traceability check shows none of scripts/baselines/hamilton_msar.py, scripts/e2_filter_comparison.py (table2.tex), or scripts/e3_bw_em.py + e3_add_hamilton.py (table3.tex) produce any table or figure that is \input or referenced in the compiled paper. The only S&P500 touchpoint in the paper is the single prose sentence in 07_real_data.tex:99-100 ('the same test rejects (H5) at p < 10^-4 for the dominant regime'), which IS correctly supported by scripts/e1_supervised_h5.py (results/e1: regime 0 under L1 has p=2.07e-31, n=1888 transitions, the larger/'calm' regime). So the published baselines are exclusively the ENSO ones from run_real_data.py: H5-exact, IMM-approx, Kalman K=1. The Hamilton-related findings above therefore do not affect any current paper claim; they are code-quality/fairness issues that would only matter if those tables are surfaced. Flagging so the severity of the Hamilton findings is read in context.

**Preuve :** grep of \input{figures/generated...} in paper/sections shows only tab_filter_M1/M2M3, tab_supervised_M1, tab_em_restarts, tab_em_basin (§6) and tab_enso_h5_test, tab_enso_filter, tab_enso_em (§7); 'Hamilton'/'MSAR'/'table2'/'table3' absent from all paper/*.tex; e1_supervised_h5.py:118 + results/e1 JSON p_value=2.07e-31 for regime 0 (n=1888).

**Suggestion :** Either remove the unused S&P500/Hamilton scripts to avoid an impression of cherry-picking, or, if they are intended as supplementary, add an explicit appendix/footnote pointing to them and fix the fairness issues above first. At minimum keep the §7:100 sentence (it is supported).

**Ajustement de sévérité (vérificateurs) :** none — severity "info" is appropriate; the finding is a correct, accurately-scoped traceability/context note and does not overstate impact (it explicitly states the Hamilton findings affect no current paper claim).

## Reproductibilité & données réelles (§7)

_Audited reproducibility and data construction for §7 (paper/sections/07_real_data.tex), focusing on ENSO (the only dataset whose numbers appear in §7) and the S&P500/VIX comparison referenced at l.99-100. Files read: paper/sections/07_real_data.tex; scripts/fetch_sp500_vix.py, build_enso_csv.py, labels.py, e1_supervised_h5.py, e3_bw_em.py (head); prg/experiments/run_real_data.py, make_figures_real.py; data/real/* and results/enso/*. I ran the actual pipeline against the committed data using the repo's .venv (pandas 3.0.2). POSITIVE / REPRODUCIBLE: (1) E1, E2, E3 ENSO numbers reproduce EXACTLY from the committed enso_sst.csv (B-norms 0.0505/0.0460/0.0261, p 0.359/0.372/0.664; E2 logL -30.09/-32.38/-37.39; E3 acc 0.522/0.544/0.478 — all matching the paper's generated tables); E3 is deterministic at seed 42. (2) enso_sst.csv rebuilds bit-identical from the committed NOAA txt files (max|diff|=0 on all columns). (3) Full-record stats (914 months, 248/423/243, 105 transitions, 8.7-month runs) match the paper. (4) Standardization correctly uses train-only mean/std (no leak) in both load_enso and make_figures_real. (5) ENSO monthly series is continuous (no hidden gaps); S&P/VIX has no >7-day gaps. (6) The §7 'p<10^-4' S&P/VIX claim reproduces (p=2.07e-31, F=141 for the dominant L1 calm regime). NEW ISSUES FOUND (see findings): a high-severity undocumented look-ahead in the ENSO regime labels (ONI is a CENTERED 3-mo mean of the observation Y, confirmed corr 0.978 centered vs 0.950 trailing; ~22% of labels change under a causal proxy), which also biases the E1 F-test selection; stale/contradictory committed E3 artifacts in results/enso/ that actually came from the S&P/VIX e3_bw_em.py scheme (V2_posthoc_A mse=3.8e234); the S&P/VIX results backing §7 are gitignored/uncommitted and the regime_trace.csv needed for the figure is gitignored; the '46%' Neutral baseline is the full-record fraction while accuracy is on the test segment (47.8%); yfinance/FRED fetch is non-deterministic with no provenance pinning; and a weakly-supported La-Nina-peak qualitative claim. I deliberately excluded the already-acted items (B=0 test, '53%' vs 0.544/0.522, EM protocol 50/5, etc.)._

### ✅ [HIGH] Regime ground-truth labels are derived from a CENTERED (non-causal) smooth of the observation Y — undocumented look-ahead leakage

`data/real/enso_sst.csv (build_enso_csv.py:99-107) + paper/sections/07_real_data.tex:28,35-38,197-198:build: 99-107; paper: 28,197-198` — statut : confirmed (2 vote(s)) — catégorie : data-construction / look-ahead bias

The regime label R_n is thresholded from ONI (build_enso_csv.py line 105-107: df.loc[df['oni']<-0.5]=0, >0.5=2). ONI is NOAA's 3-month running mean of Nino-3.4, which is the observation Y_n (run_real_data.py X=Nino1+2, Y=Nino3.4). Crucially NOAA's ONI is a 3-month mean CENTERED on the middle month, so R_n depends on Y_{n-1}, Y_n AND Y_{n+1}. I confirmed this empirically: ONI correlates 0.978 with the centered 3-mo mean of the committed nino34 vs only 0.950 with a trailing mean. Consequences: (1) the causal H5/IMM filters in E2/E3 are graded for 'accuracy' against a ground-truth label that peeks one month into the future of the observation; (2) the E1 Fisher test (run_real_data.py:105-124) conditions the per-regime OLS on mask = rs[1:]==k, i.e. selects rows by R_{n+1}, a function of future Y — selection bias on the very regressor Y_n being tested. The paper says ONI is 'a 3-month running mean' (l.28) and obliquely that regime boundaries are 'a smoothed version of one of the observables' (l.197-198) but never flags the centered/non-causal nature or the leakage it induces.

**Preuve :** corr(ONI, centered-3mo Nino34)=0.9777 vs corr(ONI, trailing-3mo)=0.9500. ~22.5% of regime labels change if a purely causal (trailing) ONI proxy is used. build_enso_csv.py:23 only says '3-month running mean'; paper l.197-198 only says 'smoothed version'.

**Suggestion :** State explicitly that the NOAA ONI is centered and that R_n is therefore a non-causal function of Y including Y_{n+1}; discuss the implied label leakage for both the supervised E1/E2 fits and the causal-filter accuracy evaluation, or report a causal-label robustness check.

**Ajustement de sévérité (vérificateurs) :** Keep 'high'. The finding is fully supported and touches the paper's central empirical justification (the E1 per-regime F-test used to argue H5 is not rejected) plus the regime-detection numbers, and the leakage is undisclosed. A defensible case for 'medium' exists — the leak is only one month, the paper gestures at the smoothing as a detection-difficulty caveat at l.197-198, and under causal labels the F-test does not actually flip from fail-to-reject to reject (Neutral F=2.74 ~ p~=0.10, still not significant) — but since the issue is genuinely undocumented and bears on the headline applicability claim, high remains appropriate. The only mild overstatement is the implication that E1 conclusions would change; they shift but do not invert. | high -> low (cap at medium). The underlying disclosure gap is real, but the CONTEXTE lens shows the impact is far smaller than "high" implies, because the leakage direction is adverse to the proposed method and no published headline claim depends on it.

### ✅ [HIGH] Committed results/enso/ contains stale, contradictory E3 artifacts produced by the S&P/VIX script, not by run_real_data.py

`results/enso/e3_summary.json + results/enso/e3_table.tex:e3_summary.json:1-86; e3_table.tex:1-15` — statut : confirmed (2 vote(s)) — catégorie : reproducibility / stale artifacts

results/enso/ holds TWO inconsistent E3 result sets. The current, paper-backing one is e3_table.json (variants V0_unconstrained / V1_posthoc_AB / V2_GEM_AB; acc 0.522/0.544/0.478) which I reproduced exactly with `python -m prg.experiments.run_real_data`. But e3_summary.json and e3_table.tex are an OLDER scheme with FOUR variants V0_unconstrained / V1_posthoc_B / V2_posthoc_A / V3_GEM_B — the exact variant names hard-coded in scripts/e3_bw_em.py:65-70 (the S&P/VIX experiment), not in run_real_data.py:223-227. e3_summary.json even contains test_LL=-58430 and mse=3.8e234 for V2_posthoc_A. These stale files contradict the paper's Table (tab_enso_em.tex) and look like S&P/VIX output mistakenly committed under results/enso/. run_real_data.py never writes e3_summary.json or e3_table.tex, so they are orphaned and will mislead anyone reproducing §7.

**Preuve :** e3_table.json variants: V0_unconstrained/V1_posthoc_AB/V2_GEM_AB. e3_summary.json variants: V0_unconstrained/V1_posthoc_B/V2_posthoc_A/V3_GEM_B with mse=3.836729561424954e+234. scripts/e3_bw_em.py:65-70 defines exactly V0_unconstr/V1_posthoc_B/V2_posthoc_A/V3_GEM_B. run_real_data.py:521-526 only writes e3_table.json/.tex.

**Suggestion :** Delete results/enso/e3_summary.json and results/enso/e3_table.tex (and e1_table.tex/e2_table.tex if also stale) or regenerate them from run_real_data.py so the committed ENSO results are internally consistent.

**Ajustement de sévérité (vérificateurs) :** high -> medium (real reproducibility-hygiene defect, but no published number is affected and the fix is a one-line git rm) | Downgrade high -> medium (arguably low-medium). The stale files do NOT feed the compiled paper: 07_real_data.tex:151 \input's paper/figures/generated/tab_enso_em.tex (3-variant, V0/V1_posthoc_AB/V2_GEM_AB), which I reproduced exactly via run_real_data. results/enso/e3_summary.json and e3_table.tex are orphaned results-directory artifacts that no published claim depends on, so this is a reproducibility-hygiene / code-quality issue, not a correctness or claim-validity defect. 'high' is inconsistent with the finding's own observation that e3_table.json backs the paper.

### ✅ [MEDIUM] S&P/VIX results backing the §7 'p<10^-4' claim are NOT committed (gitignored); only ENSO summaries are tracked

`.gitignore:35-38 + results/e1/table1.json, results/e1/table1.tex:.gitignore:35-38` — statut : confirmed (2 vote(s)) — catégorie : reproducibility

Section 7 (paper l.99-100) asserts the H5 F-test 'rejects (H5) at p<10^-4 for the dominant regime' on daily S&P500/VIX. That number is produced by scripts/e1_supervised_h5.py, which writes to results/e1/. But .gitignore line 35 ignores results/* and line 37 only un-ignores results/enso/, so results/e1/table1.json (and results/e2/, results/e3/) are NOT git-tracked — they only exist locally (file dated Apr 20, absent from `git ls-files`). I re-ran the script and reproduced p=2.07e-31 (F=141) for the L1 dominant 'calm' regime, so the claim is correct, but it is not reproducible from committed artifacts alone. Additionally .gitignore line 38 explicitly ignores results/enso/regime_trace.csv, yet make_figures_real.py:42,102-106 requires that file to regenerate fig_enso_regime_trace.pdf — so the committed figure cannot be regenerated without first re-running E3.

**Preuve :** .gitignore:35 `results/*`, :37 `!/results/enso/`, :38 `results/enso/regime_trace.csv`. `git ls-files results/` lists only results/enso/{e1,e2,e3}* — no results/e1/. make_figures_real.py:42 TRACE_CSV = results/enso/regime_trace.csv; line 103 SKIP if not exists.

**Suggestion :** Either commit the S&P/VIX results (results/e1,e2,e3) backing the §7 financial-comparison claim, or state in the paper which script regenerates them; un-ignore results/enso/regime_trace.csv so the regime-trace figure is regenerable.

**Ajustement de sévérité (vérificateurs) :** medium -> low | medium -> low (real but minor: ~1.5-1.8pp, qualitative conclusion survives)

### ✅ [MEDIUM] Neutral-only baseline '46%' is the full-record fraction, but accuracy is on the test segment where the baseline is 47.8%

`paper/sections/07_real_data.tex:190-191:190-191` — statut : confirmed (2 vote(s)) — catégorie : consistency / baseline mismatch

Line 191 compares regime accuracy '(vs. 46% for the trivial Neutral-only predictor)'. The accuracy figures are computed on the TEST period (2011-01..2026-02, 182 months, run_real_data.py:280 best_perm_acc_ari on rs_te). On that test segment the Neutral class is 87/182 = 47.8%, not 46%. The 46% (462/914=46.3%) is the FULL-record neutral fraction quoted at l.39-40. So the prose grades a test-period accuracy against a full-record baseline, understating the trivial baseline by ~1.8pp and overstating the model's edge. (This is distinct from the separately-acted '53% vs tab 0.544/0.522' item — here the issue is the baseline denominator, not the model number.)

**Preuve :** Test regime counts: La Nina 49 (26.9%), Neutral 87 (47.8%), El Nino 46 (25.3%); test neutral fraction = 0.478. Full-record neutral fraction = 0.4628. Paper l.40 quotes '423 Neutral (46%)' (full record).

**Suggestion :** Use the test-period Neutral-only baseline (47.8%) for the comparison, or state explicitly that 46% is the full-record class prior.

**Ajustement de sévérité (vérificateurs) :** medium -> low | medium -> low

### ❓ [MEDIUM] S&P/VIX fetch is non-deterministic (live yfinance + revisable FRED, no version/date pinning); committed CSV cannot be regenerated to match

`scripts/fetch_sp500_vix.py:42-66,121:42-66,121` — statut : uncertain (2 vote(s)) — catégorie : reproducibility / non-deterministic source

fetch_sp500_vix.py downloads ^GSPC live via yfinance (line 51) and VIXCLS from FRED (line 42). Neither source is version-pinned nor is the download date recorded; FRED's VIXCLS and Yahoo's adjusted ^GSPC are subject to revision/back-adjustment, and yfinance has no determinism guarantee. The script's own docstring (line 27) says 'meant to be committed for reproducibility', which is the right mitigation — and I verified the COMMITTED sp500_vix.csv is internally consistent (recomputed log_return matches to 1e-6) and reproduces the train standardization stats stored in results/e1/table1.json exactly. So results are reproducible FROM the committed CSV, but re-running the fetch script will not reproduce the committed CSV byte-for-byte. There is no provenance metadata (download date, yfinance/FRED snapshot) recorded.

**Preuve :** fetch_sp500_vix.py:51 yf.download('^GSPC', ...); :42 FRED_URL live CSV; :121 merged.to_csv(...). No seed, no snapshot date, no version capture. Committed CSV range 2004-01-05..2024-12-30 (DEFAULT_END=2024-12-31, last trading day 12-30).

**Suggestion :** Record provenance (download date + yfinance/FRED versions) in the CSV header or a sidecar, and state in the paper that the committed CSV — not a re-fetch — is the canonical reproducibility artifact.

**Ajustement de sévérité (vérificateurs) :** Si elle était conservée comme remarque, la rétrograder de medium à info: le CSV gelé committé est la convention annoncée (docstring l.27) et l'artefact canonique consommé par les scripts; la seule claim dépendante (p<10^-4) est reproductible. Au mieux un nice-to-have (métadonnée de provenance), pas un défaut de reproductibilité. | none — low is correct

### ❌ [LOW] 'La Niña posterior peaks during the 2020-23 triple-dip' is only weakly supported by the reproduced V0 trace

`paper/sections/07_real_data.tex:186-189:186-189` — statut : refuted (1 vote(s)) — catégorie : consistency / claim support

Line 187-189 claims the V0 filter posterior shows El Nino peaks aligned with 2015-16/2023-24 and La Nina peaks during the 2020-23 triple-dip. From the reproduced regime_trace (V0, best perm [0,2,1]): pi(El Nino) averages 0.471 in 2015-16 and 0.455 in 2023-24 vs 0.224 overall — so the El Nino claim holds qualitatively. But pi(La Nina) averages only 0.197 during 2020-08..2023-03 vs 0.164 overall (a tiny lift), and the confusion matrix shows 46/49 true La-Nina test months are misclassified as Neutral. The La-Nina-detection claim is thus much weaker than the El-Nino one. The paper does acknowledge detection is 'challenging', but the specific La-Nina-peak assertion overstates what the trace shows.

**Preuve :** Reproduced V0 test trace: avg pi(LaNina) 2020-08..2023-03 = 0.197 vs overall 0.164; true La Nina predicted as {Neutral:46, La Nina:3}. avg pi(ElNino) 2015-16=0.471, 2023-24=0.455 vs overall 0.224.

**Suggestion :** Soften or qualify the La-Nina-peak claim (e.g. note La Nina is largely absorbed into the Neutral class), or report the per-regime detection rates so the asymmetry is visible.

### ❌ [LOW] L2 NBER recession label is hand-coded with only two periods and an arguable COVID end date; docstring leak-safety claim is incomplete

`scripts/labels.py:14-19,31-34,64-71:31-34,64-71` — statut : refuted (1 vote(s)) — catégorie : data-construction

build_label_L2 (labels.py:64-71) hard-codes NBER recessions as just two intervals (2007-12..2009-06 Great Recession; 2020-02..2020-04 COVID). The COVID trough end is given as 2020-04-30, consistent with NBER's April-2020 trough, but the choice is undocumented and not sourced. More importantly, the module docstring (l.18-19) claims 'neither label leaks test-period statistics back into the training set' — true for L1 (threshold from train median) and for L2's dates, but L1's threshold is learned on train then APPLIED to the full df (line 58 df[column] > threshold), which is fine; however the docstring's blanket leak-safety statement does not address that the underlying X/Y standardization and the regime definition are separate concerns. Minor, but the recession periods should cite an NBER source for reproducibility.

**Preuve :** labels.py:31-34 NBER_RECESSIONS = [('2007-12-01','2009-06-30'),('2020-02-01','2020-04-30')]; docstring l.18-19 leak-safety claim. This is for the S&P/VIX suite (not directly §7 numbers, but supports the §7 financial comparison).

**Suggestion :** Cite the NBER business-cycle dating source and the date of retrieval for the recession intervals; this is the only non-market-data input and should be pinned like the rest.

**Ajustement de sévérité (vérificateurs) :** N/A (isReal=false). Si conservée, ne serait au mieux qu'un nit documentaire (sous 'low'/'info'): citer la source NBER. L'affirmation 'docstring leak-safety incomplete' est à retirer car le code est vérifiablement train-only partout.

---
_Généré automatiquement ; raisonnements complets dans `audit/raw/04b-extracted.json`._