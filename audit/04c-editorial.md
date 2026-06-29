# Vague 4c — Éditorial, bibliographie, hygiène LaTeX

Workflow `audit-4c-find` (run `wf_0450767f-bcd`) + vérif (runs `wf_98ebfd16-ab0`, `wf_66cceb90-c0d`). 3 finders + vérification adversariale. Détails : `raw/04c-extracted.json`.

**Bilan : 26 trouvailles — 18 confirmées** (0 critical, 1 high, 6 medium, 6 low, 5 info), 3 incertaines, 5 réfutées.

## Trouvailles majeures (critical + high confirmées)

- **[HIGH] Mangled author rendering in .bbl from thin-space (\,) inside name initials** — `paper/paper.bib:117`

## Cohérence éditoriale & sur-affirmations

_Audited editorial coherence and overclaims across the abstract (00), introduction (01), conclusion (08), cross-checked against the body (02 model/H5, 03 filtering, 04 AB-constraint, 05 estimation, 06 simulation, 07 ENSO real-data) and appendices A (time reversal), B (H5 derivation), C (AB proof). Focus: claims in abstract/intro/conclusion not supported by the body, abstract-conclusion inconsistencies, contributions announced but not delivered (or body results never surfaced), novelty/optimality/exactness overclaims, and key-term consistency (H5 vs AB vs exact/approx). Excluded all previously acknowledged items (b_X/b_Y in eq:mu_jk, the 'K·s>=q+s' counting gap as stated in the intro, 'unique closed-form solution' oversell, P/R/b/W/Σ_W notation collisions, mixed index conventions, dead macros, \\yN vs \\yn, ref/label issues). Reported 10 NEW findings. Highest-impact: (1) the sole real-data experiment shows the regime-blind K=1 Kalman filter beating every proposed GSS filter, a caveat absent from abstract/conclusion; (2) simulation accuracy gains over standard IMM are negligible (<=0.5% RMSE) yet abstract frames exactness as a substantive win; (3) abstract lists Σ_U among the blocks determining A,B though body+proof state A,B are Σ_U-independent (three inconsistent block lists across sections). Also: an unfinished/unadvertised BIC model-selection subsection, an unbacked 'p<10^-4' S&P500/VIX claim, 'generically necessary' asserted in abstract/conclusion without any genericity argument in the appendix, 'only source of suboptimality / almost all subsequent work' unsupported universal, a future-work item ('exact smoothing in the M-step') that contradicts the already-implemented smoothed-posterior M-step, the abstract overstating the filtering contribution vs Sec 3's 'not a new filter' disclaimer, and an undefined matrix 'W' in the H5-residual verification formula in Sec 6._

### ❓ [HIGH] Abstract/intro tout exact filtering; sole real-data result shows the regime-blind K=1 Kalman filter beats every GSS variant

`paper/sections/07_real_data.tex:120-135, 226-231` — statut : uncertain (2 vote(s)) — catégorie : overclaim-vs-body

The framing sections sell the exact H5-IMM filter as the central deliverable (abstract: 'runs without any approximation'; intro contribution 1; conclusion: 'fast, exact optimal filter'). But the only real-data benchmark reports that a single Kalman filter ignoring regimes achieves BOTH the lowest test MSE AND the highest log-likelihood/obs, beating all regime-aware GSS filters. The exact filter's headline advantage (best test log-likelihood = -30.09) holds only *among the regime-aware filters*, a qualifier buried in the body and absent from abstract/conclusion. A reader of the abstract/conclusion alone would not learn that on the paper's sole real dataset the proposed machinery is dominated by a textbook baseline.

**Preuve :** 07_real_data.tex:131-135 'The single Kalman filter (K=1) achieves both the lowest test MSE and the highest log-likelihood per observation ... an important caveat'; vs abstract:18-21 'runs without any approximation'; 08_conclusion.tex:10 'admits a fast, exact optimal filter.'

**Suggestion :** Add the K=1-dominance caveat (or at least 'among regime-aware filters') to the conclusion's empirical summary, and soften the abstract's claim that the framework is the operative win on real data.

**Ajustement de sévérité (vérificateurs) :** Would be at most low (cosmetic), not high: the only defensible kernel is that abstract/conclusion omit the K=1 caveat — but neither makes a contradicting empirical claim and the body states the caveat twice. Recommend isReal=false. | high -> low

### ❌ [HIGH] Abstract sells the exact filter as removing IMM error, but simulation gains are negligible (RMSE within 0.1-0.5%) on every model

`paper/sections/06_experiments.tex:157-205` — statut : refuted (2 vote(s)) — catégorie : overclaim-vs-body

The abstract/intro frame the Gaussian-collapse approximation as 'the only source of suboptimality' that the method eliminates. Yet the simulation study shows the exact filter and approximate IMM are 'nearly identical' on M1 (within 0.1% RMSE), 'near-identical' on M3, and differ by only 0.4595 vs 0.4616 RMSE on M2 (~0.5%). The accuracy benefit demonstrated is essentially in the noise; the real measured win is CPU (~1.4x), which the abstract never mentions. This is a contribution-vs-evidence mismatch: the headline ('removes the approximation') is technically true but the body shows the approximation was practically harmless in all tested configurations.

**Preuve :** 06_experiments.tex:162-163 'both filters achieve nearly identical RMSE (within 0.1%...)'; :197-198 '0.4595 vs 0.4616'; :205 'both filters again perform near-identically in RMSE'; vs 01_introduction.tex:24 'the only source of suboptimality'.

**Suggestion :** Temper the abstract/intro to state the exactness yields a guaranteed-correct covariance/likelihood and a CPU saving, while empirically the RMSE gap over standard IMM is small in the tested regimes; report it honestly rather than implying a large accuracy win.

**Ajustement de sévérité (vérificateurs) :** Si conservée, serait au mieux « low »: simple suggestion d'ajouter une phrase sur le CPU/calibration comme gains pratiques. Mais le cœur de la trouvaille (abstract impliquant un large gain de précision) repose sur une mauvaise caractérisation de ce que l'abstract affirme réellement (exactitude, pas amplitude RMSE), donc isReal=false. | N/A (isReal=false). Au mieux ce serait une remarque cosmetique de severite 'low' (ajouter une demi-phrase sur le gain CPU dans l'abstract), pas 'high'.

### ✅ [MEDIUM] Abstract lists Σ_U among the five blocks that determine A and B; body and proof state A,B are independent of Σ_U

`paper/sections/00_abstract.tex:32-34` — statut : confirmed (2 vote(s)) — catégorie : abstract-body-inconsistency

The abstract says the AB constraint 'determines A and B jointly from the remaining five blocks (C, D, Σ_U, Σ_V, Δ)'. But Sec 4 explicitly states A and B are determined 'independently of Σ_U' (04_constraint.tex:79), the boxed formula A=ΔΣ_V^{-1}C, B=ΔΣ_V^{-1}D contains no Σ_U, the BIC count treats only the five blocks, and the appendix proof confirms no Σ_U dependence. The intro contribution 2 even lists only four blocks '(C, D, Δ, Σ_V)'. So three different block lists circulate: abstract = {C,D,Σ_U,Σ_V,Δ}, intro = {C,D,Δ,Σ_V}, Sec 4 body = '(C, D, Σ_U, Σ_V, Δ) independently of Σ_U'. Including Σ_U as a *determinant* of A,B in the abstract is misleading.

**Preuve :** 00_abstract.tex:32-33 'from the remaining five blocks (C, D, \Sigma_U, \Sigma_V, \Delta)'; vs 04_constraint.tex:78-79 'determined by the remaining five blocks (C, D, \Sigma_U, \Sigma_V, \Delta), independently of \Sigma_U' and eq:AB containing no Σ_U; vs 01_introduction.tex:81 'determined jointly from (C, D, \Delta, \Sigma_V)'.

**Suggestion :** Use one consistent statement everywhere: A,B are determined by (C,D,Δ,Σ_V) only; Σ_U is a free block of the model but does not enter A,B. Fix the abstract to not list Σ_U as a determinant.

**Ajustement de sévérité (vérificateurs) :** medium -> low | medium -> low

### ❓ [MEDIUM] BIC model-order-selection subsection presents a method but delivers no result; only one BIC number, K=1,3,4 deferred

`paper/sections/06_experiments.tex:362-393` — statut : uncertain (2 vote(s)) — catégorie : contribution-not-delivered

Section 6.6 introduces a full BIC apparatus (parameter-count formula eq:bic_d, BIC definition) as if model-order selection were a contribution, but then states that the comparison values for K=1,3,4 are 'deferred to a companion computational study'. Only a single BIC number for the true K=2 is reported, which selects nothing. This is a half-delivered item: it occupies a numbered subsection and derives machinery whose payoff is explicitly punted. Neither the intro's four-contribution list nor the conclusion mentions BIC/model selection at all, so it is body material that is both unfinished and unadvertised.

**Preuve :** 06_experiments.tex:390-393 'A proper BIC-based model order selection requires fitting H5-GSS models with K=1,3,4 ... this is deferred to a companion computational study'; the four contributions in 01_introduction.tex:57-100 do not list model selection.

**Suggestion :** Either complete the K-sweep, or demote the BIC material to a brief remark stating the parameter count is available for future model selection, avoiding a subsection that promises selection and delivers none.

**Ajustement de sévérité (vérificateurs) :** medium -> low | If retained at all, downgrade from medium to low (cosmetic title-vs-content nitpick), since the in-text deferral disclaimer removes the substantive coherence concern.

### ✅ [MEDIUM] 'generically necessary' claim survives in abstract+intro+conclusion with no supporting argument anywhere (counting argument absent from Appendix C)

`paper/sections/00_abstract.tex:33-34` — statut : confirmed (2 vote(s)) — catégorie : overclaim-not-supported

This is the same family as the acknowledged 'K·s>=q+s' issue, but note that the weaker phrasing in the abstract and conclusion ('generically necessary', dropping the inequality) is ALSO unsupported: Appendix C (app:AB_constraint) proves only sufficiency (Prop AB_satisfies_H5) and a uniqueness-of-elimination argument that assumes the constraint must hold 'across the cone of admissible Σ(r)≻0'. There is no genericity/counting argument establishing necessity for fixed structural blocks. The three framing sections assert necessity as fact while the appendix delivers only sufficiency plus a 'for all Σ' elimination (which is a different and stronger premise than 'generic'). Flagging because the abstract/conclusion phrasing is a distinct location from the acknowledged intro inequality.

**Preuve :** 00_abstract.tex:33-34 'sufficient — and generically necessary — for (H5)-compatibility'; 08_conclusion.tex:18-19 'always sufficient ... and generically necessary in all configurations'; Appendix C contains only Prop AB_satisfies_H5 (sufficiency) and the 'across the cone of Σ≻0' elimination, no genericity statement.

**Suggestion :** Restate uniformly as: the AB form is the unique solution valid for all Σ(r)≻0 (the 'free-covariance' notion the appendix actually proves), and drop 'generically necessary' unless a genuine genericity argument for fixed blocks is added.

**Ajustement de sévérité (vérificateurs) :** Keep at medium, leaning low-medium. The claim is a verifiable overclaim of a strong mathematical term in the abstract, and the cited Appendix C demonstrably lacks the supporting argument — that justifies medium. However, its INDEPENDENT weight is reduced because it explicitly belongs to the same family as the already-acknowledged "K·s>=q+s necessity" finding (the intro at line 84 does carry the qualifier, so an attentive reader gets a more careful statement), and because the practically-used direction (sufficiency) IS rigorously proven, so the methodology is unaffected. The distinct marginal contribution here is narrow: the abstract/conclusion drop the inequality and the cited appendix has no genericity proof. Acceptable at medium; could reasonably be downgraded to low given the overlap. | medium -> low

### ✅ [MEDIUM] 'the only source of suboptimality' / 'almost all subsequent work' stated as fact without citation support

`paper/sections/01_introduction.tex:19-26` — statut : confirmed (2 vote(s)) — catégorie : overclaim-not-supported

The intro asserts the Gaussian collapse 'is the only source of suboptimality in the algorithm and the entry point for almost all subsequent work on improved JMLS filters'. The first half is defensible for the idealized IMM, but the sweeping 'almost all subsequent work' is an unsupported universal claim about a large literature, backed only by two particle-filter citations (doucet_particle_2001, andrieu_particle_2003). Particle/RBPF methods are not primarily motivated by removing the IMM collapse; many target nonlinear/non-Gaussian dynamics. This is a 'first/only/all'-type overclaim.

**Preuve :** 01_introduction.tex:24-26 'This Gaussian-collapse approximation is the only source of suboptimality in the algorithm and the entry point for almost all subsequent work on improved JMLS filters~\cite{doucet_particle_2001,andrieu_particle_2003}'.

**Suggestion :** Soften to 'a primary source of suboptimality' and 'motivates much subsequent work', or cite a survey that actually frames the literature this way.

**Ajustement de sévérité (vérificateurs) :** medium -> low | medium -> low

### ✅ [MEDIUM] Financial-signal (S&P500/VIX) H5-rejection claim 'p<10^-4' is asserted with no experiment, table, or appendix backing it

`paper/sections/07_real_data.tex:98-100` — statut : confirmed (2 vote(s)) — catégorie : empirical-overgeneralization

To contrast ENSO's H5-compatibility, the text claims the same F-test 'rejects (H5) at p<10^-4 for the dominant regime' on daily S&P500/VIX. No such experiment is reported anywhere in the paper (no table, figure, data description, or appendix). It is a concrete numerical claim about an unshown dataset used rhetorically to bolster the ENSO result. Either it is an unreported result (reproducibility gap) or an anecdote stated with false precision.

**Preuve :** 07_real_data.tex:98-100 'contrasts with financial signals (e.g., daily S&P500/VIX), where the same test rejects (H5) at p<10^-4 for the dominant regime.' No S&P500/VIX experiment appears elsewhere.

**Suggestion :** Either add the financial experiment (even briefly, in an appendix) or remove the specific p-value and present the contrast qualitatively without an unbacked number.

**Ajustement de sévérité (vérificateurs) :** none — severity 'medium' is appropriately calibrated | Keep medium; defensibly could be low-medium since it is a single illustrative aside that no result depends on, but the false-precision/reproducibility nature of stating an unbacked specific p-value keeps it at medium.

### ❌ [LOW] Conclusion lists 'exact smoothing' as future work, but the EM M-step already uses smoothing posteriors (γ,ξ)

`paper/sections/08_conclusion.tex:33-35` — statut : refuted (1 vote(s)) — catégorie : future-work-vs-body

Future-work item (i) proposes 'Extending the estimation procedure to exact smoothing, by replacing the filtering posteriors with the smoothing posteriors in the M-step.' But the semi-supervised EM in Sec 5 already runs forward-backward and uses the smoothed posteriors γ_n(k)=p(r_n|z_{1:N}) and ξ_n(j,k) in the M-step (05_estimation.tex:94-119). So the M-step is not based on filtering posteriors; the stated future direction mischaracterizes what the body already does. The intended direction is presumably RTS-style state smoothing of X, not regime smoothing — but as written it contradicts the delivered method.

**Preuve :** 08_conclusion.tex:34-35 'replacing the filtering posteriors with the smoothing posteriors in the M-step'; vs 05_estimation.tex:94-95 'smoothed posteriors \gamma_n(k):=p(r_n=k|z_{1:N})' already used in eq:M_step_Theta.

**Suggestion :** Reword to clarify the future work is exact *state* (RTS) smoothing of X within each regime, since the regime-posterior smoothing in the M-step is already implemented.

### ❌ [LOW] Abstract calls IMM the 'workhorse filter' and frames whole paper around IMM, but the filter is explicitly 'not a new filter' — a re-expression of the companion paper

`paper/sections/00_abstract.tex:5-8` — statut : refuted (1 vote(s)) — catégorie : terminology-consistency

The abstract and contribution 1 present the exact-IMM filter as a paper deliverable ('runs without any approximation'), but Sec 3 states plainly 'The recursion below is therefore not a new filter; it is a re-expression ... of the recursion derived in [companion]' (03_filtering.tex:19-22). The novelty is the IMM *reading* and the bias extension, not the filter. The abstract's phrasing ('we show that this approximation is unnecessary ... the standard IMM cycle ... runs without any approximation') reads as a new filtering result rather than a re-derivation. Minor, but the editorial altitude of the abstract overstates the filtering contribution relative to the body's honest disclaimer.

**Preuve :** 03_filtering.tex:19-22 'not a new filter; it is a re-expression, in the language of Kalman filtering and IMM, of the recursion derived in~\cite{derrode_fast_2026}'; vs abstract:17-21 framing the IMM cycle as a result of this paper.

**Suggestion :** Add a half-sentence to the abstract crediting the companion paper for the underlying filter and positioning this paper's filtering contribution as the IMM reformulation + affine-bias extension.

### ✅ [LOW] H5 residual formula uses undefined matrix W: '||Δ^T A^T + Σ_V B^T - P M^{-1} W||_F'

`paper/sections/06_experiments.tex:18-19` — statut : confirmed (1 vote(s)) — catégorie : notation-consistency

The reference-model definition gives the H5 residual as ||Δ^T A^T + Σ_V B^T - P M^{-1} W||_F < 1e-10. The compact constraint in Sec 4 (eq:H5_compact) is Δ^T A^T + Σ_V B^T = P M^{-1}(Q A^T + R B^T + Δ^T); there is no matrix 'W' in the constraint. 'W' is the process-noise vector everywhere else in the paper (eq:dynamics), so using 'W' for the inner bracket (Q A^T + R B^T + Δ^T) is an undefined/colliding symbol in a verification formula readers may want to reproduce.

**Preuve :** 06_experiments.tex:18-19 '\|\Delta^T A^T + \Sigma_V B^T - PM^{-1}W\|_F < 10^{-10}'; the inner bracket in eq:H5_compact (04_constraint.tex:44) is (Q A^T + R B^T + Δ^T), not 'W'.

**Suggestion :** Replace 'W' with the explicit bracket (QA^T+RB^T+Δ^T) or define a named symbol for it, avoiding reuse of W.

## Bibliographie

_Audited the bibliography of the exactIMM paper: paper/paper.bib (21 entries), paper/paper.bbl (18 compiled bibitems), paper/paper.blg (bibtex log, 0 warnings), paper/paper.aux (\\citation/\\bibcite records), and all \\cite usages grepped across paper/sections/ (00..08), paper/appendix/ (B, E), and paper/paper.tex. Cross-checked defined-vs-cited keys, rendered author/title metadata, and self-containedness. NO broken citation keys and NO undefined-citation warnings were found (no ?? in compiled output; paper.log clean of citation/reference warnings). New findings beyond the previously-acted items: (1) 3 defined-but-uncited entries silently dropped from the compiled bibliography (ghahramani_variational_2000, horn_matrix_2013, rabiner_tutorial_1989) — likely missing citations rather than unused refs since all three are topically relevant; (2) mangled author names in the .bbl for 4 entries (Dempster, Wu, Blom, Anderson) caused by thin-space \\, inside name initials confusing IEEEtran name-parsing — HIGH because it corrupts the foundational EM references; (3) pieczynski_triplet_2002 citekey year (2002) contradicts year field/render (2005) with a likely-wrong ASMDA venue; (4) the derivation in appendix B and the model/filter sections depend on an unpublished 'under review' companion paper (derrode_fast_2026, cited 10x and used to justify a proof step at B_h5_derivation.tex:24) — self-containedness risk; plus a missing Schur-complement citation, metadata-completeness asymmetry (only 1 DOI), title-brace over-protection on Huang, and a lone citation locator. All file:line references included per finding._

### ✅ [HIGH] Mangled author rendering in .bbl from thin-space (\,) inside name initials

`paper/paper.bib:117` — statut : confirmed (2 vote(s)) — catégorie : bibliography/metadata-rendering

Four author fields use a LaTeX thin-space \, between given-name initials (e.g. 'A.\,P.'). IEEEtran.bst treats the literal characters in the name field per BibTeX name-parsing rules and mis-tokenizes 'A.\,P.' so the trailing initial is emitted as a spurious 'von'/last-name fragment. The compiled bibliography therefore prints corrupt author names. This affects four references including the foundational Dempster-Laird-Rubin EM paper and Wu's EM-convergence paper, both load-bearing for section 05.

**Preuve :** paper.bib:117 `author = {Dempster, A.\,P. and Laird, N.\,M. and Rubin, D.\,B.}` renders in paper.bbl:78 as 'P.~Dempster, A.\, M.~Laird, N.\, and B.~Rubin, D.\,' (initials detached and reordered). Same defect at paper.bib:38 (Blom -> paper.bbl:38 'P.~Blom, Henk~A.\ and Y.~Bar-Shalom'), paper.bib:174 (Wu -> paper.bbl:98 'F.~J. Wu, C.\,'), paper.bib:206 (Anderson -> paper.bbl:109 'O.~Anderson, Brian~D.\ and J.~B. Moore').

**Suggestion :** Remove the thin-space: write initials as 'A.P.' or 'A. P.' (e.g. `author = {Dempster, A. P. and Laird, N. M. and Rubin, D. B.}`). Same fix for Blom (Henk A. P.), Wu (C. F. Jeff), Anderson (Brian D. O.). Then re-run bibtex and check the rendered names.

**Ajustement de sévérité (vérificateurs) :** Defensible at high, but a slight recalibration to medium is reasonable: this is a bibliography/typesetting defect with no impact on the paper's mathematical content and a trivial one-line-per-entry fix. It stays well above low/info because it produces visibly corrupt author names (not mere misformatting) across four references, two of them (Dempster-Laird-Rubin EM, Wu EM-convergence) load-bearing for Section 5. medium-to-high is the defensible band. | Keep high; medium would also be defensible (it is a bibliography-rendering defect with no impact on the paper's math or results), but high is justified by the reviewer-visible garbling of foundational EM/IMM author names in a manuscript under review.

### ❓ [HIGH] Derivation depends on an unpublished companion paper (derrode_fast_2026)

`paper/paper.bib:8-14` — statut : uncertain (2 vote(s)) — catégorie : bibliography/self-containedness

derrode_fast_2026 is an @unpublished 'Manuscript under review' and is the single most-cited reference (10 of the in-text \cite occurrences). Crucially it is cited to justify load-bearing steps of the derivation, not just for context: appendix/B_h5_derivation.tex:24 invokes it to assert that 'all quantities relevant to (H5) may be evaluated with the noise law', and sections/02/03 lean on it for the model and the unbiased filter being extended. Because the companion is not publicly available, a reviewer cannot verify the cited Markovianity/noise-law argument, which is a self-containedness problem (and the matter the user flagged as 'CS_FinaleBis' dependency).

**Preuve :** @unpublished{derrode_fast_2026, note={Manuscript under review}} (paper.bib:8-14). Cited 10x: 00_abstract.tex:11,23; 01_introduction.tex:44,49,59; 02_model_h5.tex:28,38,97; 03_filtering.tex:8,22,149; 08_conclusion.tex:10; appendix/B_h5_derivation.tex:24. At B_h5_derivation.tex:23-26 the proof step explicitly defers to '...the companion paper~\cite{derrode_fast_2026}'.

**Suggestion :** Either (a) include the cited noise-law / time-reversal Markovianity argument self-contained in Appendix B (the paper already has app:time_reversal it references alongside), so the derivation does not require an unavailable manuscript, or (b) add an arXiv/preprint identifier to derrode_fast_2026 so reviewers can access it. Update the note with a DOI/arXiv id once available.

**Ajustement de sévérité (vérificateurs) :** If retained at all, this is not a MATHS/verification defect (the Markovianity argument is fully proved in-paper in Appendix A). The only residual — deferring Proposition 1's proof and the §3 filter to the under-review companion — is a CONTEXTE-style editorial concern at most low/info severity, well below the asserted "high". | Downgrade high -> medium. Real self-containedness/accessibility concern, but it is a routine, transparently-attributed companion-paper dependency, not a mathematical defect or broken claim, and the single most load-bearing cited step is backstopped by a self-contained in-paper proof.

### ✅ [MEDIUM] Three .bib entries defined but never cited (dead bibliography entries)

`paper/paper.bib:160-168, 213-219, 138-147` — statut : confirmed (2 vote(s)) — catégorie : bibliography/dead-entries

Three entries exist in paper.bib but are cited nowhere in sections/, appendix/, or paper.tex. They do not appear in paper.bbl (18 bibitems vs 21 defined entries) or in the \bibcite records of paper.aux, so they silently vanish from the compiled bibliography. Either cite them where relevant or remove them. Note ghahramani_variational_2000 (Switching State-Space Models) and rabiner_tutorial_1989 (HMM tutorial) are highly relevant to the EM/Baum-Welch discussion in section 05, and horn_matrix_2013 (Matrix Analysis) is the natural reference for the Schur complement actually invoked in appendix B (see separate finding) — so this is more likely a missing-citation oversight than truly unused refs.

**Preuve :** paper.bib defines 21 keys; only 18 are cited. Uncited: ghahramani_variational_2000 (line 160), horn_matrix_2013 (line 213), rabiner_tutorial_1989 (line 138). Verified via `grep -rn` over sections/ appendix/ paper.tex: '(no usage)' for all three. paper.blg line 16: 'You've used 18 entries'. paper.bbl has 18 \bibitem.

**Suggestion :** Cite rabiner_tutorial_1989 and bilmes/baum cluster together at sections/05_estimation.tex:75; cite ghahramani_variational_2000 in the intro/EM discussion of switching SSMs; cite horn_matrix_2013 at the Schur complement in appendix/B_h5_derivation.tex:74. Otherwise delete the unused entries.

**Ajustement de sévérité (vérificateurs) :** Lower from medium to low: unused .bib entries are a cosmetic maintenance issue that BibTeX handles silently and never surface in the compiled PDF; no correctness, math, or reader-facing impact. | medium -> low

### ✅ [MEDIUM] pieczynski_triplet_2002: citekey year (2002) contradicts year field (2005); venue likely wrong

`paper/paper.bib:83-89` — statut : confirmed (2 vote(s)) — catégorie : bibliography/metadata

The entry key encodes '2002' but the year field is {2005}, and the .bbl prints 2005. 'On Triplet Markov Chains' by Pieczynski and Desbouvries is the well-known 2002 work (ICASSP / Comptes Rendus Mathematique 2002). The entry assigns it to the ASMDA symposium with year 2005, which conflates two different items and at minimum makes the citekey misleading. This is one of the two PMC/TMC foundational references in the intro, so the venue should be correct.

**Preuve :** paper.bib:83 key `pieczynski_triplet_2002` but paper.bib:88 `year = {2005}`. Rendered paper.bbl:56-59: '...in Proceedings of the International Symposium on Applied Stochastic Models and Data Analysis (ASMDA), 2005.' Cited at sections/01_introduction.tex:32 `\cite{pieczynski_pairwise_2003,pieczynski_triplet_2002}`.

**Suggestion :** Reconcile the year and venue against the intended source. If it is the 2002 'On Triplet Markov Chains' (C. R. Acad. Sci. Paris / ICASSP 2002), set year={2002} and the correct booktitle/journal; if it is genuinely an ASMDA 2005 paper, rename the key to *_2005 to stop the year/key contradiction.

**Ajustement de sévérité (vérificateurs) :** Downgrade medium -> low. The verified defect is a cosmetic citekey/year-suffix mismatch (internal identifier only); the "venue likely wrong" half is speculative and unverifiable. The safe, defensible fix is to rename the key to pieczynski_triplet_2005 to remove the contradiction, NOT to change the year/venue to 2002 (which the evidence does not establish). | medium -> info (downgrade)

### ✅ [LOW] Schur complement used without citing the available Matrix Analysis reference

`paper/appendix/B_h5_derivation.tex:74` — statut : confirmed (1 vote(s)) — catégorie : bibliography/missing-citation

Appendix B derives a conditional covariance 'by the Schur complement of...' but provides no citation, while horn_matrix_2013 (Horn & Johnson, Matrix Analysis) is defined in the .bib yet never cited (see dead-entries finding). The Schur-complement / Gaussian-conditioning fact is exactly what that reference supports. Adding the citation both fixes the dead entry and supports the algebraic step.

**Preuve :** appendix/B_h5_derivation.tex:74 '...is obtained by the Schur complement of'. horn_matrix_2013 defined at paper.bib:213 but '(no usage)' in grep. Compare appendix/E_joseph.tex:54 which does cite a matrix identity via `\cite[Sec.~5.3]{anderson_optimal_1979}`.

**Suggestion :** Add `\cite{horn_matrix_2013}` (optionally with a section locator) at the Schur-complement step in appendix/B_h5_derivation.tex:74.

**Ajustement de sévérité (vérificateurs) :** none — low is correctly calibrated

### ✅ [LOW] Incomplete venue metadata for conference/report entries (no pages, address, editors, DOIs)

`paper/paper.bib:83-89` — statut : confirmed (1 vote(s)) — catégorie : bibliography/metadata

Several entries are thin on metadata relative to the rest of the bibliography. pieczynski_triplet_2002 (@inproceedings) has no pages, address, publisher, or DOI. Most journal entries also lack DOIs, while huang_extended_2017 is the only entry carrying a DOI — an inconsistency in completeness across the bibliography. IEEEtran does not require DOIs, so this is informational, but the missing conference location/pages for the triplet entry is a genuine gap. No empty-field or missing-required-field warnings were emitted by bibtex (paper.blg shows warning$ -- 0), so nothing is broken at compile time; this is purely metadata completeness.

**Preuve :** paper.bib:83-89 pieczynski_triplet_2002 has only author/title/booktitle/year. Only huang_extended_2017 (paper.bib:233) has a doi field. paper.blg line 53: 'warning$ -- 0' (no bibtex warnings).

**Suggestion :** For consistency, either add DOIs uniformly or omit them uniformly; at minimum fill in pages/address for pieczynski_triplet_2002 (and pieczynski_pairwise_2003 already has full journal coords, so the asymmetry is visible).

**Ajustement de sévérité (vérificateurs) :** none — already correctly rated low/info; the inproceedings metadata gap is a real but minor completeness issue and the DOI-consistency point is genuinely informational, both consistent with the assigned low severity

### ✅ [INFO] Over-protected title braces vs. unprotected ones — inconsistent capitalization protection

`paper/paper.bib:221-234` — statut : confirmed (1 vote(s)) — catégorie : bibliography/style

Capitalization protection with {} is applied inconsistently. huang_extended_2017 wraps the ENTIRE title in braces (paper.bib:226-227), which is heavier than needed and prevents IEEEtran from applying its sentence-case style; other entries protect only proper nouns/acronyms ({G}aussian, {EM}, {M}arkov). This is a style consistency issue, not a compile error. Note IEEEtran.bst already down-cases titles, so most titles correctly show only initial-letter protection; the fully-braced Huang title stands out as the lone exception.

**Preuve :** paper.bib:226 `title = {{Extended Reconstructed Sea Surface Temperature, Version 5 (ERSSTv5): ...}}` (whole title braced) vs. paper.bib:118 `{Maximum Likelihood from Incomplete Data via the {EM} Algorithm}` (selective protection). Rendered difference visible in paper.bbl:103 (Huang keeps full Title Case) vs. paper.bbl:77 (Dempster sentence-cased).

**Suggestion :** For style consistency, brace only the acronym in the Huang title (e.g. {ERSSTv5}) and let IEEEtran style the rest, OR document that you intentionally preserve dataset names verbatim. Low priority.

### ❌ [INFO] Page/section locator used in only one citation (\cite[Sec.~5.3]{...})

`paper/appendix/E_joseph.tex:54` — statut : refuted (1 vote(s)) — catégorie : bibliography/style

appendix/E_joseph.tex:54 is the only citation in the paper that uses an optional locator argument: `\cite[Sec.~5.3]{anderson_optimal_1979}`. Elsewhere book references (e.g. bar-shalom_estimation_2001 at sections/03_filtering.tex:328, 06_experiments.tex:149) are cited without page/section locators even where a specific result is invoked. Not an error — just a minor citation-style inconsistency. With \bibliographystyle{IEEEtran} and plain \cite, locators render as '[ref, Sec. 5.3]', which is fine.

**Preuve :** Only locator usage: appendix/E_joseph.tex:54. Other book cites bare: sections/03_filtering.tex:328 and sections/06_experiments.tex:149 (`\cite{bar-shalom_estimation_2001}`).

**Suggestion :** Optionally add section/page locators to the other textbook citations that point to a specific theorem/identity, for uniformity. Cosmetic.

**Ajustement de sévérité (vérificateurs) :** n/a — refuted as not real (factual premise false); was already severity=info

## Hygiène LaTeX & artefacts

_Audited LaTeX hygiene and compilation artifacts for paper/ (gitignored, so all files are working-tree only, not committed). Read paper/paper.tex, paper/macros.tex, full paper/paper.log; inventoried paper/ root, figures/, figures/generated/; cross-checked all \\includegraphics and \\input targets; compared paper.tex title/author against CITATION.cff and README.md; and compared PDF timestamp (2026-05-07 06:09) against newest sources. KEY NEW FINDINGS not in the prior audit memo: (1) author/title mismatch — the paper is single-author Derrode (Pieczynski appears nowhere in the sources) yet CITATION.cff and README list two authors and a different title; (2) three superseded top-level figure pairs (fig01_data, fig03_em_restarts, fig04_regime_trace, pdf+png) are never referenced; (3) generated tab_bic.tex is produced but never \\input; (4) 24 stray test*.{aux,log,pdf}+test_table.* preview compiles from /tmp clutter paper/; (5) \\hlmath macro is dead, \\ph placeholders survive only as harmless \\IfFileExists fallbacks. COMPILATION HEALTH: clean — NO undefined references, NO undefined citations, NO multiply-defined labels, NO rerun warnings; only 5 minor Overfull hboxes (all <40pt) and cosmetic font/class notices (IEEEtran single-column attention, TS1 textmu fallback, bm \\pmb fallback). The PDF is up to date with the current sources. All 5 referenced graphics and all 8 referenced tables exist. NOTE: items already flagged in the prior audit (abstract claims, K·s>=q+s, notation collisions, \\yN vs \\yn, dead macros generally) were deliberately not re-reported._

### ❌ [HIGH] Author list and paper title disagree across paper.tex, CITATION.cff and README (single-author paper vs two-author citation)

`CITATION.cff / README.md / paper/paper.tex:paper.tex:32-40; CITATION.cff:42-52; README.md:5-7` — statut : refuted (2 vote(s)) — catégorie : metadata-consistency

The LaTeX paper is single-author: paper.tex declares only \author{St\'ephane~Derrode} (lines 35-40), and the string 'Pieczynski' appears NOWHERE in any paper source (sections/, appendix/, *.tex) — confirmed by grep. But CITATION.cff's preferred-citation (the accompanying paper) lists TWO authors, Derrode AND Wojciech Pieczynski (CITATION.cff lines 44-52), and README.md line 7 says 'Stéphane Derrode & Wojciech Pieczynski'. The TITLES also disagree three ways: paper.tex title is 'Exact IMM Filtering and Semi-Supervised EM Estimation for Gaussian Jump Markov Systems' (lines 32-33); CITATION.cff preferred-citation title is 'On Fast Optimal Filtering in Gaussian Switching Systems' (line 43); README line 6 is 'On fast optimal filtering in Gaussian switching systems'. A reader/reviewer cannot tell who the authors are or what the paper is actually called.

**Preuve :** paper.tex:35 `\author{St\'ephane~Derrode%` (sole author); grep 'Pieczynski' over paper sources -> 'NOT mentioned anywhere in the paper sources'. CITATION.cff:43 title 'On Fast Optimal Filtering in Gaussian Switching Systems'; CITATION.cff:45-52 two given-names Stéphane / Wojciech. paper.tex:32 'Exact IMM Filtering and Semi-Supervised EM Estimation for Gaussian Jump Markov Systems'.

**Suggestion :** Decide the canonical author list and title and make paper.tex, CITATION.cff (preferred-citation) and README.md agree. If Pieczynski is a co-author, add him to paper.tex \author; if the submission is solo, drop him from CITATION.cff/README. Pick one title string and reuse it verbatim.

**Ajustement de sévérité (vérificateurs) :** If reframed to the only genuine residual issue (software metadata attribution vs. vendored single-author manuscript, plus the companion's 4-author .bib entry being rendered as 2 authors in CITATION.cff/README), this would be at most LOW/INFO, not HIGH.

### ✅ [LOW] Three top-level figure pairs in figures/ are never referenced by the paper (orphan/stale assets)

`paper/figures/:figures/fig01_data.{pdf,png}, figures/fig03_em_restarts.{pdf,png}, figures/fig04_regime_trace.{pdf,png}` — statut : confirmed (1 vote(s)) — catégorie : orphan-assets

The figures actually used by the paper all live in figures/generated/ (5 PDFs via \includegraphics, all confirmed present). The three older committed pairs at the top of figures/ — fig01_data.pdf/.png, fig03_em_restarts.pdf/.png, fig04_regime_trace.pdf/.png (dated Apr 20-27, ~0.6 MB total) — are not \includegraphics'd or \input anywhere (grep on exact filenames returns nothing). They appear to be an earlier figure set superseded by figures/generated/. They bloat the directory and can confuse a co-author about which figure is current. Note the whole paper/ tree is gitignored, so this is working-tree hygiene, not repo pollution.

**Preuve :** grep 'fig01_data|fig03_em_restarts|fig04_regime_trace' over sections/ appendix/ *.tex -> 'CONFIRMED: none of the 3 top-level figure pairs are \includegraphics'd'. ls figures/ shows them dated 2026-04-20/04-27 while figures/generated/*.pdf are 2026-05-06/07.

**Suggestion :** Delete the three superseded fig01/fig03/fig04 pairs (or move them to an archive/ subfolder) so only figures/generated/ assets remain. Confirm with the author first in case they are kept intentionally as source-of-truth originals.

**Ajustement de sévérité (vérificateurs) :** none — 'low' is correctly calibrated for an untracked, non-compiled, no-correctness-impact hygiene issue

### ✅ [LOW] Generated table tab_bic.tex is produced but never \input by any section

`paper/figures/generated/tab_bic.tex:tab_bic.tex` — statut : confirmed (1 vote(s)) — catégorie : orphan-assets

The figure/table generator emits figures/generated/tab_bic.tex (104 bytes, regenerated 2026-05-07 06:09), but no section or appendix \input's it — grep 'tab_bic' over sections/ appendix/ *.tex returns nothing, whereas the other 8 generated tables are all \input. Either a BIC table was dropped from the manuscript and the generator wasn't updated, or a planned table is missing from the text.

**Preuve :** grep 'tab_bic' sections/ appendix/ *.tex -> 'NOT referenced (orphan generated file)'. All 8 other tab_*.tex are \input (sections/06_experiments.tex, sections/07_real_data.tex).

**Suggestion :** Either remove tab_bic from the figure-generation script (if BIC was intentionally cut), or add the missing \input{figures/generated/tab_bic.tex} where the BIC comparison is discussed.

**Ajustement de sévérité (vérificateurs) :** none — "low" is correctly calibrated (real generator/build inconsistency, but no PDF/correctness impact)

### ✅ [LOW] 24 throwaway test*.{aux,log,pdf} + test_table.* preview compiles sit in paper/ (working-tree clutter)

`paper/test2.{aux,log,pdf} … test8.* and test_table.*:paper/test2.pdf, test3.*, test4.*, test5.*, test6.*, test7.*, test8.*, test_table.*` — statut : confirmed (1 vote(s)) — catégorie : build-clutter

paper/ contains 8 sets of test artifacts (test2..test8 and test_table), each with .aux/.log/.pdf. Their .log headers show they were compiled from /tmp/testN.tex (e.g. '**/tmp/test2.tex'), so the SOURCES are not in the repo — only the outputs accidentally landed in paper/. Each is a 1-page article-class throwaway (e.g. 'Output written on test2.pdf (1 page, 22698 bytes)'), i.e. snippet/preview experiments, not part of the manuscript. They are not git-tracked (paper/ is fully gitignored), so this is local clutter rather than committed garbage, but it pollutes the manuscript folder and ~0.3 MB.

**Preuve :** test2.log: '**/tmp/test2.tex' and 'Document Class: article'; ls paper/ shows test2..test8 + test_table .aux/.log/.pdf; 'no matches found: test*.tex' (no sources present). git ls-files for these returns empty (untracked).

**Suggestion :** Delete all paper/test2..test8.* and paper/test_table.* — they are stray preview outputs from /tmp experiments with no source and no role in the build.

**Ajustement de sévérité (vérificateurs) :** none — 'low' is correctly calibrated (gitignored working-tree clutter, no compile/reference impact)

### ✅ [INFO] \hlmath highlight macro is defined but never used in any section (dead draft macro)

`paper/macros.tex:macros.tex:8-11` — statut : confirmed (1 vote(s)) — catégorie : dead-code

macros.tex defines \hlmath (yellow \colorbox highlight, lines 8-11) intended for draft highlighting. It is not used anywhere in the manuscript: grep 'hlmath' over sections/ and appendix/ returns nothing (the only hit is the definition line in macros.tex itself). It is a leftover drafting aid. (Separately, \ph on macros.tex:86 IS still 'used' 3x but only inside \IfFileExists{...}{}{\ph{...}} fallback branches at sections/06_experiments.tex:184,259,332 — since all figures exist, the yellow placeholders never render in the PDF, so this is harmless but worth being aware of.)

**Preuve :** grep 'hlmath' sections/ appendix/ -> empty. macros.tex:8 `\newcommand{\hlmath}[1]{%`. sections/06_experiments.tex:182-184 shows \ph only in the missing-file branch of \IfFileExists.

**Suggestion :** Remove the unused \hlmath definition (and the xcolor dependency it pulls in via macros.tex:7 if nothing else needs color — note \ph also uses \colorbox, so xcolor must stay while \ph exists). For camera-ready, consider also removing the \ph placeholder scaffolding once figures are final.

**Ajustement de sévérité (vérificateurs) :** none — severity 'low' is well-calibrated; the desc correctly notes these are untracked (paper/ gitignored) local clutter, not committed garbage. Minor: stated '~0.3 MB' is actually 388K, an immaterial approximation.

### ✅ [INFO] Five minor Overfull \hbox warnings (all <40pt); no undefined refs, no multiply-defined labels

`paper/paper.log:paper.log:893-913, 952-959, 1005-1009` — statut : confirmed (1 vote(s)) — catégorie : typesetting

The compile is clean on the important axes: no 'Reference undefined', no 'Citation undefined', no 'multiply-defined labels', no 'rerun' / 'labels may have changed' (grep returned none), 0 Underfull hbox, 0 Overfull vbox. There are 5 Overfull \hbox warnings, all minor: 32.2pt (03_filtering lines 158), 35.5pt (03_filtering 30-33), 38.8pt (03_filtering 145-151), 24.7pt (06_experiments 215-221), and 2.5pt (appendix B 30). These are inline-math lines that run slightly past the margin — cosmetic, but the ~32-39pt ones are visibly into the margin in a one-column layout.

**Preuve :** paper.log:893 'Overfull \hbox (32.19382pt too wide) in paragraph at lines 158--4'; :900 '(35.45659pt...30--33)'; :908 '(38.76222pt...145--151)'; :952 '(24.71945pt...215--221)'; :1005 '(2.48752pt...30--4)'. grep 'undefined|multiply.defined' -> none.

**Suggestion :** Optionally fix the three ~32-39pt overfulls in §3 by rewording or allowing line breaks in the long inline expressions (e.g. wrap conditional-expectation expressions in \(...\) with explicit \allowbreak, or restructure the sentence). The 2.5pt one is negligible.

**Ajustement de sévérité (vérificateurs) :** None — severity 'info' is correct. It is a real but purely cosmetic dead-macro hygiene item with zero functional impact.

### ✅ [INFO] Cosmetic font/class notices: IEEEtran single-column 'ATTENTION', TS1 textmu fallback, bm \pmb fallback

`paper/paper.log:paper.log:25, 940-942, 260-262, 1045` — statut : confirmed (1 vote(s)) — catégorie : typesetting

Several non-fatal notices that are expected given the setup but worth a one-line awareness: (1) IEEEtran prints '** ATTENTION: Single column mode is not typically used with IEEE publications.' (paper.log:25) because paper.tex:5 uses options [journal,onecolumn] — fine for a preprint/arXiv submission but will need [journal] two-column for an actual IEEE journal submission. (2) 'Font shape TS1/ptm/m/sc undefined ... for symbol textmu' (paper.log:940-942) — a µ (micro) sign in a generated table (figures/generated/tab_filter_M1.tex) falls back to upright; check the µ renders correctly. (3) 'Package bm Info: No bold for ...' (paper.log:260-262) and 'Some font shapes were not available, defaults substituted' (paper.log:1045) — Times has no bold-small-caps / certain math bolds, so bm uses \pmb; cosmetic only.

**Preuve :** paper.log:25 single-column ATTENTION; paper.log:940 'Font shape `TS1/ptm/m/sc' undefined' / :942 'for symbol `textmu''; paper.log:260 'Package bm Info: No bold for \OMX/cmex/m/n, using \pmb.'; paper.tex:5 `\documentclass[journal,onecolumn,a4paper,11pt]{IEEEtran}`.

**Suggestion :** No action needed for a preprint. Before IEEE journal submission, switch to two-column IEEEtran and re-check the overfull boxes and the µ glyph in the filter tables; if µ matters typographically, load textcomp/siunitx or use \mu in math mode in the generator.

**Ajustement de sévérité (vérificateurs) :** none — 'low' is correctly calibrated for an untracked, gitignored orphan generated file that does not affect the compiled PDF

### ✅ [INFO] Trailing-whitespace-only line in preamble (paper.tex:22)

`paper/paper.tex:paper.tex:22` — statut : confirmed (1 vote(s)) — catégorie : hygiene

Line 22 of paper.tex (between \usepackage{hyperref} and \input{macros}) is a line containing only a trailing space. Purely cosmetic, but it is the one trailing-whitespace line in the master file and trivially removable.

**Preuve :** grep -n ' $' paper/paper.tex -> '22: ' (single blank-with-space line).

**Suggestion :** Strip the trailing space / make line 22 empty.

---
_Généré automatiquement ; raisonnements complets dans `audit/raw/04c-extracted.json`._