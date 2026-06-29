# Vague 2 — Audit du code périphérique

Workflow `audit-code-periphery` (run `wf_61178f24-945`), terminé le 2026-06-11.
Méthode : 5 finders + vérification adversariale (2 votes high/medium, 1 low/info).
Détails complets : `raw/02-code-periphery-result.json`.

**Bilan : 89 trouvailles — 85 confirmées** (2 critical, 12 high, 31 medium, 29 low, 11 info), 4 réfutées.

## Trouvailles majeures (critical + high confirmées)

- **[CRITICAL] Le mode imm_general n'est jamais testé contre une référence — le seul test de correction est une comparaison de MSE** — `tests/test_gss_filter.py:495-507`
- **[CRITICAL] Les variantes V1/V2/V3 de e3_bw_em.py passent des contraintes 'b'/'a' qui sont des no-op silencieux : les 4 variantes sont devenues identiques** — `scripts/e3_bw_em.py:65-70`
- **[HIGH] QThread détruit pendant qu'il tourne — crash fatal Qt à la fermeture** — `prg/gui/main_window.py:728-733, 1855-1858`
- **[HIGH] begin_simulation/load_external ne purgent pas filter_E_xs/Var_xs/pis/log_lik → matrice de confusion silencieusement fausse** — `prg/gui/session_state.py:66-70, 90-95`
- **[HIGH] StochasticMatrixWidget : la matrice P uniforme par défaut est INVALIDE pour K=3 et K=6 (tolérance 1e-6 vs affichage 6 chiffres significatifs)** — `prg/gui/matrix_widget.py:394, 422, 474-475`
- **[HIGH] h5_exact n'est jamais testé sur un modèle qui satisfait (H5) — le filtre 'exact' tourne uniquement hors de son hypothèse** — `tests/test_gss_filter.py:362, 419`
- **[HIGH] Filtre, simulateur et apprentissage testés exclusivement en K=2, q=1, s=1 — tout est scalaire** — `tests/test_gss_filter.py:1-507 (et test_gss_simulator.py, test_supervised.py, test_semi_supervised.py)`
- **[HIGH] _fast_logpdf / _precompute_gaussian_logpdf jamais validés contre scipy ; fallback eigen mort-né** — `prg/filter/gss_filter.py:1050-1105`
- **[HIGH] La PSD du Σ_W joint après EM/projection AB n'est jamais testée ; aucun round-trip apprentissage → GSSParams → filtre** — `tests/test_semi_supervised.py:330-335`
- **[HIGH] fill_placeholders réécrit le .tex en place et détruit les \ph{} : les chiffres narratifs du papier sont désormais figés et inrafraîchissables** — `prg/experiments/fill_placeholders.py:297-304`
- **[HIGH] Labels de régime ENSO dérivés de l'ONI : look-ahead d'un mois et circularité avec l'observable Y, biaisant le test (H5) E1 vers la non-rejection** — `scripts/build_enso_csv.py:104-107`
- **[HIGH] Protocole EM : défauts du script (100 runs, n_inits=10, N jusqu'à 5000) ≠ CSV commis localement (10 runs, n_inits=5, N∈{500,2000}) ≠ em_run.log (30 runs, 3 modèles)** — `prg/experiments/run_em.py:69-75`
- **[HIGH] Le job 'Security audit' (pip-audit) échoue à chaque exécution hebdomadaire depuis au moins 3 semaines** — `.github/workflows/audit.yml:28`
- **[HIGH] Le parcours d'installation documenté (README quick-start et Makefile) est cassé sur machine vierge : pytest plante au démarrage sans PyQt6** — `Makefile:35-45 (et README.md:111-137, 668-671)`

## GUI cœur (main_window + modules extraits)

_Audit intégral du cœur GUI : prg/gui/main_window.py (2214 lignes lues en entier), workers.py, session_state.py, dialogs.py, diagnostics.py, avec vérification croisée des symboles dans param_panel.py, plot_panel.py, main.py et prg/filter/gss_filter.py. Le split récent est globalement propre — aucun import cassé ni circulaire, tous les symboles extraits correspondent à leurs sites d'appel (vérifié par grep) — mais il a laissé des résidus : constantes de couleurs mortes re-hardcodées dans dialogs.py, et un fort couplage du worker aux attributs privés de GSSFilter avec dégradation silencieuse. Les points les plus graves sont (1) un crash fatal Qt possible (« QThread: Destroyed while thread is still running ») car ni Cancel ni closeEvent n'attendent les workers parentés à la fenêtre, et (2) _SessionState qui ne purge pas filter_pis/E_xs au re-Simulate/Load-CSV, produisant une matrice de confusion silencieusement fausse mêlant nouvelles vérités-terrain et anciens postérieurs. S'y ajoutent un contournement du Cancel par la touche Échap qui fait sauter toute la protection anti-double-lancement, une incohérence mathématique de centrage (global vs intra-régime) dans la standardisation des innovations qui biaise les badges S/K, un vecteur d'exécution de code via np.load(allow_pickle=True) sur les fichiers de session, et la syntaxe PEP 758 « except A, B: » qui pince le module à Python ≥ 3.14. Ljung-Box (ddl, p-value, ACF) et Jarque-Bera sont corrects ; π_∞ est correct mais sans garde sur la valeur propre et dupliqué avec le filtre._

### ✅ [HIGH] QThread détruit pendant qu'il tourne — crash fatal Qt à la fermeture

`prg/gui/main_window.py:728-733, 1855-1858` — statut : confirmed (2 vote(s)) — catégorie : threads

_cancel_active_workers() appelle requestInterruption() + quit() mais ne fait jamais wait() (« We let it die alone »), et closeEvent() ne fait pas mieux. Les workers sont créés avec parent=self (main_window.py:763 et 843-849) : quand la fenêtre est détruite, Qt détruit l'objet QThread alors que le thread OS tourne peut-être encore (le check isInterruptionRequested n'a lieu que toutes les 256 itérations, et un seul step de filtre peut être long). Scénario concret : Cancel sur un long filtrage (le thread continue en arrière-plan, _filter_worker est mis à None donc plus jamais attendu), puis fermeture immédiate de la fenêtre → « QThread: Destroyed while thread is still running » → abort. quit() est par ailleurs un no-op ici (run() surchargé, pas de boucle d'événements).

**Preuve :** if w.isRunning():
    w.requestInterruption()
    w.quit()
    # Don't block the UI: ... We let it die alone.
setattr(self, attr, None)
...
def closeEvent(self, event):
    self._cancel_active_workers()
    self._save_settings()

**Suggestion :** Dans closeEvent, garder une liste des workers « détachés » et faire w.wait(timeout) avant super().closeEvent ; ou déparenter le thread (setParent(None)) et le connecter à QThread.finished→deleteLater pour qu'il se détruise lui-même une fois terminé.

**Ajustement de sévérité proposé par les vérificateurs :** Keep high, with one mitigating note: the abort fires after _save_settings() has already run in closeEvent, so no user data/settings are lost — the harm is a SIGABRT crash on a common action (closing during a long filter/simulation) and a broken restart-with-new-preset flow (app dies instead of reopening). If the team weighs shutdown-time crashes lower, medium-high would also be defensible, but the deterministic-leaning nature of the race justifies high for a GUI app. | high → medium: crash fatal réel et reproductible (qFatal/abort), mais uniquement au moment du teardown de la fenêtre (quitter ou restart), après sauvegarde des settings, sans perte ni corruption de données; de plus le dialogue d'attente modal rend le scénario « fermer la croix en plein run » impossible tel que décrit, les chemins restants étant Cmd+Q en plein run (fiable) et Cancel+fermeture rapide (race sub-seconde).

### ✅ [HIGH] begin_simulation/load_external ne purgent pas filter_E_xs/Var_xs/pis/log_lik → matrice de confusion silencieusement fausse

`prg/gui/session_state.py:66-70, 90-95` — statut : confirmed (2 vote(s)) — catégorie : état incohérent / statistiques

begin_simulation() et load_external() remettent innovations=None mais laissent filter_pis (et E_xs/Var_xs/log_lik) du run précédent. Or _on_regime_diag (main_window.py:1883) passe pis=self._state.filter_pis avec les rs de la NOUVELLE simulation — le commentaire « None if filter not run » n'est vrai qu'avant le premier filtrage. Après un Simulate ou un Load CSV, le bouton « Regime diagnostics » (toujours/ré-activé) affiche une matrice de confusion comparant les vrais régimes du nouveau jeu de données aux π_n de l'ancien run. Si les N diffèrent, zip() tronque silencieusement (dialogs.py:494) et accuracy=trace/N utilise le mauvais N ; si N est identique (cas courant, N=1000), le résultat est plausible mais entièrement faux.

**Preuve :** def begin_simulation(self, params, signature):
    self.params = params
    self.params_signature = signature
    self.innovations = None  # filter result no longer matches
# ← filter_E_xs / filter_Var_xs / filter_pis / filter_log_lik non purgés

# main_window.py:1882-1883
pis = self._state.filter_pis  # None if filter not run

**Suggestion :** Ajouter une méthode _SessionState.clear_filter_results() (innovations + les 4 champs filter_*) et l'appeler dans begin_simulation, load_external et reset.

**Ajustement de sévérité proposé par les vérificateurs :** Keep high. Although the bug is confined to a GUI diagnostics dialog (core filter math, CLI, and paper results unaffected), it silently displays a plausible but entirely false confusion matrix/accuracy under default settings (auto-filter off) on routine workflows (Simulate, Load CSV), and the zip() truncation hides even the N-mismatch tell. Silent plausible-wrong scientific output in a research tool justifies high; medium would be the floor if one weights GUI-only scope heavily.

### ✅ [MEDIUM] Standardisation des innovations : centrage global incohérent avec le centrage intra-régime du reste du code

`prg/gui/diagnostics.py:110-125` — statut : confirmed (2 vote(s)) — catégorie : statistiques

_standardise_innovations construit S = Σ w_jk [Γ(j,k) + δδᵀ] avec δ_jk = μ_Y,jk − μ_marg (moyenne GLOBALE). Mais l'innovation du filtre est ν_n = y_n − E[y_n | passé] : la prédiction conditionne sur le régime précédent (cf. gss_filter.py:624, mean_jk = μ_Y,jk + M_t(y_prev − μ_Y(j))). _InnovHistDialog utilise d'ailleurs le centrage intra-régime-précédent δ̃_jk = μ_jk − Σ_k' w_jk'/π_j · μ_jk' (dialogs.py:129-142) pour le mélange théorique. Le centrage global ajoute à S la variance inter-régimes des moyennes, terme que le filtre élimine déjà via π_n : S est surestimée dès que les moyennes de Y diffèrent entre régimes → ν̃ sous-dispersées (Var<1), badges S/K biaisés (kurtosis tirée vers le négatif), une mauvaise calibration peut être masquée en « ✓ ».

**Preuve :** mu_marg += float(mix_w[j, k]) * mu_Y_jk[j][k]   # moyenne globale
...
delta = mu_Y_jk[j][k] - mu_marg
S += w * (Gamma[j][k] + delta @ delta.T)
# vs dialogs.py:131-139 : prev_mean_j_i = Σ_k' w[j,k']·μ[j][k'] / π_j ; delta = μ[j][k] − prev_mean_j_i

**Suggestion :** Centrer δ̃_jk dans le régime précédent j : δ̃_jk = μ_Y,jk − (Σ_k' P_jk' μ_Y,jk')  et  S = Σ_j π_∞(j) Σ_k P_jk [Γ_jk + δ̃ δ̃ᵀ], cohérent avec la densité affichée dans le dialogue d'histogrammes.

**Ajustement de sévérité proposé par les vérificateurs :** Downgrade medium → low. The core inconsistency and the overestimation of S are real, but the headline impact (biased S/K badges, masked calibration verdicts) does not materialise for scalar observations because the shape diagnostics are scale-invariant, and no Var≈1 check exists in the GUI. Residual impact: internal inconsistency between diagnostics.py and dialogs.py, incorrect docstring claim, and a rotation/mixing effect on badges only when s ≥ 2. | medium → low. La sur-estimation de S est réelle et le correctif proposé est juste, mais le seul consommateur des innovations standardisées (_shape_diagnostics) est invariant par échelle/affine par composante: badges S/K strictement inchangés pour s=1, effet de second ordre sans signe garanti pour s>1, et aucune métrique de variance de ν̃ n'est affichée — le scénario de masquage d'une mauvaise calibration n'est pas atteignable dans le code actuel.

### ✅ [MEDIUM] Échap sur _WaitDialog contourne le Cancel : la modalité saute, double-lancement et corruption d'état possibles

`prg/gui/dialogs.py:662-697` — statut : confirmed (2 vote(s)) — catégorie : threads / robustesse

_WaitDialog masque le bouton de fermeture (WindowCloseButtonHint=False) mais ne neutralise pas la touche Échap : QDialog.reject() ferme le dialogue SANS appeler on_cancel. Le worker continue, _wait_dlg reste non-nul, et surtout la protection par modalité disparaît alors que _on_filter ne désactive PAS le bouton Simulate (main_window.py:830-838). Scénario : filtrage long → Échap → clic Simulate (N petit) → store_data(nouvelles données) → le filtre finit → _on_filter_finished passe les gardes sender/has_data et superpose les résultats de l'ancien run aux nouvelles données ; si les N diffèrent, xs−E_xs lève une exception de broadcast en plein slot. Toute la machine anti-double-lancement repose sur la seule modalité du dialogue.

**Preuve :** self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
...
btn_cancel.clicked.connect(on_cancel)
btn_cancel.clicked.connect(self.reject)
# aucun override de reject()/keyPressEvent → Esc ferme sans on_cancel

**Suggestion :** Surcharger reject() pour appeler on_cancel (ou ignorer Échap via keyPressEvent), et désactiver _btn_simulate/_act_simulate pendant un filtrage (symétrique de ce que fait _on_simulate_single).

**Ajustement de sévérité proposé par les vérificateurs :** Keep medium. Real correctness bug with a plausible trigger (Esc is a natural reflex on a modal wait dialog; long filter runs are the very reason the Cancel button exists), producing either an uncaught exception mid-slot or silently wrong results overlaid on unrelated data — but it requires a specific user sequence, corrupts only the GUI session (no data loss, recoverable via Reset), and the proposed two-line fix (override reject() to route through on_cancel; disable _btn_simulate/_act_simulate during filtering) is straightforward. | medium (inchangée) — le crash est en réalité un abort process complet (qFatal PyQt6, pas de excepthook), ce qui conforte plutôt le haut de la fourchette medium, sans justifier high pour un GUI de recherche mono-utilisateur.

### ✅ [MEDIUM] Les signaux personnalisés « finished » masquent le signal natif QThread.finished

`prg/gui/workers.py:24, 58` — statut : confirmed (2 vote(s)) — catégorie : threads / Qt

_SimWorker.finished et _FilterWorker.finished redéfinissent le signal built-in QThread.finished() (émis par Qt quand le thread se termine). Conséquences : (1) impossible de connecter le vrai thread-finished pour un nettoyage type deleteLater/wait — c'est précisément ce qui manque au finding sur closeEvent ; (2) piège classique PyQt : un futur connect(w.finished, ...) pensé « fin de thread » ne sera jamais déclenché si run() retourne par interruption silencieuse (return sans emit, workers.py:42 et 89) ; (3) la sémantique diffère silencieusement : le signal custom n'est PAS émis en cas d'abort ou d'exception, alors que le natif le serait.

**Preuve :** class _SimWorker(QThread):
    finished = pyqtSignal(list, list, object, object)  # ns, rs, xs, ys
...
class _FilterWorker(QThread):
    finished = pyqtSignal(object, object, object, object, float)

**Suggestion :** Renommer en result_ready / results (et error inchangé), libérant QThread.finished pour le cycle de vie (connect à deleteLater, ou à un slot qui retire le worker du registre des threads détachés).

**Ajustement de sévérité proposé par les vérificateurs :** medium confirmé, mais en borne basse : c'est un piège latent (aucun dysfonctionnement actif aujourd'hui, le code n'utilise nulle part le signal natif) dont l'importance vient surtout de ce qu'il bloque le correctif du finding lifecycle/closeEvent. Si l'audit réserve medium aux bugs actifs, reclasser en low. | medium → low-medium. Pris isolément c'est un piège latent sans dysfonctionnement utilisateur actuel (plutôt low); medium se justifie uniquement comme bloqueur du correctif de cycle de vie des threads (finding closeEvent). Si les deux findings sont traités ensemble, medium est acceptable; en standalone, low.

### ✅ [MEDIUM] Annuler un filtrage détruit la simulation : on_cancel=_on_reset efface tout l'état

`prg/gui/main_window.py:840` — statut : confirmed (2 vote(s)) — catégorie : robustesse / UX

Le bouton Cancel du _WaitDialog de filtrage appelle _on_reset, qui fait _state.reset() + _plot_panel.clear() : l'utilisateur qui annule un filtrage trop long perd aussi sa simulation (données, tracés, graine, paramètres capturés) et doit tout re-simuler. Pour la simulation c'est défendable, pour le filtre c'est une perte de données disproportionnée — il suffirait d'interrompre le worker et de garder data/params.

**Preuve :** self._wait_dlg = _WaitDialog("Filtering…", on_cancel=self._on_reset, parent=self)

**Suggestion :** Introduire un _on_cancel_filter dédié : _cancel_active_workers() limité à _filter_worker, fermeture du dialogue, ré-activation de _btn_filter — sans toucher à _state.data.

**Ajustement de sévérité proposé par les vérificateurs :** none — medium is well calibrated: explicit user action required and no silent corruption, but disproportionate, potentially irreproducible data loss (random-seed runs and loaded CSV data cannot be regenerated by re-clicking Simulate) | medium (unchanged) — real forced data-loss path, but a GUI usability defect rather than a correctness/security bug; partially recoverable when seed is fixed or data came from CSV.

### ✅ [MEDIUM] _on_regime_diag utilise self._P obsolète au lieu de la matrice P éditée/capturée

`prg/gui/main_window.py:1883-1889` — statut : confirmed (2 vote(s)) — catégorie : état incohérent / statistiques

self._P n'est mis à jour que dans __init__, _load_model et _restore_session_from_npz. Si l'utilisateur édite la matrice P dans StochasticMatrixWidget puis simule, la simulation utilise le P du widget (_build_gss_params → _p_widget.get_matrix()), mais le dialogue de diagnostics de régimes reçoit P=self._P (l'ancienne valeur) : les PMF géométriques théoriques Geom(1−P_kk) et les moyennes théoriques des durées (dialogs.py:603-619) sont calculées avec la mauvaise matrice — la comparaison observé/théorique devient trompeuse.

**Preuve :** dlg = _RegimeDiagDialog(
    K=self._K,
    rs=rs,
    P=self._P,   # ← jamais resynchronisé avec _p_widget après édition
    pis=pis,
    parent=self,
)

**Suggestion :** Passer le P capturé avec les données (l'ajouter à _SessionState au moment de begin_simulation/load_external), ou à défaut self._p_widget.get_matrix().

**Ajustement de sévérité proposé par les vérificateurs :** Keep medium. It is display-only (no impact on filtering/learning results), which caps it below high, but the dialog's entire purpose is the observed-vs-theoretical comparison, and a silently wrong theoretical overlay in a research diagnostics tool is actively misleading rather than cosmetic. | Keep medium. Display-only (no impact on filtering, learning, or session persistence — session save uses the widget P), but it silently invalidates the core purpose of a scientific diagnostics dialog in a routine workflow; the only mitigant is that the legend prints the numeric q_kk, which an attentive user might notice disagrees with their edited P.

### ✅ [MEDIUM] La restauration de session ne purge ni les workers en vol ni _filter_worker.cond_moments périmé

`prg/gui/main_window.py:1356-1498, 1901-1911` — statut : confirmed (2 vote(s)) — catégorie : état incohérent

Deux problèmes liés : (1) _restore_session_from_npz fait _state.reset() sans _cancel_active_workers() — si un filtre tourne (atteignable via le trou Échap), il finira sur l'état restauré et le garde « if not self._state.has_data(): return » (lignes 869-870) laisse alors le _WaitDialog modal ouvert pour toujours ; (2) la restauration ne remet pas self._filter_worker à None : _on_innov_hist (1901-1911) lira les cond_moments (mix_w, Γ, μ_Y_jk) d'un run effectué avec d'AUTRES paramètres et superposera un mélange gaussien théorique faux aux innovations restaurées de la session.

**Preuve :** # _restore_session_from_npz : pas d'appel à _cancel_active_workers, _filter_worker inchangé
self._state.reset()
...
# _on_innov_hist:
if self._filter_worker is not None and hasattr(self._filter_worker, "cond_moments"):
    cm = self._filter_worker.cond_moments  # ← peut dater d'un run pré-session

**Suggestion :** Appeler _cancel_active_workers() en tête de _restore_session_from_npz et y mettre self._filter_worker = None ; plus durable : stocker cond_moments dans _SessionState plutôt que de le lire sur le dernier worker.

**Ajustement de sévérité proposé par les vérificateurs :** Keep medium. The stale-overlay bug is a silent correctness defect in a research diagnostic (wrong theoretical curves vs restored innovations) reachable via plain normal usage (filter, then load session), and the missing worker purge can silently corrupt a just-restored session. But it is GUI-diagnostic-only, transient, and does not affect the filter math, learning code, or saved data — not high. The "modal dialog stuck forever" sub-claim should be dropped/corrected in the write-up. | Keep medium. The stale cond_moments overlay is on a natural workflow and silently misleads scientific diagnostics; the worker-race state corruption needs the separate Escape hole. GUI-only, no crash by default, no persistent data loss — not high, but too reachable for low. The 'WaitDialog stuck forever' detail should be corrected to 'in-flight filter results corrupt the restored state' in the write-up.

### ✅ [MEDIUM] np.load(allow_pickle=True) sur les fichiers .exactIMM : exécution de code arbitraire à l'ouverture

`prg/gui/main_window.py:1764` — statut : confirmed (2 vote(s)) — catégorie : sécurité

Le chargement de session désérialise avec allow_pickle=True (rendu nécessaire par le tableau dtype=object « _filter_mode » écrit à la sauvegarde, ligne 1299). Un fichier .exactIMM forgé (reçu d'un collègue, joint à un mail…) peut exécuter du code Python arbitraire au chargement via le mécanisme pickle de numpy. Pour un outil de recherche dont les sessions sont faites pour être échangées, c'est un vecteur réel.

**Preuve :** npz = np.load(path, allow_pickle=True)
# rendu nécessaire par :
"_filter_mode": np.array(self._mode_combo.currentData() or "imm_general", dtype=object)

**Suggestion :** Encoder le mode en entier ou en bytes UTF-8 (np.frombuffer) côté sauvegarde, puis charger avec allow_pickle=False (en gardant un fallback explicite + avertissement pour les anciens fichiers).

**Ajustement de sévérité proposé par les vérificateurs :** None needed. Medium is well-calibrated: arbitrary code execution argues for higher, but it is gated behind the user deliberately opening an attacker-supplied file on a local desktop research tool, which argues for lower. The innocuous-looking .exactIMM extension and the fact sessions are meant to be exchanged legitimately keep it from being merely low. Medium is the defensible middle. | Aucun ajustement — medium est juste. RCE réelle et atteignable, mais bornée par exploitation locale + action utilisateur requise (ouvrir un fichier forgé), sans vecteur réseau ni auto-déclenchement.

### ✅ [MEDIUM] Load CSV laisse le panneau « Innovation diagnostics » et le bouton d'histogrammes dans l'état du run précédent

`prg/gui/main_window.py:1243-1251, 1744-1751` — statut : confirmed (2 vote(s)) — catégorie : état incohérent

_on_load_data (et son quasi-doublon _load_csv_from) masque _mse_frame mais PAS _innov_frame, et ne désactive pas _btn_innov_hist/_act_innov_hist. Après un filtrage puis un Load CSV : les badges Ljung-Box/Skew-Kurt de l'ancien jeu de données restent affichés à côté des nouvelles courbes (diagnostics attribuables par l'utilisateur aux données chargées), et le bouton « Innov. histograms » reste cliquable mais ne fait silencieusement rien (has_filter() est devenu False). Les overlays π_n du plot sont eux aussi seulement écrasés via clear_filter_overlay.

**Preuve :** self._sync_menu_actions()
self._mse_frame.setVisible(False)
# ← pas de self._innov_frame.setVisible(False)
# ← pas de self._btn_innov_hist.setEnabled(False)

**Suggestion :** Factoriser un _clear_filter_ui() (mse+innov frames, boutons innov/pred-Y tab) appelé par _on_reset, _on_load_data, _load_csv_from et _on_simulate_single ; au passage fusionner _on_load_data/_load_csv_from (~55 lignes dupliquées) en un _load_csv(path).

**Ajustement de sévérité proposé par les vérificateurs :** none — medium is well calibrated: misleading stale diagnostics in a research GUI plus a silently dead button, but no data corruption or incorrect computation | medium is fair (low-medium also defensible): pure GUI staleness, but the stale content is scientific diagnostics shown next to a newly loaded dataset in a research tool, so misattribution by the user is plausible; the silently dead histogram button/menu action adds confusion.

### ✅ [MEDIUM] Le worker extrait fouille 6 attributs privés de GSSFilter, avec dégradation silencieuse

`prg/gui/workers.py:105-141` — statut : confirmed (2 vote(s)) — catégorie : split / couplage

_FilterWorker.run() lit filt._mu_Y, _S_YY, _mu_Y_jk, _M_t, _Gamma, _pi_inf — des privés de prg/filter/gss_filter.py. Les gardes hasattr font qu'un simple renommage côté filtre ne casse rien visiblement : les diagnostics h5 (mélange théorique, standardisation stationnaire, onglet Predicted-Y) disparaissent juste en silence et la standardisation retombe sur la covariance échantillon sans que l'utilisateur le sache. Noter aussi que _pi_inf est lu SANS garde (ligne 135) à l'intérieur du bloc hasattr("_mu_Y_jk") — vrai aujourd'hui, fragile demain. Le calcul de M_simple/Gamma2 (formules (f)/(h) de la proposition de markovianité) duplique par ailleurs de l'algèbre du filtre dans la couche GUI.

**Preuve :** if hasattr(filt, "_mu_Y") and hasattr(filt, "_S_YY"):
    self.cond_moments["mu_Y"] = filt._mu_Y
...
mix_w = filt._pi_inf[:, None] * p.P   # pas de hasattr

**Suggestion :** Exposer une API publique côté filtre, p.ex. GSSFilter.stationary_moments() -> dict (mu_Y, S_YY, mu_Y_jk, M_t, Gamma, pi_inf), et y déplacer le calcul de M_simple/Gamma2 ; le worker ne ferait que copier le dict.

**Ajustement de sévérité proposé par les vérificateurs :** Keep medium for the coupling/encapsulation finding itself, but the description must be corrected: the failure mode on rename is loud (worker error signal, or KeyError crash at main_window.py:897/900), not silent disappearance of diagnostics; only the innovation-standardization fallback is silent-ish, and it is labeled in a tooltip ("sample S"). Note the related but distinct HIGH-severity companion bug the implicit contract already caused: GUI filter runs in imm_general mode crash with KeyError("mu_Y_jk") at main_window.py:897 today (regression from making cond_moments unconditional in the a1b8487 GUI split). | Keep medium. It is a latent fragility (no present-day defect), which argues for low, but the silent AND mislabeled degradation of scientific diagnostics plus the high likelihood of near-term filter-internal refactoring (confirmed wave-1 bug in the same file) justify medium for a research codebase.

### ✅ [LOW] Constantes _COL_X/_COL_Y/_COL_R mortes après le split, littéraux re-hardcodés dans dialogs.py

`prg/gui/main_window.py:77-79` — statut : confirmed (1 vote(s)) — catégorie : split / code mort

Les trois palettes nommées « C3 » de main_window.py ne sont plus référencées nulle part (grep sur tout prg/gui/) : les tracés ont migré vers plot_panel.py et dialogs.py, qui re-hardcodent les mêmes listes hex (dialogs.py:87 « colours », :266 « regime_colours », :482 « colours »). Résidu du découpage : trois sources de vérité divergentes pour les mêmes couleurs.

**Preuve :** _COL_X = ["#1f77b4", ...]  # hidden components   ← 0 usage
# dialogs.py:87  colours = ["#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
# dialogs.py:266 regime_colours = ["#1f77b4", "#ff7f0e", ...]

**Suggestion :** Déplacer les palettes dans un petit module partagé (p.ex. prg/gui/diagnostics.py ou un colors.py) et les importer depuis dialogs.py/plot_panel.py ; supprimer les définitions mortes.

### ✅ [LOW] Monkeypatch de QWidget.setVisible et _WaitDialog jamais détruits

`prg/gui/main_window.py:538-549, 760/840` — statut : confirmed (1 vote(s)) — catégorie : cycle de vie Qt

Deux fragilités de cycle de vie : (1) _wrap_setVisible remplace frame.setVisible par une closure Python — tout setVisible déclenché côté C++ (par le parent, par Qt) contourne le patch, et le pattern surprendra tout relecteur ; une approche signal/slot ou un override de showEvent serait plus robuste. (2) Chaque Simulate/Filter crée un _WaitDialog(parent=self) fermé par accept()/reject()/close() mais sans WA_DeleteOnClose ni deleteLater : les dialogues s'accumulent comme enfants de la fenêtre pour toute sa durée de vie (avec la lambda de progress qui en capture un par run), fuite mémoire lente sur une longue session interactive.

**Preuve :** frame.setVisible = patched  # type: ignore[method-assign]
...
self._wait_dlg = _WaitDialog("Filtering…", on_cancel=self._on_reset, parent=self)
# fermé via accept()/close(), jamais deleteLater()

**Suggestion :** Pour (2) : self._wait_dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose) ou appeler deleteLater() là où le dialogue est fermé/nullifié ; pour (1) : sous-classer QFrame avec un signal visibilityChanged ou piloter _results_scroll.setVisible explicitement aux 6 sites d'appel.

**Ajustement de sévérité proposé par les vérificateurs :** Keep [low] — correctly calibrated: no functional bug today, a bounded slow memory leak and a maintainability trap in a research GUI.

### ✅ [LOW] Load CSV ne valide pas r_n ∈ [0, K) : crash ou statistiques faussées en aval

`prg/gui/main_window.py:1215-1217` — statut : confirmed (1 vote(s)) — catégorie : validation

Les colonnes r sont parsées en int sans contrôle de plage (idem _load_csv_from:1726). Un CSV avec r=5 pour K=2 fait planter _RegimeDiagDialog (IndexError sur conf[t, p] += 1, dialogs.py:495) en plein constructeur au clic sur « Regime diagnostics » ; un r négatif wrappe silencieusement (conf[-1]) et fausse la matrice sans erreur. Les valeurs y/x ne sont pas non plus contrôlées (NaN acceptés, le filtre échouera plus tard avec un message générique).

**Preuve :** rs = [int(float(row["r"])) for row in rows]
# dialogs.py:494-495
for t, p in zip(rs_arr, r_pred):
    conf[t, p] += 1

**Suggestion :** Après parsing : if not all(0 <= r < self._K for r in rs): QMessageBox.warning(...) et abandon ; éventuellement np.isfinite check sur ys.

**Ajustement de sévérité proposé par les vérificateurs :** low (inchangée)

### ✅ [LOW] _on_save (CSV) sans gestion d'erreur d'E/S, contrairement aux autres exports

`prg/gui/main_window.py:1105-1121` — statut : confirmed (1 vote(s)) — catégorie : robustesse

mkdir() et l'écriture CSV ne sont pas enveloppés dans try/except : un répertoire data/simulated non inscriptible (lancement depuis un cwd en lecture seule — les chemins sont relatifs au cwd, autre fragilité) lève une OSError non capturée dans le slot → traceback Qt silencieux, aucun message utilisateur. _on_export_model (1097) et _on_save_session (1336) capturent, eux.

**Preuve :** output_dir = pathlib.Path("data/simulated")
output_dir.mkdir(parents=True, exist_ok=True)
...
with filepath.open("w", newline="") as fh:   # aucun try/except

**Suggestion :** Envelopper dans try/except OSError → QMessageBox.critical, comme _on_export_model ; envisager d'ancrer data/ sur la racine du dépôt plutôt que le cwd.

**Ajustement de sévérité proposé par les vérificateurs :** Keep [low]. The trigger (read-only/unwritable cwd or disk error) is uncommon for this research GUI, even though the actual failure mode under PyQt6 is an application abort rather than just a silent traceback, which puts it at the high end of low.

### ✅ [LOW] log L total ignore silencieusement les pas non finis

`prg/gui/workers.py:97-98` — statut : confirmed (1 vote(s)) — catégorie : statistiques

log_lik_total n'accumule que les res.log_lik finis : si certains pas produisent NaN/Inf (modèle mal conditionné, mode imm_general avec son bug Γ non-PSD connu), le log L affiché dans « Filter quality » — et le tooltip qui propose AIC/BIC dessus — porte sur un sous-ensemble de pas non signalé. Comparer deux modèles via ce log L devient invalide sans même un avertissement.

**Preuve :** if np.isfinite(res.log_lik):
    log_lik_total += float(res.log_lik)

**Suggestion :** Compter les pas écartés et les propager (signal finished ou cond_moments) ; afficher « log L = … (k pas non finis ignorés) » et badge warn si k>0.

**Ajustement de sévérité proposé par les vérificateurs :** none — [low] is correctly calibrated: real silent-data-integrity issue in a diagnostics display, but only triggered by degenerate/ill-conditioned models, with partial console-level warnings already present.

### ✅ [LOW] _stationary_dist : pas de vérification que la valeur propre choisie vaut ≈ 1

`prg/gui/diagnostics.py:150-160` — statut : confirmed (1 vote(s)) — catégorie : statistiques

La fonction prend argmin |λ−1| sans vérifier que ce minimum est petit : pour une matrice presque-stochastique mais invalide (ou si la validation du widget évolue), elle renvoie un π* plausible mais faux au lieu de None. Pour une chaîne réductible (valeur propre 1 multiple), eig choisit un vecteur arbitraire parmi les distributions stationnaires sans avertissement. Par ailleurs le filtre calcule son propre π_∞ (GSSParams.stationary_distribution, gss_filter.py:322) — deux implémentations qui peuvent diverger numériquement entre l'affichage « π* : … » et les poids mix_w réellement utilisés.

**Preuve :** idx = int(np.argmin(np.abs(vals - 1.0)))
pi = np.real(vecs[:, idx])
pi = np.maximum(pi, 0.0)

**Suggestion :** Ajouter if abs(vals[idx] - 1.0) > 1e-8: return None ; idéalement déléguer à GSSParams.stationary_distribution pour une source unique.

**Ajustement de sévérité proposé par les vérificateurs :** Keep [low] — it is a GUI display-only issue (the filter uses the guarded GSSParams implementation, so filtering results are never corrupted), and the adjacent validity pill partially mitigates user confusion for non-stochastic P.

### ❌ [LOW] Syntaxe « except TypeError, RuntimeError: » (PEP 758) : SyntaxError sur tout Python < 3.14

`prg/gui/main_window.py:717, 721, 726` — statut : refuted (1 vote(s)) — catégorie : portabilité

Les trois clauses except non parenthésées de _cancel_active_workers utilisent la nouvelle syntaxe PEP 758 acceptée en 3.14 (vérifié : ast.parse OK sous le 3.14.5 local). C'est cohérent avec l'exigence Python 3.14 du projet, mais (1) c'est le seul endroit du module qui pince la GUI à ≥3.14 au niveau syntaxe — l'import échoue avec SyntaxError, message peu parlant, sur 3.12/3.13 ; (2) le reste du fichier utilise le style parenthésé, donc cela ressemble plus à une coquille qu'à un choix.

**Preuve :** try:
    w.finished.disconnect()
except TypeError, RuntimeError:
    pass

**Suggestion :** Écrire except (TypeError, RuntimeError): — strictement équivalent et compatible avec toutes les versions supportées par les linters/outils.

**Ajustement de sévérité proposé par les vérificateurs :** N/A (rejected). If retained at all, downgrade to info/style: the parenthesized form is harmless and linter-friendlier, but applying it consistently would mean touching plot_panel.py and test_no_stale_refs.py too, since the unparenthesized style is a deliberate repo-wide convention.

### ✅ [INFO] Docstring de _cancel_active_workers trop optimiste sur disconnect() vs signaux déjà postés

`prg/gui/main_window.py:704-710` — statut : confirmed (1 vote(s)) — catégorie : threads

Le docstring affirme qu'un signal « déjà queued sur l'event loop sera discarded par Qt » après disconnect. Ce n'est pas garanti : une invocation queued déjà postée peut encore être délivrée après le disconnect. Le code est néanmoins sûr grâce à la double défense « if self.sender() is not self._worker » dans chaque slot (768-777, 867-868) + remise à None de l'attribut — mais c'est cette garde, pas le disconnect, qui fait le travail. À documenter correctement pour qu'un futur refactor ne supprime pas la garde sender en la croyant redondante.

**Preuve :** """Disconnecting first ensures that any `finished` / `error` / `progress`
signal already queued on the event loop will be discarded by Qt
instead of running our handlers..."""

**Suggestion :** Corriger le docstring : le disconnect réduit la fenêtre de course, la garde sender()/attribut-None est l'invariant de sûreté.

**Ajustement de sévérité proposé par les vérificateurs :** none — [info] is correct: code is behaviorally safe, the issue is documentation accuracy that could mislead a future refactor into removing the load-bearing sender() guard

### ✅ [INFO] Divers : seed re-parsée à la fin de la simulation ; tooltip Bonferroni trompeur sur les badges de forme ; état après _on_sim_error

`prg/gui/main_window.py:783-784` — statut : confirmed (1 vote(s)) — catégorie : robustesse

(1) _on_sim_finished relit la seed depuis le widget au lieu d'utiliser celle réellement passée au worker (workers.py:31, self._seed) — exact aujourd'hui grâce à la modalité, faux dès que le trou Échap est exploité (le CSV sauvegardé étiquettera la mauvaise graine). (2) Les badges Skew/Kurt décident sur des seuils fixes |S|<0.25, |K|<0.50 (lignes 1037-1042) mais leur tooltip cite la correction de Bonferroni α_per — qui ne s'applique qu'au p de Ljung-Box ; le JB p-value affiché n'entre pas dans le verdict. (3) Après une erreur de simulation, _state garde l'appariement incohérent {data de l'ancien run + params fraîchement capturés par begin_simulation} ; inoffensif car les boutons restent désactivés, mais piégeux.

**Preuve :** seed = self._parse_seed()           # relu du widget, pas du worker
self._state.store_data(ns, rs, xs, ys, seed)

**Suggestion :** (1) émettre la seed dans le signal finished ou la lire sur le worker ; (2) reformuler le tooltip des badges de forme (seuils descriptifs, pas de test multiple) ; (3) restaurer params/params_signature précédents dans _on_sim_error.

**Ajustement de sévérité proposé par les vérificateurs :** Keep [info]. All three points are real but latent (modality covers point 1 in normal use, point 2 is a tooltip-wording issue, point 3 is currently shielded by disabled buttons). If anything, point 3 is a hair stronger than stated because Save Session remains enabled after a sim error and would persist the incoherent data/params pairing — still within [info]/low.

## Widgets GUI (param_panel, plot_panel, matrix_widget)

_Audit intégral des trois widgets GUI : prg/gui/param_panel.py (691 lignes), prg/gui/matrix_widget.py (629), prg/gui/plot_panel.py (1257), avec vérification croisée contre prg/utils/h5_constraint.py, prg/filter/gss_filter.py (H5_TOL), prg/gui/main_window.py et prg/gui/main.py, plus tests numériques exécutés (round-trip :.6g, comportement numpy sur NaN/inf, CovarianceMatrix.check). Santé globale correcte : les formules AB/H5 de la GUI sont fidèles à h5_constraint.py, les gardes anti-récursion de MatrixTableWidget sont bien pensées (commentaires explicites), et la crainte de fuite mémoire matplotlib n'est PAS confirmée (Figures créées hors pyplot, clf() systématique, recréation de fenêtre au changement de K/q/s — la resynchronisation inter-widgets sur K repose entièrement sur ce restart, qui est sain côté GUI mais cassé côté CLI -K/-q/-s). Les deux problèmes les plus sérieux : un crash dur de l'application (abort PyQt6) en tapant 'nan' dans Σ_V avec la contrainte AB cochée (LinAlgError non capturée), et la matrice P par défaut invalide pour K=3/K=6 à cause du conflit entre l'affichage 6 chiffres significatifs et la tolérance de stochasticité 1e-6 (avec un message d'erreur « sum = 1 ≠ 1 »). Viennent ensuite des incohérences d'état de la contrainte AB (randomize et preset in-place déverrouillent silencieusement A,B), un badge (H5) absolu là où le filtre est relatif, et un ordre d'émission de value_changed qui expose des A,B pré-projection aux consommateurs._

### ✅ [HIGH] StochasticMatrixWidget : la matrice P uniforme par défaut est INVALIDE pour K=3 et K=6 (tolérance 1e-6 vs affichage 6 chiffres significatifs)

`prg/gui/matrix_widget.py:394, 422, 474-475` — statut : confirmed (2 vote(s)) — catégorie : validation / précision d'affichage

Le constructeur remplit P avec 1/K, mais `set_matrix` formate chaque cellule en `:.6g` et `_validate_all` re-parse le TEXTE affiché. Pour K=3 : '0.333333'×3 = 0.999999, écart = 1.0000000000287557e-06 > tolérance 1e-6 → widget invalide dès l'ouverture. Pour K=6 : 1.000002, écart 2e-6 → invalide aussi (vérifié numériquement). Pire, le message d'erreur formate la somme en `.4g`, donc affiche « Row 0: sum = 1 ≠ 1 », incompréhensible pour l'utilisateur. Le même mécanisme frappe toute matrice P chargée (preset, session, P appris par EM) dont l'arrondi à 6 chiffres déplace une somme de ligne de plus de 1e-6 — probable dès K ≥ 3 avec des valeurs quelconques.

**Preuve :** matrix_widget.py:394 `self.set_matrix(np.full((K, K), 1.0 / K))` ; :422 `item = QTableWidgetItem(f"{val:.6g}")` ; :474-475 `if abs(row_sum - 1.0) > 1e-6: self._set_valid(False, f"Row {r}: sum = {row_sum:.4g} ≠ 1")`. Test : K=3 → 0.999999 (INVALIDE), K=6 → 1.000002 (INVALIDE), message affiché « sum = 1 ».

**Suggestion :** Augmenter la précision d'affichage (ex. `:.12g`), ou élargir la tolérance à ~K·1e-6, ou re-normaliser les lignes après parsing. Formater le message avec assez de chiffres (`:.8g`) pour qu'il ne dise jamais « 1 ≠ 1 ».

**Ajustement de sévérité proposé par les vérificateurs :** Lower from high to medium. The mechanism, numerics, and useless '1 ≠ 1' message are all real, and the bug hard-blocks the GUI when triggered, but no shipped launch path actually exhibits 'invalid at open' today: all presets/models supply P values that survive the 6-digit round-trip, and the -K CLI flag is overridden by the default preset. It is a latent correctness/UX bug on edge paths (custom models without P, unlucky EM/session matrices), not a guaranteed regression for current users. | high → medium. The mechanism and the absurd error message are real and the fix is warranted (more display digits, K-scaled tolerance, or row re-normalisation), but the default/at-open scenario for K=3 and K=6 cannot occur through any existing launch path: the uniform constructor default is always replaced before the user sees it, all shipped models/presets provide a P that survives .6g rounding, and the -K CLI flag is silently overridden by the default preset. Impact is limited to user-supplied or EM-exported K>=3 models with arbitrary-precision probabilities (~12% failure chance per 3x3 matrix), which blocks Simulate/Export with a confusing message — a medium-severity usability bug, not a high-severity default breakage.

### ❌ [HIGH] Saisir 'nan' dans Σ_V avec la contrainte AB cochée lève une LinAlgError non capturée → abort de l'application

`prg/gui/param_panel.py:312-330 (+ prg/utils/h5_constraint.py:138)` — statut : refuted (2 vote(s)) — catégorie : crash / contrainte AB

`_recompute_AB` ne capture que ValueError autour de `compute_AB`. Or la première instruction de `compute_AB` est `np.linalg.cond(SV)`, qui lève `numpy.linalg.LinAlgError: SVD did not converge` si Σ_V contient un NaN (vérifié avec numpy 2.4.4). Comme `float('nan')` est accepté par la validation des cellules (la cellule n'est même pas marquée rouge, voir finding matrix_widget), taper 'nan' dans une cellule du bloc Σ_V de Σ_W(k) pendant que la case « AB constraint » est cochée fait remonter l'exception à travers le slot Qt `_on_value_changed`. PyQt6 sans `sys.excepthook` personnalisé (aucun dans prg/gui/main.py) appelle qFatal → abort() : toute la GUI meurt. À noter : 'inf' est, lui, géré par accident (cond → inf > 1e12 → ValueError capturée).

**Preuve :** param_panel.py:319-327 :
    try:
        A_new, B_new = compute_AB(
            C=F[q:, :q], D=F[q:, q:], Dt=Sw[:q, q:], SV=Sw[q:, q:],
        )
    except ValueError:
h5_constraint.py:138 : `cond_SV = np.linalg.cond(SV)` — test exécuté : `np.linalg.cond([[nan,0],[0,1]])` → `LinAlgError: SVD did not converge`.

**Suggestion :** Capturer `(ValueError, np.linalg.LinAlgError)` dans `_recompute_AB`, et/ou rejeter les valeurs non finies au niveau de `compute_AB` (np.isfinite). Installer un sys.excepthook dans prg/gui/main.py comme filet de sécurité.

**Ajustement de sévérité proposé par les vérificateurs :** n/a (not a real defect; at most a trivial/cosmetic mislabeled status message, far below 'high') | N/A — finding refuted. If retained at all, downgrade to info/style: make the except clause explicit (`except (ValueError, np.linalg.LinAlgError)`) for robustness against hypothetical numpy hierarchy changes; the claimed application abort does not occur.

### ✅ [MEDIUM] 'nan', 'inf', '1e999' passent la validation des cellules : F(k) et les vecteurs non finis sont déclarés valides

`prg/gui/matrix_widget.py:126-138, 255-262, 562-573` — statut : confirmed (2 vote(s)) — catégorie : validation

`_validate_cell` ne teste que `float(item.text())`, qui accepte 'nan', 'inf', '-inf', '1e999' (→ inf). Pour la matrice F (is_covariance=False) et pour VectorWidget (μ_z0, b_X, b_Y), il n'y a AUCUN autre contrôle : `is_valid()` reste True, la cellule reste blanche, et `get_matrix()/get_vector()` renvoient des valeurs non finies que Simulate/Filter consomment telles quelles. Σ_W est protégé indirectement (CovarianceMatrix(NaN).check() → invalide, vérifié) et P rejette nan via `v >= 0.0` ; F et les vecteurs sont les trous. C'est aussi la porte d'entrée du crash décrit dans le finding « LinAlgError » (nan dans Σ_V est marqué invalide par le check SPD mais `get_matrix()` le renvoie quand même au pipeline AB). Côté locale : `float()` est indépendant de la locale, donc une virgule française '0,5' est rejetée (cellule rouge) — comportement visible, pas de misparse silencieux.

**Preuve :** matrix_widget.py:258-262 :
    try:
        float(item.text())
        ok = True
    except ValueError:
        ok = False
Aucun np.isfinite dans le fichier. Vérifié : float('nan') → nan, float('1e999') → inf.

**Suggestion :** Remplacer le test par `v = float(text); ok = math.isfinite(v)` dans les trois widgets, et/ou faire vérifier np.isfinite dans get_matrix()/get_vector().

**Ajustement de sévérité proposé par les vérificateurs :** medium is fair, or arguably medium-low: requires a user to deliberately type 'nan'/'inf'/'1e999', and the Simulate path fails fast with a clear SimulationError (under __debug__); but the Filter path and the AB-constraint pipeline consume the non-finite values with no guard, producing NaN results or LinAlgError, and the widget reports valid state, so keeping medium is defensible. | Keep medium (low end). The validation hole and the call path are real and exactly as described, but in default (non -O) Python the failure is always surfaced as a caught, user-visible error (SimulationError from the simulator guard, LinAlgError from the filter precompute) rather than silent wrong results — so the practical harm is degraded UX/diagnostics. Silent nan propagation only occurs under python -O, where the __debug__ guards in GSSParams and GSSSimulator vanish; that residual silent path justifies keeping medium rather than dropping to low.

### ✅ [MEDIUM] Randomize 🎲 et chargement de preset in-place violent silencieusement la contrainte AB active (blocs déverrouillés + pastille « ✓ » périmée)

`prg/gui/param_panel.py:475-494 (et prg/gui/main_window.py:1143-1145, 2206)` — statut : confirmed (2 vote(s)) — catégorie : contrainte AB / cohérence d'état

`MatrixTableWidget.set_matrix` recrée tous les QTableWidgetItem avec les flags par défaut (éditables) et la couleur normale. `_StateTab._randomize` appelle `set_matrix(F_rand)` sans re-projeter ni re-verrouiller : si la case « AB constraint » est cochée, les blocs A,B deviennent éditables, contiennent des valeurs aléatoires qui violent A=ΔΣ_V⁻¹C, la pastille « ✓ A, B satisfy the AB constraint » reste affichée (set_matrix ne touche pas _constraint_label) et la case reste cochée. Le badge (H5) passe à l'ambre mais contredit la pastille. Même problème pour le chargement de preset à dimensions identiques : `_on_preset_selected` → `_load_model` → `set_state_params` sans appel à `reapply_active_constraints()` — contrairement au chemin session (`_load_session`, main_window.py:1400, qui l'appelle correctement). De plus, comme set_matrix n'émet pas value_changed (garde _building), `_on_value_changed` ne rattrape pas le coup.

**Preuve :** param_panel.py:491-494 :
    self._f_widget.set_matrix(F_rand)
    self._sigma_widget.set_matrix(Sw)
    self._update_stability_badges()
    self.constraint_toggled.emit()  # reset plots in main window
(aucun _recompute_AB) ; main_window.py:1143-1145 : `self._load_model(model); self._on_reset()` — _load_model (2181-fin) ne contient pas reapply_active_constraints.

**Suggestion :** Dans `_randomize`, appeler `self._recompute_AB()` (ou décocher la case) après set_matrix ; dans `_load_model`, appeler `self._param_panel.reapply_active_constraints()` comme le fait `_load_session`.

**Ajustement de sévérité proposé par les vérificateurs :** none — medium is correctly calibrated | Keep medium. Aggravating: silent violation of an actively-displayed invariant, trivially reachable (one click on 🎲 or a preset selection), can invalidate h5_exact filter results since the runtime warning is console-only. Mitigating: the live (H5) badge correctly turns amber on the same screen, and the state self-heals on the next manual cell edit, limiting the window to randomize/preset-load followed directly by simulate/filter.

### ✅ [MEDIUM] Le badge (H5) utilise un résiduel ABSOLU alors que le filtre warne sur le résiduel RELATIF — le commentaire « kept in sync » est faux

`prg/gui/param_panel.py:55-58, 411, 417 (vs prg/filter/gss_filter.py:270-276)` — statut : confirmed (2 vote(s)) — catégorie : contrainte AB / cohérence GUI-filtre

Le badge vert/ambre compare `‖F‖_F < 1e-6` (norme de Frobenius absolue du résiduel). gss_filter._check_h5 calcule `rel = ‖F‖_F / max(‖Z‖_F, 1)` avec Z = ΔᵀAᵀ + Σ_V Bᵀ et warne si rel > H5_TOL = 1e-6. Le commentaire en tête de param_panel (« the GUI shows a green badge in exactly the regime where mode='h5_exact' would not warn ») est donc incorrect dès que ‖Z‖_F > 1 : un modèle avec ‖F‖=5e-5 et ‖Z‖=100 affiche un badge ⚠ ambre alors que le filtre (rel=5e-7) ne warnerait pas. L'incohérence est unilatérale (badge vert ⇒ filtre silencieux, car scale ≥ 1) mais le badge — présenté comme « source of truth » dans le tooltip — sur-alerte pour tout modèle à grande échelle.

**Preuve :** param_panel.py:411 `res_norm = float(np.linalg.norm(res, "fro"))` puis :417 `if res_norm < _H5_BADGE_TOL:` ; gss_filter.py:270-272 :
    Z = Dt.T @ A.T + SV @ B.T
    scale = max(float(np.linalg.norm(Z, "fro")), 1.0)
    rel = float(np.linalg.norm(F, "fro")) / scale

**Suggestion :** Normaliser le résiduel du badge exactement comme le filtre (même Z, même max(·,1)), idéalement en factorisant ce calcul dans h5_constraint.py pour qu'il n'existe qu'à un seul endroit.

**Ajustement de sévérité proposé par les vérificateurs :** Medium is defensible but at the upper edge; low-medium would also be fair. The mismatch is strictly conservative (badge can be falsely amber, never falsely green), GUI-only, and affects no filter output — its cost is user confusion / spurious over-alerting on large-scale models plus a false "kept in sync" comment and a misleading "source of truth" tooltip. Keep medium if diagnostics trustworthiness matters for the paper's companion GUI; otherwise downgrade to low. | Lower from medium to low: one-sided GUI over-alert with the safe direction fully preserved (green badge still guarantees the filter will not warn), no effect on any numerical output, and a narrow trigger window (near-(H5) model with absolute residual in [1e-6, 1e-6*||Z||)). The only escalating factor — amber badge nudging users toward the buggy imm_general mode — is real but requires multiple conditions to line up. The proposed fix (factor the normalized residual into h5_constraint.py and use it in both places) is correct and low-cost.

### ✅ [MEDIUM] value_changed est forwardé AVANT la re-projection AB, et la re-projection n'émet jamais de value_changed → les consommateurs voient des A,B périmés

`prg/gui/param_panel.py:202-215, 304-345` — statut : confirmed (2 vote(s)) — catégorie : signaux / ordre d'émission

Ordre de connexion sur `_f_widget.value_changed` : (1) forward vers ParamPanel.value_changed (ligne 210) puis (2) `_on_value_changed` → `_recompute_AB` (lignes 212-213). Qt appelle les slots dans l'ordre de connexion, donc main_window (`_refresh_filter_button_drift_indicator`, connecté ligne 193 de main_window.py) lit les matrices PENDANT l'émission, avant que A,B ne soient re-projetés. Ensuite, `_recompute_AB` modifie A,B via `set_matrix` sous garde `_building` : aucun value_changed n'est émis pour ces nouvelles valeurs. Résultat : tout listener de value_changed reste sur l'état pré-projection jusqu'à l'édition suivante. Effet secondaire fragile : le `set_matrix` synchrone détruit (via setItem) le QTableWidgetItem en cours d'émission de itemChanged — ça ne plante pas aujourd'hui parce que les internals Qt6 ne re-déréférencent pas l'item après l'emit, mais c'est précisément le scénario décrit dans le commentaire de matrix_widget.py:264-267.

**Preuve :** param_panel.py:209-213 :
    w.validity_changed.connect(self._on_child_validity)
    w.value_changed.connect(self.value_changed)  # forward upward  ← connecté AVANT
    ...
    self._f_widget.value_changed.connect(self._on_value_changed)   ← recompute APRÈS
puis :335-340 : set_matrix + set_block_editable sous self._updating, sans ré-émission.

**Suggestion :** Connecter `_on_value_changed` AVANT le forward (ou faire le forward depuis `_on_value_changed` après la projection), ou différer la re-projection avec QTimer.singleShot(0, ...) puis émettre value_changed une fois A,B recalculés.

**Ajustement de sévérité proposé par les vérificateurs :** lower: medium -> low (only consumer is the cosmetic Filter-button drift badge; boolean wrong only in a revert-to-captured corner case; filter/simulate paths recompute fresh post-projection values; the use-after-free-shaped fragility is latent and currently harmless) | medium → low. The stale-read affects exactly one consumer, a cosmetic drift-warning badge; no numerical, capture, or persistence path reads matrices mid-emission. False negatives (missing a real drift) are mathematically impossible given the deterministic AB projection; the only failure mode is a transient/sticky spurious warning after an edit-revert, corrected by any subsequent refresh trigger. The QTableWidgetItem destruction during emission is a genuine latent fragility worth fixing (same class as the mitigation already present in matrix_widget._validate_cell), but it is non-crashing today as the finding itself concedes.

### ✅ [MEDIUM] Les options -K/-q/-s documentées sont silencieusement écrasées par le preset par défaut

`prg/gui/main.py:63-86` — statut : confirmed (2 vote(s)) — catégorie : cohérence inter-widgets / CLI

Quand `--model` n'est pas fourni, la branche else charge PRESETS[0] et réassigne K, q, s depuis ses paramètres : `python -m prg.gui.main -K 3 -q 2 -s 1` (l'usage montré dans le docstring du fichier, ligne 11) démarre en réalité avec K=2, q=1, s=1 sans aucun message. Les arguments CLI ne prennent effet que si le chargement du preset échoue. Comme l'architecture repose sur « K fixé à la construction + recréation de fenêtre » (tous les widgets — ParamPanel, StochasticMatrixWidget, PlotPanel, PredYPanel — reçoivent K/q/s au constructeur), ce chemin CLI est le seul moyen direct de démarrer à dimensions arbitraires, et il est cassé.

**Preuve :** main.py:75-83 :
    else:
        # No model specified → load the Reference preset (K=2, q=1, s=1) by default
        try:
            model = PRESETS[0].load()
            p = model.get_params()
            K, q, s = p["K"], p["q"], p["s"]

**Suggestion :** Ne charger le preset par défaut que si l'utilisateur n'a passé aucun de -K/-q/-s (comparer aux defaults argparse ou utiliser default=None), sinon respecter les dimensions demandées.

**Ajustement de sévérité proposé par les vérificateurs :** medium is defensible and I would keep it; low-medium would also be fair since this is a CLI/GUI usability defect (documented flags silently ignored) rather than a numerical-correctness bug, and an indirect workaround exists via --model with a hand-written model file. | Keep medium. It silently breaks a documented CLI feature that is the only direct route to arbitrary dimensions in a construction-time-fixed GUI architecture; mitigated (not high) because the wrong dimensions are immediately visible in the launched window — no risk of silently wrong scientific output — and a --model workaround exists.

### ✅ [LOW] StochasticMatrixWidget et VectorWidget n'ont pas la garde blockSignals de _validate_cell → ré-entrance et double émission de value_changed

`prg/gui/matrix_widget.py:447-455, 608-616 (contraste : 264-277)` — statut : confirmed (1 vote(s)) — catégorie : signaux / ré-entrance

MatrixTableWidget._validate_cell entoure setBackground/setForeground de blockSignals avec un commentaire expliquant que ces appels ré-émettent itemChanged de façon synchrone. Les deux autres widgets du même fichier font les mêmes setBackground/setForeground SANS cette garde. À chaque transition invalide→valide d'une cellule (rouge→blanc), itemChanged est ré-émis pendant le traitement : `_on_item_changed` ré-entre (la garde `_building` est False), et `value_changed` est émis deux fois par édition. Pour le widget P cela double `_update_stationary_display` et `_refresh_filter_button_drift_indicator` (main_window.py:208-209). La récursion est bornée (Qt6 ignore les setData sans changement de valeur), mais la protection est incohérente entre les trois classes du fichier.

**Preuve :** matrix_widget.py:453-454 (StochasticMatrixWidget) :
    item.setBackground(QBrush(QColor("white") if ok else _COLOUR_BAD))
    item.setForeground(QBrush(QColor("black")))
— sans blockSignals, alors que :268 (MatrixTableWidget) fait `self._table.blockSignals(True)` avec le commentaire « setBackground / setForeground emit itemChanged synchronously ».

**Suggestion :** Appliquer la même garde blockSignals (ou un flag de ré-entrance) dans les _validate_cell de StochasticMatrixWidget et VectorWidget.

**Ajustement de sévérité proposé par les vérificateurs :** Keep [low]. Real but consequence-light: double emission only on colour-transition edits, slots are idempotent UI refreshes, no item-deletion/crash path for these two widgets. Worth fixing for consistency exactly as proposed.

### ✅ [LOW] Collision de couleur : X^3 (vérité terrain) et l'overlay du filtre utilisent le même rouge #d62728

`prg/gui/plot_panel.py:124, 182` — statut : confirmed (1 vote(s)) — catégorie : couleurs / lisibilité

`colours_x[3] = "#d62728"` (palette des composantes cachées) est exactement la couleur `colour_filt = "#d62728"` de `add_filter_overlay`. Pour q ≥ 4, le sous-graphe X^3 superpose la trajectoire vraie (rouge continu) et l'estimée filtrée (rouge pointillé) + bande ±2σ rouge : illisible. Par ailleurs, vérification de la cohérence inter-panneaux demandée : la palette régimes de `add_pi_overlay` (tab10 par k) n'est référencée nulle part ailleurs (R_n est gris, diagnostics.py n'utilise aucune couleur, PredYPanel._COLOURS est défini ligne 417 mais jamais utilisé) — pas d'incohérence régime/couleur entre panneaux, seulement cette collision composante/overlay.

**Preuve :** plot_panel.py:124 `colours_x = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]` ; :182 `colour_filt = "#d62728"  # red`.

**Suggestion :** Retirer #d62728 de colours_x (ou choisir une couleur d'overlay hors palette, p.ex. noir/magenta), et supprimer PredYPanel._COLOURS mort.

**Ajustement de sévérité proposé par les vérificateurs :** Keep [low] — correctly calibrated. The collision is real but cosmetic, only reachable with custom models of q ≥ 4 (no shipped preset triggers it), and partially mitigated by differing linestyles and the legend.

### ✅ [LOW] update_plots efface (cla) les axes d'innovation mais ne restaure pas leurs labels ν^i

`prg/gui/plot_panel.py:102-103 vs 384-389` — statut : confirmed (1 vote(s)) — catégorie : axes / légendes

`update_plots` fait `ax.cla()` sur TOUTES les axes (y compris les s axes d'innovation du bas) puis ne re-étiquette que R_n, π_n, X^i et Y^i. Entre un Simulate et le prochain run du filtre, les bandes d'innovation restent des axes nus sans ylabel ni grille (alors que `_draw_empty` et `clear_innovations` les étiquettent soigneusement). Incohérence visuelle systématique après chaque simulation.

**Preuve :** plot_panel.py:102-103 :
    for ax in self._axes:
        ax.cla()
— puis aucune écriture sur self._axes[self._innov_offset + i] dans update_plots (contrairement à _draw_empty:384-389).

**Suggestion :** À la fin d'update_plots, ré-appliquer ylabel/grid/yticks sur les axes ν^i (factoriser le bloc de _draw_empty).

**Ajustement de sévérité proposé par les vérificateurs :** Keep at low — purely cosmetic inconsistency on axes that hold no data at that point in the workflow; no incorrect information shown, self-heals on the next filter run.

### ✅ [LOW] PredYPanel : chaque frappe clavier dans les spinbox déclenche un clf() + reconstruction complète de la figure (dont 2 grilles mvn 100×100 en s=2)

`prg/gui/plot_panel.py:519-524, 532-544, 561-566` — statut : confirmed (1 vote(s)) — catégorie : performance / redraws

`_n_spin.valueChanged` et chaque `QDoubleSpinBox.valueChanged` (keyboardTracking actif par défaut → un signal PAR caractère tapé) sont connectés à `_refresh_density`, qui fait `fig.clf()`, recrée tous les subplots et, pour s=2, recalcule deux grilles `multivariate_normal.pdf` 100×100 + contourf/contour ×4. Un changement de j/k (`_refresh_both`) reconstruit en plus les deux figures de trajectoire. Taper « 123 » dans un champ y^i = 3 reconstructions complètes. Aucune fuite mémoire en revanche : clf() libère les axes et les Figures sont créées hors pyplot (pas de registre global).

**Preuve :** plot_panel.py:563-566 :
    self._n_spin.valueChanged.connect(self._on_n_changed)
    ...
    for sp in self._yn_spins:
        sp.valueChanged.connect(self._refresh_density)
et :915 `self._fig_dens.clf()` à chaque appel.

**Suggestion :** Appeler `setKeyboardTracking(False)` sur _n_spin et les _yn_spins, et/ou débouncer via QTimer ; en 2D, mettre en cache la grille tant que (j,k) ne change pas.

### ✅ [LOW] Affichage en :.6g = troncature à 6 chiffres significatifs de tous les paramètres chargés — le filtre tourne sur des valeurs ≠ modèle

`prg/gui/matrix_widget.py:146, 422, 582` — statut : confirmed (1 vote(s)) — catégorie : précision / round-trip

Tous les set_matrix/set_vector formatent en `:.6g`, et les getters re-parsent le texte affiché. Tout paramètre injecté (preset, session .npz, A/B re-projetés par la contrainte AB, P appris par EM) subit donc une erreur relative jusqu'à ~5e-7 avant d'atteindre le filtre/simulateur. Conséquences concrètes : (a) la racine du bug « P uniforme invalide pour K=3/6 » ; (b) les A,B projetés en float64 puis tronqués laissent un résiduel (H5) ~1e-7·‖·‖ au lieu de ~1e-16 ; (c) un round-trip session → GUI → session dérive les paramètres.

**Preuve :** matrix_widget.py:146 `item = QTableWidgetItem(f"{mat[r, c]:.6g}")` (idem :422 et :582) ; get_matrix:135 `mat[r, c] = float(item.text())` — la valeur float64 d'origine n'est conservée nulle part.

**Suggestion :** Stocker la valeur exacte dans item.setData(Qt.UserRole, val) et la préférer au texte tant que l'utilisateur n'a pas édité la cellule, ou afficher en `:.12g`.

**Ajustement de sévérité proposé par les vérificateurs :** Raise from low to medium: it makes the default GUI state invalid for K=3 and K=6 (blocks Simulate until the user hand-edits P), can trigger false H5-bias warnings on models where the AB constraint was explicitly enabled (truncated residual ~1e-6 sits at the H5_TOL=1e-6 warning threshold), and silently degrades parameter fidelity in saved sessions and filter inputs of a tool whose purpose is exact filtering.

### ✅ [LOW] Cas limites du toggle AB : F invalide au moment du cochage → rien n'est verrouillé ni sauvegardé ; Σ_V singulier laisse des A,B verrouillés périmés

`prg/gui/param_panel.py:287-298, 310-330, 347-364` — statut : confirmed (1 vote(s)) — catégorie : contrainte AB / cas limites

(1) Si une cellule de F est invalide quand on coche la case, `_saved_A/_saved_B` restent None et `_recompute_AB` retourne avant le verrouillage : la case est cochée mais les blocs A,B restent éditables et rien ne sera restauré au décochage (les valeurs projetées restent). (2) Si Σ_V devient singulier/mal conditionné pendant que la contrainte est active, le except affiche la pastille d'erreur mais laisse les blocs A,B VERROUILLÉS sur la dernière projection valide — l'utilisateur ne peut ni les éditer ni les voir recalculés, et le message dit « Σ_V singular » même pour un simple mauvais conditionnement (cond > 1e12). Cohérence avec h5_constraint.py par ailleurs vérifiée : mêmes formules compute_AB/compute_h5_residual, mêmes découpages de blocs F et Σ_W.

**Preuve :** param_panel.py:312-314 `if F is None or Sw is None: return` (pas de lock, pas de statut) ; :327-330 :
    except ValueError:
        self._f_widget.set_constraint_status(
            "✗  AB constraint — Σ_V singular", _CONSTRAINT_ERR_STYLE)
        return  ← blocs laissés dans l'état précédent

**Suggestion :** Au cochage avec F invalide : refuser (message) ou différer ; dans le chemin d'erreur : déverrouiller les blocs A,B ou griser la case, et distinguer « singulière » de « mal conditionnée » dans le message.

**Ajustement de sévérité proposé par les vérificateurs :** none — low is correctly calibrated (GUI-only edge cases; only persistent harm is loss of the original A,B values in case 1)

### ✅ [INFO] Syntaxe PEP 758 `except A, B:` (sans parenthèses) — SyntaxError immédiate sur Python ≤ 3.13

`prg/gui/param_panel.py:412 (et plot_panel.py:291, 324)` — statut : confirmed (1 vote(s)) — catégorie : portabilité / syntaxe

Trois clauses except utilisent la nouvelle forme sans parenthèses de Python 3.14 (PEP 758) : `except np.linalg.LinAlgError, ValueError:` et `except ValueError, NotImplementedError, AttributeError:` (×2 dans plot_panel). Conforme au requires-python = ">=3.14" du pyproject (vérifié, et py_compile passe en local), mais c'est un point de rupture dure : tout collaborateur/outil sur 3.13 obtient une SyntaxError à l'import de la GUI (pas une ImportError explicable), et certains linters/formatters ne parsent pas encore cette forme. À garder en tête, surtout si le papier doit être accompagné de code reproductible.

**Preuve :** param_panel.py:412 `except np.linalg.LinAlgError, ValueError:` ; plot_panel.py:291 et :324 `except ValueError, NotImplementedError, AttributeError:`.

**Suggestion :** Aucune action obligatoire ; pour la robustesse outillage, la forme parenthésée `except (A, B):` est équivalente et universelle.

**Ajustement de sévérité proposé par les vérificateurs :** None — [info] is correctly calibrated. The code conforms to the declared requires-python >=3.14; the concern is limited to collaborators/tooling on older Pythons and reproducibility packaging, which is exactly what an informational note should flag.

## Qualité de la suite de tests

_Lecture intégrale des 12 fichiers de tests/ (3458 lignes, 219 tests collectés) croisée avec les sources qu'ils prétendent couvrir (prg/filter/gss_filter.py, prg/utils/h5_constraint.py, prg/learning/{supervised,semi_supervised}.py, prg/classes/GSSParams.py, prg/gui/{diagnostics,param_panel,workers}.py, prg/filter/main.py). La suite est de bonne facture sur l'algèbre statique (FMatrix, NoiseCovariance, propriétés de la contrainte AB, forward/backward EM validé par force brute, tests ParamPanel GUI qui vérifient de vraies valeurs) et les graines sont partout fixées. En revanche le cœur dynamique est structurellement sous-testé : le mode imm_general n'a aucune référence de correction (seule une comparaison de MSE — exactement le trou qui a laissé passer le bug Γ non-PSD de la vague 1), le mode h5_exact n'est jamais exécuté sur un modèle satisfaisant (H5), toute la chaîne filtre/simulateur/EM tourne exclusivement en K=2, q=1, s=1 malgré des modèles K=3 et q=s=2 présents dans le dépôt, et _fast_logpdf n'est jamais confronté à scipy. S'y ajoutent des tests mensongers ou tautologiques (TestOptionB compare deux exécutions identiques ; test_missing_destination_regime_raises ne raise rien ; Ljung-Box « high p-value » asserte 0≤p≤1), un test statistique du simulateur qui passe à 0.0015 de sa tolérance sur une prémisse fausse, et des gardes récentes (_compute_stationary, planchers EM) sans aucune pression de test (zéro pytest.warns dans la suite). test_no_stale_refs est un lint lexical utile mais étroit (regex minuscules, whitelist par sous-chaîne, dépendance git, syntaxe PEP 758 non portable)._

### ✅ [CRITICAL] Le mode imm_general n'est jamais testé contre une référence — le seul test de correction est une comparaison de MSE

`tests/test_gss_filter.py:495-507` — statut : confirmed (2 vote(s)) — catégorie : trou de couverture

Tous les tests du filtre en mode imm_general (mode par défaut, donc TestFilterResult, TestRecursion, TestRun, etc.) ne vérifient que des invariants structurels (formes, somme des pi=1, PSD). L'unique test 'de correction' est test_imm_general_matches_pre_v0_10_behavior qui affirme seulement que le MSE d'imm_general est inférieur à celui de h5_exact sur un modèle non-(H5) — c'est-à-dire qu'il bat un filtre connu pour être biaisé sur ce modèle. Le code mathématique de _update_step_general (prg/filter/gss_filter.py:760-878), en particulier la formule Γ = S_YY_np1 − M_t·Cov^T à la ligne 809 (le bug non-PSD confirmé en vague 1, masqué par _psd_floor), n'est validé par aucune valeur de référence. C'est exactement le trou qui a laissé passer le bug high de la vague 1 : un biais mathématique substantiel dans imm_general passe toute la suite.

**Preuve :** tests/test_gss_filter.py:507 « assert df_gen["sq_err"].mean() < df_h5["sq_err"].mean() » — seule assertion de correction du mode imm_general. Code non couvert : prg/filter/gss_filter.py:809 « Gamma = _psd_floor(_sym(S_YY_np1 - M_t @ Cov_Ynp1_Yn.T)) » (jamais vérifié PSD avant floor, jamais comparé à une référence).

**Suggestion :** Deux tests de référence : (1) construire un modèle satisfaisant (H5) via apply_AB_constraint(params) et asserter l'égalité pas-à-pas (E_x, pi, log_lik, atol~1e-8) entre GSSFilter(mode='imm_general') et GSSFilter(mode='h5_exact') — sous (H5) les deux récursions doivent coïncider ; (2) pour N=3-4 pas et K=2, calculer p(X_n|y_{1:n}) par énumération brute des séquences de régimes (comme le fait déjà test_log_lik_brute_force pour l'EM) et comparer.

**Ajustement de sévérité proposé par les vérificateurs :** Downgrade critical -> high. This is a test-coverage gap, not itself a math defect; the underlying imm_general bias it failed to catch was rated high in wave 1, and a coverage gap should not outrank the defect it enables. It still merits high (not medium) because imm_general is the default mode, the gap demonstrably let a confirmed mathematical bias pass the entire suite, and the only 'correctness' assertion is a comparison against a filter known to be biased on the test model. | Lower from critical to high. This is a test-coverage/meta finding, not a direct defect; the concrete math bug it masked (non-PSD Γ in imm_general) was itself rated high in wave 1, and a gap should not outrank the defect it let through. High remains justified because imm_general is the default mode used by every production entry point (CLI, GUI, experiments) and the gap demonstrably allowed a substantive math bug to pass the entire 219-test suite.

### ✅ [HIGH] h5_exact n'est jamais testé sur un modèle qui satisfait (H5) — le filtre 'exact' tourne uniquement hors de son hypothèse

`tests/test_gss_filter.py:362, 419` — statut : confirmed (2 vote(s)) — catégorie : trou de couverture

Toutes les classes de test h5_exact (TestJosephForm, TestStationaryMoments, TestFilterModes) utilisent ModelGssK2Q1S1, qui VIOLE (H5) (le test test_h5_warns_on_non_h5_model le confirme), et suppriment le RuntimeWarning « The filter will be biased » avec @pytest.mark.filterwarnings. La contribution centrale du papier — l'exactitude du filtre sous (H5) — n'est donc validée nulle part : ni l'absence de biais, ni l'optimalité, ni la cohérence des moments stationnaires sur un modèle AB-contraint. apply_AB_constraint n'apparaît que dans tests/test_h5_constraint.py (propriétés algébriques de la projection), jamais en amont d'un filtrage.

**Preuve :** tests/test_gss_filter.py:362 et 419 « @pytest.mark.filterwarnings("ignore:mode='h5_exact'.*:RuntimeWarning") » ; grep apply_AB_constraint dans tests/ → uniquement test_h5_constraint.py. Aucun test ne filtre un modèle (H5)-compatible.

**Suggestion :** Fixture params_h5 = apply_AB_constraint(GSSParams.from_model(ModelGssK2Q1S1())) ; vérifier qu'aucun RuntimeWarning n'est émis à la construction en mode h5_exact, que le MSE bat imm_general (ou l'égale) sur des données simulées de ce modèle, et que les moments stationnaires P(k) satisfont leur équation de point fixe.

**Ajustement de sévérité proposé par les vérificateurs :** Keep high. This is a test-coverage gap rather than a runtime defect, but for research code whose headline contribution is the exactness of the h5_exact filter under (H5), the suite validating the filter only outside its hypothesis (with the bias warning explicitly silenced) leaves the paper's central claim unverified — especially salient given the confirmed imm_general math bug from wave 1, which removes the other mode as a trustworthy reference. | Keep high. This is a test-coverage gap rather than a runtime defect (arguably the only argument for medium), but for a research codebase whose central contribution is the exactness of the filter under (H5), having the headline claim validated nowhere — while the warning that flags the violation is explicitly silenced in every h5_exact test — gives false confidence on the paper's core result. Wave 1's confirmed imm_general math bug makes the absence of any cross-check on h5_exact's own hypothesis more consequential, not less.

### ✅ [HIGH] Filtre, simulateur et apprentissage testés exclusivement en K=2, q=1, s=1 — tout est scalaire

`tests/test_gss_filter.py:1-507 (et test_gss_simulator.py, test_supervised.py, test_semi_supervised.py)` — statut : confirmed (2 vote(s)) — catégorie : trou de couverture

Le dépôt fournit model_gss_K3_q1_s1.py, model_gss_K2_q2_s2.py, model_gss_K2_q1_s2.py, model_gss_K2_q2_s1.py (prg/models/), mais aucun test ne les importe (grep négatif). Toute la chaîne dynamique (simulation, filtrage des deux modes, EM, fit supervisé) ne tourne qu'en scalaire, où la plupart des erreurs de transposition/ordre de produit matriciel se compensent. Conséquences précises : la branche eigen de _psd_floor (gss_filter.py:1116-1119) n'est jamais exercée sur les matrices du filtre (Γ, P_post, Var_x sont 1×1) ; la forme Joseph (TestJosephForm) est testée là où elle est arithmétiquement triviale (q=s=1) alors que sa raison d'être est q>1 ; les gains de Kalman (q,s) non carrés, chol_W avec Δ corrélé multivarié, et la combinatoire K>2 de _update_step_general ne sont jamais exécutés. Les tests multidimensionnels existants (test_fmatrix, test_noise_covariance, test_read_csv multivariate) s'arrêtent à la construction/accesseurs.

**Preuve :** grep -rn "model_gss_K3|model_gss_K2_q2|model_gss_K2_q1_s2" tests/ → aucun résultat. prg/filter/gss_filter.py:1114-1115 « if M.shape == (1, 1): return np.maximum(M, eps) » — seul chemin atteint par les tests du filtre.

**Suggestion :** Paramétrer (pytest.mark.parametrize) les tests structurels du filtre et un run() court sur ModelGssK3Q1S1 et ModelGssK2Q2S2 ; ajouter le test d'équivalence h5_exact/imm_general (finding 1) sur le modèle q=2, s=2 AB-contraint, où la forme Joseph et _psd_floor travaillent réellement.

**Ajustement de sévérité proposé par les vérificateurs :** Keep high. As a pure test-coverage gap it could read as medium, but the context justifies high: the paper/code claims general (K, q, s) support via shipped models that are never executed end-to-end, and a real multivariate-sensitive bug (imm_general, wave 1) already slipped through the scalar-only suite. Minor correction to the writeup: drop or qualify the "_psd_floor eigen branch never exercised" evidence (it runs on the 2×2 stationary Σ(k) in every filter test; only the flooring at line 1119 and the recursion matrices are scalar-shielded). | Conserver high. (Argument pour medium possible — c'est un manque de tests, pas un défaut — mais le bug imm_general de la vague 1 démontre que le gap a déjà laissé passer un vrai bug mathématique, et les modèles multivariés sont exposés aux utilisateurs via les presets GUI.)

### ✅ [HIGH] _fast_logpdf / _precompute_gaussian_logpdf jamais validés contre scipy ; fallback eigen mort-né

`prg/filter/gss_filter.py:1050-1105` — statut : confirmed (2 vote(s)) — catégorie : trou de couverture

Le chemin rapide qui remplace multivariate_normal.logpdf dans la boucle chaude de _update_step_h5 (gss_filter.py:632-636) n'est comparé à scipy nulle part. Les tests Joseph (test_joseph_filter_outputs_match) comparent deux exécutions h5_exact qui utilisent toutes deux le chemin rapide — auto-cohérence, pas référence. Une erreur de log_det ou de normalisation (facteur 2, signe) serait invisible : elle déformerait les pi de manière systématique sans casser somme=1 ni PSD. De plus le fallback eigen (lignes 1081-1090, censé mimer allow_singular=True de scipy) est inatteignable dans les tests actuels : tous les Γ(j,k) sont planchéifiés à 1e-9 donc Cholesky réussit toujours ; ce fallback n'a jamais été exécuté.

**Preuve :** gss_filter.py:1093-1105 (_fast_logpdf) et 1081-1090 (fallback eigen). Seuls usages testés : auto-comparaisons h5_exact dans tests/test_gss_filter.py:387-401, en s=1 où la factorisation est un scalaire.

**Suggestion :** Test unitaire direct : pour des covariances aléatoires PD (s=1..4) et des covariances singulières (rang déficient), comparer _fast_logpdf(dev, *_precompute_gaussian_logpdf(cov)) à scipy.stats.multivariate_normal.logpdf(x, mean, cov, allow_singular=True) à 1e-12.

**Ajustement de sévérité proposé par les vérificateurs :** high -> medium. The facts are accurate and the proposed unit test is cheap and worthwhile, but empirical verification shows the fast path currently produces scipy-identical values (1e-14 on PD covariances, the production regime guaranteed by the 1e-9 PSD floor) and the dead eigen fallback is itself correct when exercised. No current output of the filter is wrong, so this is a hardening/regression-protection item, not an active correctness bug. | Rétrograder high → medium. Vérification empirique : le chemin rapide est actuellement correct (écart max 1.8e-13 vs scipy sur covariances PD s=1..4 ; fallback eigen à 3.6e-15 quand forcé). Aucun impact présent sur les résultats — l'impact est conditionnel à l'introduction future d'une erreur. Reste medium (pas low) car c'est le noyau numérique central du filtre "exact" revendiqué par le papier, en s=1 seulement dans les tests, et le correctif (test unitaire direct vs scipy, ~15 lignes) est trivial et à forte valeur.

### ✅ [HIGH] La PSD du Σ_W joint après EM/projection AB n'est jamais testée ; aucun round-trip apprentissage → GSSParams → filtre

`tests/test_semi_supervised.py:330-335` — statut : confirmed (2 vote(s)) — catégorie : trou de couverture

Les tests EM et supervisés vérifient cholesky(Sigma_U) et cholesky(Sigma_V) séparément (test_semi_supervised.py:334-335, test_supervised.py:274-284), mais jamais la PD du bloc joint [[Σ_U, Δ],[Δᵀ, Σ_V]]. Or _apply_constraints (prg/learning/semi_supervised.py:246-247) appelle _nearest_spd(SU) et _nearest_spd(SV) séparément sans projeter le joint : un Δ appris qui viole la PD jointe passe tous les tests, puis GSSNoiseCovariance lèverait CovarianceError au moment de filtrer. Aucun test ne fait le round-trip complet : fit_supervised/fit_semi_supervised → GSSParams.from_model (ou constructeur) → GSSFilter.run. Les tests CLI (test_generated_importable) s'arrêtent à get_params() sans valider le dict ni construire le modèle. C'est la zone des « planchers de covariance fragiles » identifiée en vague 1 : aucun test ne la met sous pression.

**Preuve :** prg/learning/semi_supervised.py:246-247 « SU = _nearest_spd(SU); SV = _nearest_spd(SV) » — pas de projection du bloc joint. tests/test_semi_supervised.py:334-335 « np.linalg.cholesky(params["Sigma_U_list"][k]); np.linalg.cholesky(params["Sigma_V_list"][k]) » — Δ jamais inclus.

**Suggestion :** Après chaque fit (supervisé, EM post-hoc, EM each-iter) : np.linalg.cholesky(np.block([[SU, Dt],[Dt.T, SV]])) pour chaque k, puis construire GSSParams (via le dict) et exécuter GSSFilter(params).run(N=50) — le constructeur valide P, pi0, Σ_W, Σ_z0 d'un coup.

**Ajustement de sévérité proposé par les vérificateurs :** Keep high, with a caveat. The missing learn→GSSParams→GSSFilter round-trip is the dominant gap and justifies high for this pipeline (learning output feeding the exact filter is the project's main workflow, and wave 1 already flagged fragile covariance floors). However the joint-PD sub-claim alone would be medium: the M-step estimator is PSD-by-construction and the per-block _nearest_spd cannot create an indefinite joint, so failures are confined to singular/near-singular cases (small effective sample per regime, near-deterministic X/Y coupling, EM weight collapse) rather than arbitrary learned Δ. | Keep high. As a test-gap finding the rating is justified: the gap hides a reachable crash (47% trigger rate under plausible data-poor-regime conditions) on real production paths (e3 script, GUI) in the exact zone wave 1 flagged as fragile, plus a silent near-singular variant where the joint narrowly passes cholesky and a degenerate Σ_W flows into the filter undetected. The only mitigating factor — the common failure is a loud fail-fast CovarianceError rather than silent corruption — is offset by that silent borderline case.

### ✅ [MEDIUM] TestOptionB.test_zero_mean_model_identical_to_nonzero_zero_init est tautologique : il compare deux exécutions identiques des mêmes paramètres

`tests/test_gss_filter.py:327-342` — statut : confirmed (2 vote(s)) — catégorie : test faible (tautologique)

Le test prétend vérifier l'équivalence « Option B / zero-mean », mais son corps construit deux GSSFilter sur le MÊME objet params et compare leurs sorties — le commentaire l'admet (« params already has mu_z0=0 — just run twice and compare »). Cela ne teste que le déterminisme, déjà couvert par TestReset. De plus la prémisse « zero-mean » est fausse : ModelGssK2Q1S1 a b_list=[[1,2],[-2,-1]] ≠ 0, donc E[Z] stationnaire ≈ [0.11, 2.42] ≠ 0 (vérifié numériquement). Le chemin réellement intéressant (mu_z0 ≠ 0, gestion des moyennes non nulles dans la récursion) n'est testé nulle part.

**Preuve :** tests/test_gss_filter.py:334-336 « filt_a = GSSFilter(params); filt_b = GSSFilter(params) » puis comparaison ra vs rb — aucune variation de paramètre entre a et b.

**Suggestion :** Construire deux GSSParams réellement différents (l'un avec mu_z0=0, l'autre avec mu_z0 ≠ 0 mais le même régime stationnaire) si c'est la propriété revendiquée, ou supprimer le test ; ajouter un test du filtre sur un modèle à mu_z0 ≠ 0.

**Ajustement de sévérité proposé par les vérificateurs :** medium → low. The finding is accurate, but it is purely a test-quality issue with no functional impact: the test is redundant with TestReset rather than hiding a plausible failure mode, and the property it claims to check (zeroing mu_z0 leaves output unchanged) holds trivially by the filter's design since mu_z0 only seeds a fixed-point iteration that converges to a unique stationary limit regardless of seed. | Lower from medium to low. The defect is a tautological test giving false coverage confidence (the file docstring even advertises 'Option B / zero-mean equivalence' coverage), but the property it fails to test verifiably holds in the current implementation (filter output is insensitive to mu_z0 to ~1e-14), so no bug is being masked today. Impact is limited to a missing regression guard for paper-related claims.

### ✅ [MEDIUM] test_empirical_mean_near_zero : prémisse fausse (le modèle n'est pas centré) et passage à 0.0015 de la tolérance

`tests/test_gss_simulator.py:225-236` — statut : confirmed (2 vote(s)) — catégorie : fragilité + test faible

La docstring affirme « With zero initial mean and a stable model, the long-run empirical mean of Z_n should be close to 0 », mais ModelGssK2Q1S1 a b ≠ 0 : la moyenne stationnaire théorique de X est ≈ 0.112 (et celle de Y ≈ 2.42). Mesuré avec la graine du test (seed=123, N=5000) : moyenne empirique de x = 0.19846, contre atol=0.2 — le test passe avec une marge de 0.0015. C'est le seul test statistique du simulateur : il repose sur une hypothèse fausse, est au bord de l'échec sur un simulateur correct, et ne détecterait pas un vrai biais (un simulateur ignorant b ou Δ donnerait une moyenne ~0 et passerait encore mieux). Aucun test ne compare les fréquences de transition empiriques à P ni la covariance empirique à la covariance stationnaire.

**Preuve :** Exécution reproduite : xs.mean() = 0.19846454 ; assertion np.testing.assert_allclose(mean_x, 0.0, atol=0.2). Moyenne stationnaire théorique E[Z]=[0.112, 2.418] (calcul via _precompute_stationary).

**Suggestion :** Comparer la moyenne empirique à la moyenne stationnaire THÉORIQUE (pas à 0) avec une tolérance dérivée de l'écart-type ; ajouter un test des fréquences de transition r_n→r_{n+1} contre P (tolérance binomiale) et de la variance empirique contre Σ stationnaire.

**Ajustement de sévérité proposé par les vérificateurs :** Keep medium. It is a test-quality defect (false premise, near-zero power, passes by seed luck — 72% of seeds would fail on a correct simulator), not a production-code bug, and the fixed seed makes it deterministic in CI, so it does not warrant high; but it is the sole statistical guard on the simulator that underpins all filter/EM validation, so low would understate it. | Keep medium. It is a test-only issue (no runtime bug), but the sole statistical guard on the simulator that underpins the paper's empirical validation is both blind to real bugs (a b-dropping simulator passes better than the correct one) and statistically miscalibrated (the correct simulator fails it on ~70% of seeds), which is if anything at the upper end of medium.

### ✅ [MEDIUM] FilterResult.innovation et log_lik ne sont assertés par aucun test

`prg/filter/gss_filter.py:119-146` — statut : confirmed (2 vote(s)) — catégorie : trou de couverture

Les champs innovation (ν_n, base des diagnostics GUI de qualité du filtre) et log_lik (log-vraisemblance incrémentale) de FilterResult ne figurent dans aucune assertion de tests/test_gss_filter.py (grep négatif sur 'innovation' et 'log_lik'). La log-vraisemblance du HMM est validée par force brute côté EM (test_log_lik_brute_force), mais pas celle du filtre — alors qu'elle est calculée par un code différent (logsumexp sur log_alpha, deux modes distincts). Une innovation mal centrée ou un log_lik faux passerait toute la suite et corromprait silencieusement les diagnostics d'innovation de la GUI (Ljung-Box, JB).

**Preuve :** grep -n "innovation\|log_lik" tests/test_gss_filter.py → aucune occurrence. Champs définis dans gss_filter.py:142-146, calculés lignes 590-596, 642-643, 679, 751-757, 827, 869.

**Suggestion :** Pour N=3-4, K=2 : comparer la somme des log_lik du filtre à la log-vraisemblance jointe brute-force des observations (même technique que test_log_lik_brute_force) ; vérifier que la moyenne des innovations standardisées tend vers 0 et leur variance vers 1 sur un long run d'un modèle (H5)-compatible.

**Ajustement de sévérité proposé par les vérificateurs :** Keep medium. It is a coverage gap rather than a demonstrated defect (which would cap it below high), but the untested fields feed the GUI filter-quality diagnostics and the incremental log-likelihood used for model assessment, and the computation paths are nontrivial and duplicated across two modes — medium is well calibrated. | Keep medium. It is a coverage gap rather than a demonstrated bug (so not high), but given that wave 1 already showed tests missed a confirmed math bug in imm_general, leaving the filter's likelihood and innovation outputs entirely unasserted in a correctness-critical research codebase justifies medium over low.

### ✅ [MEDIUM] Les trois gardes de _compute_stationary (chaîne réductible, signes mixtes, dégénérescence) n'ont aucun test

`prg/classes/GSSParams.py:286-326` — statut : confirmed (2 vote(s)) — catégorie : trou de couverture

Les gardes ajoutées lors du suivi d'audit (commit a1b8487 « stationary guards ») — RuntimeWarning si plusieurs valeurs propres ≈ 1 (chaîne réductible), si le vecteur propre a des signes mixtes, et repli uniforme si la normalisation dégénère — ne sont exercées par aucun test : il n'y a aucun pytest.warns dans toute la suite (grep négatif), et test_gss_params.py:134-149 ne teste _compute_stationary que sur des P irréductibles bien conditionnées. Le repli uniforme (return np.full(K, 1.0/K)) n'est jamais exécuté. Idem pour _stationary_dist côté GUI (testé sur un seul P sain, test_main_window_gui.py:53-58).

**Preuve :** grep -rn "pytest.warns" tests/ → aucun résultat. GSSParams.py:287-295 (garde réductibilité), 303-313 (signes mixtes), 318-326 (repli uniforme) : 0 test.

**Suggestion :** Trois tests ciblés : P=np.eye(2) (réductible → pytest.warns(RuntimeWarning, match='reducible') et résultat toujours une distribution valide) ; P bistochastique par blocs ; vérifier le repli uniforme via un P pathologique construit ad hoc.

**Ajustement de sévérité proposé par les vérificateurs :** Borderline medium/low. This is a test-coverage gap, not a code defect — the guards themselves appear correct. Medium is defensible only because the guards were the sole deliverable of a recent audit-fix commit and protect pi0 which propagates into filtering; if calibrating strictly, low-medium is more accurate than medium. | lower to low — real, well-evidenced coverage gap with a reachable trigger path, but no current defect: the guards were empirically verified to behave as documented, warnings are advisory, and the function still returns a valid distribution in the reducible case. The proposed three targeted tests are cheap and correct as specified.

### ✅ [MEDIUM] test_missing_destination_regime_raises ne 'raise' rien : il asserte que le fit RÉUSSIT avec K=1

`tests/test_supervised.py:289-298` — statut : confirmed (2 vote(s)) — catégorie : test faible (nom mensonger)

Le nom et la docstring (« A CSV where regime 1 never appears as destination → ValueError ») annoncent une levée d'exception, mais le corps asserte le succès : result["K"] == 1. Le comportement censé être gardé (erreur sur régime de destination manquant) n'est donc pas testé. Pire, le test entérine un fit K=1 alors que tout l'aval (FMatrix, GSSParams) exige K ≥ 2 — le dict produit est inutilisable, et personne ne le vérifie. Le cas voulu (K=2 déclaré, régime 1 jamais destination) reste non couvert.

**Preuve :** tests/test_supervised.py:290 docstring « → ValueError » vs lignes 296-298 « result = fit_supervised(...); assert result["K"] == 1 » — aucune pytest.raises.

**Suggestion :** Soit passer K=2 explicitement avec un r toujours nul et asserter le ValueError attendu, soit renommer le test et décider (et tester) le contrat de fit_supervised pour K=1.

**Ajustement de sévérité proposé par les vérificateurs :** Keep medium. Test-only defect (no production bug), but it combines an untested production error path with a misleadingly-named test that entrenches an output contract (K=1 fit dict) the entire downstream codebase rejects. Note: the suggested fix should use a regime appearing only at t=0 (source-but-never-destination) rather than r always 0 with K=2, which would hit the source check instead.

### ✅ [MEDIUM] Diagnostics GUI : assertions tautologiques (0 ≤ p ≤ 1) et mode h5 de _standardise_innovations non testé

`tests/test_main_window_gui.py:38-65` — statut : confirmed (2 vote(s)) — catégorie : test faible (GUI offscreen)

test_ljung_box_white_noise_high_pvalue promet « high p-value » mais n'asserte que 0.0 <= p <= 1.0 — vrai pour n'importe quelle implémentation renvoyant une probabilité ; une statistique Ljung-Box fausse passerait. test_standardise_innovations_sample_mode ne vérifie que la forme de sortie, pas le blanchiment (variance ≈ 1) ; et le mode h5_exact de _standardise_innovations (prg/gui/diagnostics.py:110-125 : covariance marginale stationnaire S = Σ w[Γ + δδᵀ]) n'est pas couvert du tout (arguments mix_w/Gamma/mu_Y_jk toujours None dans les tests). test_shape_diagnostics_gaussian ne contraint pas non plus la p-value JB sur un échantillon gaussien (devrait être > 0.01).

**Preuve :** tests/test_main_window_gui.py:42-43 « _q, p, _h = _ljung_box(x); assert 0.0 <= p <= 1.0 » ; lignes 63-65 « out = _standardise_innovations(innov, None, None, None); assert out.shape == innov.shape ».

**Suggestion :** Bruit blanc seedé : asserter p_LB > 0.05 ; série AR(1) corrélée : asserter p_LB < 0.01 (le test devient bilatéral). Pour _standardise_innovations : vérifier std(out) ≈ 1 en mode échantillon, et tester le mode h5 avec des Γ/μ jouets dont la solution est analytique.

**Ajustement de sévérité proposé par les vérificateurs :** Keep medium (lower edge). Pure test-strength issue, no production defect, and the file self-describes as smoke tests — but the uncovered h5 branch is exactly the code path the GUI uses in the project's flagship h5_exact mode, and the wave-1 imm_general math bug shows untested mixture-covariance math in this codebase does regress, so medium rather than low is justified. | Keep medium. The gap is on nontrivial mixture-covariance math in the production-default (h5_exact) display path with a silent exception fallback, which justifies medium; but it is display/diagnostics code only — filter and experiment outputs are unaffected — so it should not be raised above medium.

### ✅ [MEDIUM] Workers jamais exécutés et main_window (2214 lignes) couvert par un seul smoke-test constructeur

`tests/test_main_window_gui.py:90-94, 128-132` — statut : confirmed (2 vote(s)) — catégorie : trou de couverture (GUI)

TestWorkers.test_construct asserte « _SimWorker(...) is not None » — un constructeur Python ne peut pas renvoyer None : assertion tautologique. La logique run() des workers (157 lignes : simulation/filtrage en thread, signaux de progression, gestion d'erreur) n'est jamais exécutée. TestMainWindow.test_construct ne vérifie que windowTitle != "" sur les 2214 lignes de main_window.py. Ces tests gardent bien le découpage en modules (imports), conformément à leur docstring, mais la session GUI réelle (simulate → filter → diagnostics via _SessionState) n'a aucun chemin testé. À l'inverse, test_param_panel_gui.py est substantiel (valeurs compute_AB vérifiées) — le contraste montre que c'est faisable offscreen.

**Preuve :** tests/test_main_window_gui.py:93-94 « assert _SimWorker(params, N=10, seed=0) is not None » ; ligne 132 « assert win.windowTitle() != "" ».

**Suggestion :** Avec qtbot.waitSignal, démarrer _SimWorker/_FilterWorker sur N=20 et asserter le contenu des signaux de résultat (formes, longueurs) ; pour la fenêtre, dérouler un scénario minimal simulate→filter via les slots publics et vérifier _SessionState.

**Ajustement de sévérité proposé par les vérificateurs :** medium (unchanged) — a test-coverage gap rather than a code defect, but correctly rated medium: the untested _FilterWorker.run contains real mathematical computation (μ₂/Γ₂ moments), defaults to mode="imm_general" which wave 1 confirmed mathematically buggy, and the entire 2214-line main window plus the simulate→filter→diagnostics session path has no executed test. | none — medium est correct

### ✅ [MEDIUM] La CLI du filtre (prg/filter/main.py) n'a aucun test, contrairement aux CLIs d'apprentissage

`prg/filter/main.py:124-190` — statut : confirmed (2 vote(s)) — catégorie : trou de couverture

Aucun fichier de tests n'importe prg.filter.main (grep négatif), alors que les CLIs supervised et semi_supervised ont chacune une classe TestCLI complète (smoke, flags, erreurs). Le parseur argparse (lignes 124+, ~10 add_argument) et le main() du filtre sont à 0 % de couverture. C'est précisément là que la vague 1 a trouvé que --mode n'est pas exposé : un test CLI aurait documenté/figé ce contrat (et le futur ajout de --mode arrivera sans filet).

**Preuve :** grep -rn "filter.main\|from prg.filter import" tests/ → aucun résultat ; prg/filter/main.py:124-190 (parser) non couvert.

**Suggestion :** Répliquer le pattern TestCLI des fichiers d'apprentissage : smoke run sur un CSV simulé, --output, CSV manquant → SystemExit ; ajouter le test de --mode en même temps que son implémentation.

**Ajustement de sévérité proposé par les vérificateurs :** Keep medium. A pure test-coverage gap would normally be low, but this CLI defaults to GSSFilter's mode="imm_general" (the mode confirmed buggy in wave 1) with no flag to select h5_exact and no test documenting that contract, which elevates the practical risk to medium. | keep medium (lower bound of medium; would be low if the CLI were not the README-documented primary entry point and the locus of the wave-1 --mode contract gap)

### ✅ [MEDIUM] Planchers de covariance EM sans pression de test : famine de régime, Σ_W quasi-singulier, ridge 1e-8 testé pour la seule finitude

`prg/learning/semi_supervised.py:303-316` — statut : confirmed (2 vote(s)) — catégorie : trou de couverture

La branche non dégénérée de _weighted_fit calcule SigW = Σ w r rᵀ / Σw sans plancher SPD (lignes 303-316) — quand les poids γ se concentrent (K surestimé, régime affamé), SigW devient quasi-singulier et seul le ridge 1e-8 de _log_mvn_batch (lignes 113-118) sauve l'itération suivante. Or : (1) test_handles_near_singular n'asserte que np.isfinite(result) sur le ridge ; (2) test_zero_weight_falls_back ne teste que w=0 exactement (Wsum < 1e-12), pas le régime quasi-affamé ; (3) aucun test ne lance l'EM avec K=3 sur des données à 2 régimes, le scénario qui exerce réellement ces planchers fragiles (vague 1). Les branches de _em_run pour denom[k] <= _LOG_FLOOR (ligne 552 : conservation de l'ancienne ligne de P) sont également non couvertes.

**Preuve :** semi_supervised.py:312-314 « SigW = (residuals.T @ (w[:, None] * residuals)) / Wsum » sans _nearest_spd ; tests/test_semi_supervised.py:95-99 « assert np.isfinite(result[0]) » seule assertion sur le chemin singulier.

**Suggestion :** Test EM avec K=3 sur les données K=2 (seed fixe) : asserter que ça termine, que tous les Σ_W joints sont PD et que P reste stochastique ; test unitaire de _weighted_fit avec w concentré sur 3 points colinéaires (SigW singulier) vérifiant la sortie exploitable.

**Ajustement de sévérité proposé par les vérificateurs :** medium (inchangée) — le trou de test est réel et le mode de défaillance est silencieux (run dégénéré pouvant gagner la sélection multi-start), mais c'est du code de recherche avec garde-fous partiels (_nearest_spd sur blocs diagonaux à chaque M-step, runs en échec ignorés proprement), donc pas de quoi monter en high. | low — the cited test gaps are real and the proposed tests are worthwhile regression guards, but the impact is overstated: an unconditional _nearest_spd eigenvalue floor (1e-8) already protects Sigma_U/Sigma_V in every M-step, the ridge provably covers the remaining PSD-singular joint case, per-run exception containment bounds the blast radius, and the K-overestimation scenario empirically terminates with PD covariances and a stochastic P.

### ✅ [LOW] Branches défensives du filtre non couvertes : M singulier dans _check_h5, non-convergence des moments stationnaires, fallback lstsq, point fixe P(k)

`prg/filter/gss_filter.py:259-269, 362-368, 1037-1042` — statut : confirmed (1 vote(s)) — catégorie : trou de couverture

Quatre chemins défensifs/mathématiques sans test : (1) _check_h5 branche LinAlgError quand M(k) est singulier (lignes 259-269, warning spécifique) ; (2) le warning de non-convergence du point fixe stationnaire après 1000 itérations (lignes 362-368, modèle instable ρ(F)≥1) ; (3) le fallback lstsq de _safe_solve (1037-1042) ; (4) TestStationaryMoments ne vérifie que le point fixe de µ(k) (test_mu_fixed_point, tests/test_gss_filter.py:436-444) — l'équation de point fixe du second moment P(k) (lignes 351-355), plus riche en termes croisés (Fw_mu bᵀ, etc.), n'est pas vérifiée.

**Preuve :** tests/test_gss_filter.py:436-444 teste « µ(k) = F_k Σ_j p_rev[j,k] µ(j) + b_k » uniquement ; aucun test ne référence _safe_solve, ni le message « did not fully converge », ni « M(k) is singular ».

**Suggestion :** Ajouter test_P_fixed_point symétrique à test_mu_fixed_point ; un modèle avec A=1.5 (instable) sous caplog pour le warning de non-convergence ; un Σ_V=0-régime pour la branche M singulier.

### ✅ [LOW] Tolérances statistiques très lâches : MSE < 0.8 et variance < 0.8 ne détectent qu'un filtre catastrophique

`tests/test_gss_filter.py:287-303` — statut : confirmed (1 vote(s)) — catégorie : test faible (tolérance)

test_rmse_below_naive accepte MSE jusqu'à 0.8 contre une base naïve de ~1.0, et test_posterior_variance_decreases_on_average accepte une variance moyenne jusqu'à 0.8. Un filtre dégradé de 50-70 % (p.ex. mauvais gain de Kalman, mauvaise pondération des régimes) passe. Combiné au finding sur l'absence de référence pour imm_general, la suite ne peut détecter qu'une divergence franche, pas un biais modéré. Ces seuils sont assumés comme « sanity », mais aucun test plus serré ne les complète.

**Preuve :** tests/test_gss_filter.py:294 « assert filter_mse < 0.8 » ; ligne 303 « assert mean_var < 0.8 ».

**Suggestion :** Sur un modèle (H5)-compatible avec graine fixe, figer le MSE attendu par un test de non-régression à ±10 % (valeur dorée recalculée une fois la référence du finding 1 en place).

**Ajustement de sévérité proposé par les vérificateurs :** Maintenir low : c'est un déficit de pouvoir de détection des tests, pas un bug du code de production. Noter toutefois que le docstring de test_rmse_below_naive est factuellement faux (base naïve réelle ~63, pas ~1) et mériterait correction en même temps que le resserrage des seuils.

### ✅ [LOW] Fixture rng partagée scope='module' : les entrées aléatoires de chaque test dépendent de l'ordre d'exécution

`tests/test_h5_constraint.py:55-58` — statut : confirmed (1 vote(s)) — catégorie : fragilité (ordre des tests)

La fixture rng (np.random.default_rng(42), scope='module') est un générateur AVEC ÉTAT consommé séquentiellement par les tests : les matrices vues par test_nonzero_for_random_AB dépendent du nombre de tirages effectués par les tests précédents. Lancer un test isolé (pytest -k) ou réordonner/ajouter un test change toutes les entrées — un échec observé en suite complète n'est pas reproductible en isolation. test_nonzero_for_random_AB (lignes 196-203, assert norm > 1e-3 sur A,B « aléatoires ») est le plus exposé : la marge dépend du tirage, donc de l'ordre.

**Preuve :** tests/test_h5_constraint.py:55-58 « @pytest.fixture(scope="module") def rng(): return np.random.default_rng(42) » consommé par 6 tests ; lignes 196-203 assertion sur une norme de tirage dépendant de l'état.

**Suggestion :** Passer la fixture en scope='function' (chaque test repart de la graine 42), ou créer un rng local np.random.default_rng(graine_propre_au_test) dans chaque test.

### ✅ [LOW] Syntaxe PEP 758 « except UnicodeDecodeError, OSError: » : SyntaxError à la collection sur tout Python < 3.14

`tests/test_no_stale_refs.py:134` — statut : confirmed (1 vote(s)) — catégorie : fragilité (portabilité)

La ligne 134 utilise la nouvelle syntaxe sans parenthèses de Python 3.14 (PEP 758). Sur tout interpréteur ≤ 3.13 — y compris l'IDE d'un collaborateur ou un CI mal épinglé — ce n'est pas le test qui échoue mais le FICHIER entier qui casse à la collection pytest (SyntaxError), masquant le vrai problème de version. Le gain sur « except (UnicodeDecodeError, OSError): » est nul ; le coût de portabilité/diagnostic est réel pour du code de test.

**Preuve :** tests/test_no_stale_refs.py:134 « except UnicodeDecodeError, OSError: » — compile en 3.14.5 (vérifié), SyntaxError en ≤ 3.13.

**Suggestion :** Remettre les parenthèses : « except (UnicodeDecodeError, OSError): ».

**Ajustement de sévérité proposé par les vérificateurs :** Keep at low. The project hard-pins Python >=3.14 (pyproject.toml requires-python and all CI workflows), so nothing supported is broken; this is a portability/diagnostic-clarity nit at most. If acted on, the fix should also cover the 3 identical occurrences in prg/gui/main_window.py:717,721,726, otherwise the stated goal (clean failure on older interpreters) is not achieved.

### ✅ [LOW] Hypothèse d'ordre sur les warnings et fichier temporaire fuité

`tests/test_gss_filter.py:468-477, 275` — statut : confirmed (1 vote(s)) — catégorie : fragilité (hygiène)

Deux points mineurs : (1) test_h5_warns_on_non_h5_model asserte « "B(k)" in str(runtime_ws[0].message) » — il suppose que le warning (H5) est le PREMIER RuntimeWarning émis ; si une garde future (p.ex. stationnarité, finding GSSParams) émet un RuntimeWarning avant lui, l'assertion vise le mauvais warning et échoue de façon déroutante. (2) test_csv_without_x_has_no_sq_err utilise tempfile.NamedTemporaryFile(delete=False) sans suppression : un fichier .csv est fuité dans /tmp à chaque exécution (et le pattern réouverture-pendant-ouvert n'est pas portable Windows) alors que la fixture tmp_path fait mieux.

**Preuve :** tests/test_gss_filter.py:476-477 « assert "B(k)" in str(runtime_ws[0].message) » ; ligne 275 « tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) ».

**Suggestion :** (1) any("B(k)" in str(w.message) for w in runtime_ws) ; (2) utiliser tmp_path.

**Ajustement de sévérité proposé par les vérificateurs :** Keep as low — correctly calibrated. Both are test-hygiene issues with no current test failure: a brittle warning-index assertion and a leaked temp file per run. Note the Windows-portability side remark in point (2) is inaccurate for delete=False on modern Python and could be dropped from the finding text.

### ❌ [LOW] test_constraint_note_ab : « or 'ab' in code.lower() » rend l'assertion quasi indéfaillible

`tests/test_supervised.py:462-464` — statut : refuted (1 vote(s)) — catégorie : test faible (tautologique)

L'assertion « assert "AB" in code or "ab" in code.lower() » est satisfaite dès que le code généré contient une sous-chaîne 'ab' n'importe où (p.ex. 'probability', 'table', un float '0.ab…' non, mais tout mot contenant ab) — y compris si la note de contrainte AB a disparu. Le test ne vérifie donc pas que la provenance 'constraint=ab' est consignée dans le fichier généré, contrairement à test_delta_zero_note qui vérifie la chaîne exacte « Delta=0    : yes ».

**Preuve :** tests/test_supervised.py:464 « assert "AB" in code or "ab" in code.lower() ».

**Suggestion :** Asserter la chaîne de provenance exacte émise par _generate_model_code pour constraint='ab' (même pattern que « Delta=0    : yes »).

**Ajustement de sévérité proposé par les vérificateurs :** n/a (not a real bug; at most an optional test-hardening style suggestion)

### ❌ [LOW] test_post_hoc_vs_each_iter_differ : assertion « doit différer » susceptible d'échec spurieux

`tests/test_semi_supervised.py:425-464` — statut : refuted (1 vote(s)) — catégorie : fragilité

Le test exige que les modes post-hoc et each-iter produisent des A différents (max diff > 1e-6). C'est une propriété générique, pas un contrat : si une amélioration légitime (meilleure initialisation, convergence plus poussée) fait coïncider les deux estimations sur ce jeu de données/graine, le test échoue alors que rien n'est cassé. Le test apporte peu (les deux modes sont déjà distingués par test_post_hoc_keeps_log_lik_monotone vs test_constraint_each_iter_b) pour un risque de faux positif non nul.

**Preuve :** tests/test_semi_supervised.py:460-464 « assert max(diffs) > 1e-6, "Post-hoc and each-iter modes produced identical estimates..." ».

**Suggestion :** Remplacer par une propriété directionnelle : log-vraisemblance finale du mode post-hoc (avant projection) ≥ celle du mode each-iter (le GEM contraint ne peut pas faire mieux que l'EM libre), ou supprimer.

### ✅ [INFO] test_no_stale_refs : lint lexical utile mais aux limites précises — il ne garde que des chaînes littérales dans les fichiers suivis par git

`tests/test_no_stale_refs.py:124-145` — statut : confirmed (1 vote(s)) — catégorie : analyse test_no_stale_refs

Ce que le test garde réellement : il greppe tous les fichiers texte suivis par git (code, docs, paper/*.tex) contre 11 regex — anciens noms d'API v0.11-v0.12 (compute_A_from_h5, apply_h5_constraint, *_lehmann…), anciennes valeurs du flag --constraint, anciens numéros d'équations « eq. (4.4/4.8/4.20) », et le nom « Lehmann ». C'est un garde-fou anti-régression copier-coller efficace, mais : (1) purement lexical — une référence sémantiquement périmée sous un autre libellé passe ; (2) la regex d'équations est en minuscules sans IGNORECASE : « Eq.~(4.4) » ou « Eq. (4.8) » (capitalisation LaTeX usuelle) passent au travers ; (3) la whitelist est un match par SOUS-CHAÎNE (« any(w in rel_path ...) », lignes 117-118) : tout chemin contenant « CHANGELOG.md » est exempté ; (4) le motif « \bLehmann\b » s'applique aussi au .tex suivi — une future entrée bibliographique citant un auteur Lehmann ferait échouer la suite ; (5) dépendance à git : subprocess.check_output(git ls-files) (ligne 86) → CalledProcessError (ERROR, pas SKIP) hors dépôt git ; (6) la docstring dit « fails on the first hit » mais le code accumule toutes les offenses (lignes 128-139) — divergence mineure doc/comportement.

**Preuve :** tests/test_no_stale_refs.py:117-118 « return any(w in rel_path for w in _WHITELIST) » ; ligne 60 « re.compile(r"\beq\.\s*\(4\.(?:4|8|20)\)") » sans re.IGNORECASE ; ligne 86 « subprocess.check_output(["git", ...]) ».

**Suggestion :** Ajouter re.IGNORECASE au motif d'équations ; ancrer la whitelist sur le chemin relatif exact ; pytest.skip si git ls-files échoue ; corriger la docstring.

**Ajustement de sévérité proposé par les vérificateurs :** Keep at [info] — correctly calibrated as an informational note on the lint test's limits. Drop sub-claim (4) (Lehmann-in-bibliography risk: paper/ is gitignored, so no tracked .bib/.tex bibliography exists) and correct '11 regex' to 10. The actionable items remain (2) IGNORECASE plus a [~\s]* tolerance for LaTeX ties, (3) anchored whitelist, (5) pytest.skip on git failure, (6) docstring fix.

## Scripts & expériences

_Audit complet des scripts d'expériences (prg/experiments : run_simulations, run_supervised, run_em, run_real_data, metrics, models_paper, make_figures, make_figures_real, fill_placeholders — lus intégralement) et survol de scripts/ (e1/e2/e3, baselines hamilton_msar et kalman_single, labels, params_utils, fetch/build de données, scripts de figures et de vérification). La discipline de graines est globalement bonne (seeds 0..n-1 partout, comparaisons appariées h5/imm sur les mêmes trajectoires, standardisation train-only sans fuite dans labels.py et load_enso) et la baseline kalman_single est mathématiquement correcte. Les problèmes les plus graves sont (1) une dérive d'API silencieuse : les contraintes legacy 'b'/'a' passées par e3_bw_em.py et e1_supervised_h5.py sont des no-op dans le code actuel, rendant les variantes V1/V2/V3 identiques à V0 et les résultats commités results/e3 irréproductibles ; (2) fill_placeholders a déjà consommé destructivement les \ph{} du papier, figeant des dizaines de chiffres narratifs alors que ni les CSV de résultats ni les tables générées ne sont versionnés, avec en plus une dérive avérée de protocole EM (defaults du script ≠ CSV ≠ log ≠ papier) ; (3) un look-ahead/circularité des labels de régime ENSO (ONI centré, fonction lissée de Y) qui biaise le test (H5) E1 vers la conclusion favorable au papier, et un look-ahead dans la baseline Hamilton (probabilités lissées). S'y ajoutent des incohérences de métriques (RMSE papier vs code pour q>1, légende tab_em_basin contredite par tab_em_restarts, CPU/step incluant le simulateur, table BIC factice) de sévérité moyenne à faible._

### ✅ [CRITICAL] Les variantes V1/V2/V3 de e3_bw_em.py passent des contraintes 'b'/'a' qui sont des no-op silencieux : les 4 variantes sont devenues identiques

`scripts/e3_bw_em.py:65-70` — statut : confirmed (2 vote(s)) — catégorie : Correction / API silencieusement cassée

VARIANTS utilise constraint='b' (V1, V3) et constraint='a' (V2), mais l'API actuelle de prg/learning ne reconnaît que 'ab' : prg/learning/supervised.py:258 et prg/learning/semi_supervised.py:249 font `if constraint == "ab": ...` sans aucune validation ni erreur pour les autres chaînes. Toute autre valeur est ignorée silencieusement. Conséquence : si on relance e3_bw_em.py aujourd'hui, V1_posthoc_B, V2_posthoc_A et V3_GEM_B sont strictement équivalentes à V0_unconstr (même graine, même EM, aucune projection) — Table 3 et Figure 3 (figure3_em_restarts.py, qui documente encore « V3 ~ -66k vs V0/V2 ~ -2k ») deviennent silencieusement fausses. Les résultats commités results/e3/table3.json (V0 trainLL=-2009.5 mse=0.931 ; V1 mse=10.44 ; V2 mse=11.21 ; V3 trainLL=-65971.9) datent d'une API antérieure (supervised_run.log mentionne encore proj 'a'/'su') et ne sont PLUS reproductibles avec le code actuel. Même problème dans e1_supervised_h5.py:109 (`constraint="b"` → no-op, result_proj == result_raw, et le JSON écrit "has_projection": true de façon trompeuse).

**Preuve :** scripts/e3_bw_em.py:66-69 : VARIANTS = { "V1_posthoc_B": dict(constraint="b", ...), "V2_posthoc_A": dict(constraint="a", ...), "V3_GEM_B": dict(constraint="b", constraint_each_iter=True) } — vs prg/learning/supervised.py:258 : `if constraint == "ab":` (aucun else/raise). Docstring fit_supervised : « constraint : None | 'ab' ».

**Suggestion :** Valider la valeur de `constraint` dans fit_supervised/_apply_constraints (raise ValueError pour toute chaîne ≠ {'ab', None}), puis soit mettre à jour e3_bw_em.py/e1_supervised_h5.py vers la nouvelle API ('ab'), soit les supprimer/archiver avec leurs résultats legacy.

**Ajustement de sévérité proposé par les vérificateurs :** Keep critical, or at most downgrade to high. The library runtime is unaffected (no wrong numbers in prg/ itself), but for a paper-accompanying research repo this silently invalidates Table 3 and Figure 3: re-running the committed scripts would produce four identical variants while the committed JSON/figures show distinct ones, and nothing errors or warns. Silent irreproducibility of published experimental results under the CORRECTION lens justifies critical. | Keep critical (defensible) or at minimum high. For a paper-accompanying research codebase, the paper's Table 3 / Figure 3 variant comparison is silently non-reproducible with the current code, and the API accepts any constraint string as a silent no-op (this also bites anyone still passing 'lehmann' post-rename). Mild counterweights: the committed artifacts were generated correctly under the old API (the published numbers are not themselves wrong), the library core and CLI are unaffected, and the e1 sub-issue is cosmetic (result_proj is never used numerically). If the team reserves 'critical' for wrong-results-as-committed or runtime-correctness bugs, downgrade to high; otherwise critical stands.

### ✅ [HIGH] fill_placeholders réécrit le .tex en place et détruit les \ph{} : les chiffres narratifs du papier sont désormais figés et inrafraîchissables

`prg/experiments/fill_placeholders.py:297-304` — statut : confirmed (2 vote(s)) — catégorie : Reproductibilité / chiffres du papier

fill_placeholders remplace définitivement chaque \ph{clé} par sa valeur dans paper/sections/06_experiments.tex (tex_path.write_text). L'opération est à usage unique : une fois exécutée, les marqueurs disparaissent et un second passage avec de nouveaux CSV ne peut plus rien mettre à jour. C'est déjà le cas : il ne reste que 3 \ph{} dans 06_experiments.tex (uniquement les messages de repli des figures), et la narration contient maintenant des chiffres codés en dur (« NEES H5-exact: 0.980, IMM-approx: 1.002 », « 309 vs 425 µs », « LB pass ≥ 92% », « BIC̄₂ = 1846.8 », « 0.24 vs 0.29 », « ε̂_b = 0.2189 », « 0.06 vs 0.14 », « -30.09 vs -32.38 » dans 07_real_data.tex). Comme les CSV sources (data/experiments/*.csv) et les tables générées (paper/figures/generated/*) sont GITIGNORÉS, toute régénération des tables \input{} peut diverger silencieusement de la narration figée — incohérence tableau/texte indétectable.

**Preuve :** fill_placeholders.py:302-304 : `tex_path.write_text(new_text, ...)` ; grep \ph{ → 3 occurrences restantes (lignes 184, 259, 332, toutes des fallbacks de figures) ; 06_experiments.tex:166-169 : « H5-exact is approximately 1.4× faster (309 vs 425 µs per step at N=2000) » en dur ; `git check-ignore data/experiments/mc_results.csv paper/figures/generated/tab_filter_M1.tex` → les deux ignorés.

**Suggestion :** Remplacer la substitution destructive par la génération d'un fichier de macros LaTeX (\newcommand{\valNEESh5}{0.980}...) inclus par le papier, et committer les CSV de résultats (ou au minimum un hash/manifeste de protocole) pour tracer les chiffres.

**Ajustement de sévérité proposé par les vérificateurs :** Keep high. This is reproducibility/paper-integrity rather than runtime correctness, which could argue for medium, but the divergence is near-certain to trigger: the confirmed imm_general bug fix will change regenerated tables while the hand-frozen narrative (NEES 1.002, 425 µs, etc.) cannot be refreshed, and neither the CSVs nor the .tex are under version control to detect or recover from it. | Keep high. The issue is irreversible (paper/ untracked in git, no recovery of placeholders), silent (no marker remains to flag staleness), and the trigger is already queued: the confirmed imm_general bug fix from wave 1 will force re-running the benchmarks, after which regenerated \input{} tables will diverge from the frozen narrative with no detection mechanism. For an audit whose deliverable is the paper, that is a high-impact reproducibility/integrity defect. If the audit scale reserved high strictly for runtime/math errors, medium would be the floor — but high is the better calibration here.

### ✅ [HIGH] Labels de régime ENSO dérivés de l'ONI : look-ahead d'un mois et circularité avec l'observable Y, biaisant le test (H5) E1 vers la non-rejection

`scripts/build_enso_csv.py:104-107` — statut : confirmed (2 vote(s)) — catégorie : Look-ahead / circularité des labels

Le régime au mois n est seuillé sur l'ONI, qui est la moyenne glissante CENTRÉE sur 3 mois de l'anomalie Niño 3.4 : r_n dépend donc de Y_{n+1} (look-ahead d'un mois), et r_{n+1} — utilisé pour stratifier le test de Fisher H0:B(k)=0 dans run_real_data.py:105-124 (mask = rs[1:] == k) ainsi que pour l'ajustement supervisé E1/E2 — embarque Y_n, Y_{n+1} et Y_{n+2}. Double problème : (1) les paramètres supervisés du train utilisent de l'information future ; (2) circularité : le régime est une fonction déterministe lissée de l'observable Y lui-même ; conditionner sur r_{n+1} absorbe une partie du contenu prédictif de Y_n pour X_{n+1}, ce qui biaise mécaniquement le test vers « B(k)=0 non rejeté » — précisément la conclusion mise en avant dans 07_real_data.tex (« p-value uniformly above 0.35 ... cannot reject (H5) »). De plus le seuillage instantané strict (oni < -0.5 / > 0.5, valeur ±0.5 exacte classée Neutre) diffère de la convention NOAA (≥ +0.5 / ≤ -0.5 sur 5 saisons consécutives).

**Preuve :** build_enso_csv.py:23 : « oni.txt Oceanic Niño Index (3-month running mean of Niño 3.4) » ; :105-107 : `df["regime"]=1 ; df.loc[df["oni"] < -0.5, "regime"]=0 ; df.loc[df["oni"] > 0.5, "regime"]=2` ; run_real_data.py:106 : `mask = rs[1:] == k` puis F-test de B=0.

**Suggestion :** Au minimum, documenter le look-ahead dans le papier ; mieux : décaler les labels (utiliser l'ONI du mois n-1 pour r_n) ou utiliser une moyenne trailing, et vérifier la robustesse du test E1 à ce décalage. Discuter la circularité régime↔Y dans l'interprétation du test (H5).

**Ajustement de sévérité proposé par les vérificateurs :** Keep at high. The robustness check shows the look-ahead/circularity materially flips the paper's central section-7 empirical conclusion (non-rejection of H5 becomes rejection for at least one regime under a one-month label shift), which justifies high severity for a correction-lens audit of a research paper. Not critical because it affects the empirical validation narrative, not the correctness of the filter mathematics or library code, and standard ENSO labeling conventions partially mitigate the framing (though not the test's validity). | keep high — confirmed by counterfactual experiment: removing the one-month look-ahead flips the largest regime stratum from p=0.372 to p=0.018 (rejection) and the pooled test rejects at p=0.006, so the paper's §7 headline conclusion depends on the circular label construction

### ✅ [HIGH] Protocole EM : défauts du script (100 runs, n_inits=10, N jusqu'à 5000) ≠ CSV commis localement (10 runs, n_inits=5, N∈{500,2000}) ≠ em_run.log (30 runs, 3 modèles)

`prg/experiments/run_em.py:69-75` — statut : confirmed (2 vote(s)) — catégorie : Reproductibilité / dérive de protocole

Les constantes DEFAULT_N_RUNS=100, DEFAULT_N_INITS=10, DEFAULT_N_LIST=(500,2000,5000) et le docstring (« PH et GEM, each with n_inits=10 restarts ») ne correspondent pas aux données qui ont servi au papier : em_results.csv contient 10 seeds × N∈{500,2000} avec 5 restarts stockés (vérifié : `n_inits stored: [5]`), conformément au texte du papier (« n_init = 5 ... 10 MC runs »). Le log adjacent em_run.log décrit pourtant un AUTRE run (« 3 model(s) × 2 N values × 30 runs ... n_inits=5 ») : le CSV a été écrasé par un run ultérieur sans trace de la commande exacte. La commande documentée `python -m prg.experiments.run_em` (sans options) écraserait em_results.csv avec un protocole différent de celui du papier. Aucun de ces fichiers n'est versionné (data/experiments/ gitignoré sauf .gitkeep).

**Preuve :** run_em.py:70-72 : `DEFAULT_N_LIST=(500,2000,5000); DEFAULT_N_RUNS=100; DEFAULT_N_INITS=10` ; em_results.csv : groupby(N,variant).count → 10 partout, N∈{500,2000} ; em_run.log ligne 1 : « 3 model(s) × 2 N values × 30 runs × 2 variants = 360 rows (n_inits=5) » ; 06_experiments.tex : « n_init = 5 ... Results are on model M1, N∈{500, 2000}, 10 MC runs ».

**Suggestion :** Aligner les défauts du script sur le protocole du papier (ou inversement), écrire un manifeste (arguments CLI + git SHA + date) à côté de chaque CSV, et committer les CSV utilisés pour le papier.

**Ajustement de sévérité proposé par les vérificateurs :** Severity 'high' is appropriate for a paper-accompanying research codebase: the published results cannot be reproduced from the documented command, the data backing the paper is unversioned and was already silently overwritten once (log vs CSV mismatch proves it). Note the published numbers themselves are consistent with the on-disk CSV, so this is a provenance/reproducibility defect, not a correctness defect in the results — if the audit scale reserves 'high' for wrong published numbers, 'medium-high' would also be defensible.

### ✅ [MEDIUM] Baseline Hamilton : les prédictions de X utilisent les probabilités LISSÉES (défaut statsmodels) calculées sur train+test → look-ahead, contrairement au commentaire « filtered probs »

`scripts/baselines/hamilton_msar.py:114-115` — statut : confirmed (2 vote(s)) — catégorie : Baseline / look-ahead

`fitted_by_regime = res_full.predict()` appelle MarkovSwitchingResults.predict sans argument `probabilities` ; le défaut statsmodels est 'smoothed' (vérifié dans la source installée : `if probabilities is None or probabilities == 'smoothed': results = self.smooth(...)`). Les probabilités lissées sont calculées sur l'échantillon complet [train, test], donc la « one-step-ahead filtered mean » annoncée (commentaire ligne 112-114 : « take weighted sum with filtered probs ») utilise en réalité de l'information future : le MSE test de Hamilton (1.1675 dans table3.json) est non-causal. Le biais avantage ici la baseline (pas la méthode du papier), mais rend la colonne MSE de Table 3 incomparable avec les filtres GSS strictement causaux. Accessoirement : le paramètre `seed` de fit() (ligne 48) n'est jamais utilisé malgré le commentaire « we pass a seed via em_algorithm init », et `P_T` (ligne 104) est du code mort.

**Preuve :** hamilton_msar.py:114 : `fitted_by_regime = np.asarray(res_full.predict())  # shape (N_full,)` précédé du commentaire « take weighted sum with filtered probs » ; statsmodels markov_switching.predict : défaut probabilities=None → smoothed_joint_probabilities.

**Suggestion :** Appeler `res_full.predict(probabilities='filtered')` (ou 'predicted' pour du vrai one-step-ahead), supprimer le paramètre seed inutilisé et la variable P_T.

**Ajustement de sévérité proposé par les vérificateurs :** medium is well calibrated: it is a real scientific-validity defect in a published comparison table (non-causal baseline metric labeled as causal), but the bias favors the baseline, so the paper's headline conclusion (GSS V0 beats Hamilton, 0.931 vs 1.167) is conservative and survives; not high severity since no result flips, not low since a reported number is wrong relative to its stated definition. | medium → low. Le bug de look-ahead est réel et le correctif proposé (probabilities='filtered' ou 'predicted') est juste, mais l'artefact contaminé (results/e3/table3.json/.tex) n'est pas inclus dans le papier (la section real-data est ENSO avec tab_enso_*), et le biais joue contre la méthode du papier (Hamilton perd même avantagé). Re-qualifier en medium seulement si l'expérience S&P500/VIX est (ré)intégrée au papier.

### ✅ [MEDIUM] tab_em_basin : la légende décrit « fraction des runs atteignant le bassin avec 5 restarts » mais la valeur affichée est la fraction moyenne de restarts dans le bassin — contradiction avec tab_em_restarts dans le même papier

`prg/experiments/make_figures.py:603-667` — statut : confirmed (2 vote(s)) — catégorie : Correction des métriques / incohérence interne

make_tab_em_basin affiche `g["basin_rate"].mean()` où basin_rate (run_em.py:270-277) est la proportion des n_inits restarts d'UN run dont la LL est dans le bassin. La légende générée affirme pourtant « basin selection rate (fraction of MC runs reaching the best LL basin with n_init=5 restarts) » — ce qui correspondrait à la métrique « au moins un restart dans le bassin » calculée par make_tab_em_restarts. Résultat dans le papier : tab_em_basin affiche PH N=2000 → 98.0% alors que tab_em_restarts affiche pour n_init=5 → 100.0% pour exactement la même configuration ; le lecteur voit deux « basin rates » contradictoires. Le texte du papier (« Both variants achieve high basin selection rates (96–100%) ») hérite de l'ambiguïté.

**Preuve :** make_figures.py:629-631 : `br = g["basin_rate"].mean()` avec caption lignes 637-642 « fraction of MC runs reaching the best LL basin » ; run_em.py:274-275 : `basin_rate = float(np.mean([ll >= threshold for ll in all_lls]))` (fraction de restarts) ; tab_em_basin.tex : « 2000 & 98.0% » vs tab_em_restarts.tex : « 5 & 100.0% ».

**Suggestion :** Soit corriger la légende (« mean fraction of restarts in the best basin »), soit calculer réellement la fraction de runs avec ≥1 restart dans le bassin (comme tab_em_restarts) pour unifier les deux tables.

**Ajustement de sévérité proposé par les vérificateurs :** medium (unchanged) — caption/metric mismatch only, no algorithmic error, but it yields a directly observable 98.0% vs 100.0% contradiction between two tables of the same paper for the identical configuration, which a reviewer could flag. | none — medium is correctly calibrated: it is a caption/metric mislabeling, not a computational bug (the displayed numbers are correct under the right definition), but it produces two directly contradictory 'basin rates' for the same configuration within one section of a scientific paper, which a referee can spot via the monotonicity violation; the fix is trivial.

### ✅ [MEDIUM] RMSE : le code normalise par N·q, le papier définit sqrt((1/N)Σ‖·‖²) — les valeurs M2 (q=2) de Table 3 sont 1/√2 de la définition affichée

`prg/experiments/metrics.py:90-114` — statut : confirmed (2 vote(s)) — catégorie : Correction des métriques / cohérence papier-code

compute_rmse retourne sqrt(Σ‖x−x̂‖²/(N·q)) (docstring et code), tandis que 06_experiments.tex:144-145 définit « RMSE: √((1/N)Σ‖x̂−x‖²) » sans le facteur 1/q. Pour M1/M3 (q=1) c'est identique, mais pour M2 (q=2) les RMSE publiés (0.4595 vs 0.4616) sont √2 fois plus petits que la définition donnée. La comparaison h5 vs imm n'est pas affectée (même facteur), mais la définition imprimée est fausse pour Table 3 et pour toute comparaison externe.

**Preuve :** metrics.py:94 : « RMSE = sqrt( (1 / (N·q)) · Σ ... ) » et :114 : `np.sqrt(np.sum((x_true - x_est) ** 2) / (N * q))` ; 06_experiments.tex:144-145 : « \sqrt{\frac{1}{N}\sum_{n=1}^{N} \|\hat x_{n|n} - x_n\|^2} ».

**Suggestion :** Ajouter le facteur 1/q dans la formule du papier (comme pour le NEES qui l'a déjà), ou retirer la normalisation par q du code et regénérer Table 3.

**Ajustement de sévérité proposé par les vérificateurs :** medium is appropriate (arguably medium-low): the printed metric definition is wrong for the M2 row, breaking external reproducibility/comparison of those two numbers, but no internal conclusion or method ranking changes. The one-line fix (add 1/q to the paper formula, mirroring NEES) is correct; note the affected table is Table IV, not Table 3. | medium (unchanged) — correctly calibrated: definitional error visible in print affecting M2 in Table 3 and external comparability, but no impact on the paper's internal h5-vs-imm conclusions and trivially fixable in the LaTeX.

### ✅ [MEDIUM] Le « CPU (µs/step) » publié inclut le coût du simulateur et des appends Python, pas seulement le filtre ; la légende code en dur « Apple M2 Pro »

`prg/experiments/run_simulations.py:162-173` — statut : confirmed (2 vote(s)) — catégorie : Correction des métriques / mesure CPU

Dans run_one_trial, le chronomètre englobe l'itération du générateur GSSSimulator (tirages aléatoires, construction des échantillons) et les list.append, en plus de filt.step. Les chiffres µs/step de Table 2 (et la narration « H5-exact is approximately 1.4× faster (309 vs 425 µs) ») sont donc gonflés d'un surcoût commun aux deux modes — le ratio réel filtre-seul est plus grand que 1.4×, et la valeur absolue ne mesure pas le filtre. Par ailleurs make_figures.py:251 code en dur « single core, Apple M2 Pro » dans la légende générée, qui devient fausse dès que les CSV sont régénérés sur une autre machine.

**Preuve :** run_simulations.py:162-173 : `t0 = time.perf_counter(); for _, _r, x, y in sim: result = filt.step(y); ...append...; cpu_s = time.perf_counter() - t0` ; make_figures.py:248-252 : caption « CPU in \textmu s per step (single core, Apple M2 Pro) ».

**Suggestion :** Chronométrer uniquement filt.step (accumuler dt autour de l'appel), pré-générer la trajectoire avant la boucle, et paramétrer la mention matérielle (platform.machine() ou argument CLI).

**Ajustement de sévérité proposé par les vérificateurs :** Keep medium. The bias inflates published per-step CPU numbers in a paper table captioned as a filter benchmark, but it does not invalidate the qualitative conclusion (H5-exact remains faster; the true ratio is even larger than the reported 1.4×). The hardcoded "Apple M2 Pro" caption is a minor reproducibility issue bundled into the same fix. | keep medium (publication-bound mislabeled measurement; bias is common-mode and conservative, no conclusion flips — do not raise, but low would underweight the paper impact)

### ✅ [MEDIUM] Producteurs en chemins relatifs au CWD vs consommateurs ancrés au dépôt : risque de CSV orphelins et de figures générées depuis des données périmées

`prg/experiments/run_simulations.py:65` — statut : confirmed (2 vote(s)) — catégorie : Reproductibilité / chemins et ordre d'exécution

run_simulations/run_supervised/run_em écrivent par défaut dans `pathlib.Path("data")/"experiments"` RELATIF au répertoire courant, alors que make_figures.py:70-71 et fill_placeholders.py:32-34 lisent REPO_ROOT/data/experiments (absolu, dérivé de __file__). Lancé depuis un autre CWD, un run dépose ses CSV ailleurs et make_figures consomme silencieusement l'ancien mc_results.csv du dépôt (il ne signale que l'absence de fichier, pas sa fraîcheur). Plus largement, la chaîne run_simulations → make_figures → fill_placeholders et e3_bw_em → e3_add_hamilton → figure3/figure4 n'a aucune vérification d'ordre ni de fraîcheur : e3_add_hamilton.py:114 plante (ou modifie en place) si table3.json n'existe pas, et tous les fichiers de sortie (mc_results.csv, em_results.csv, table*.tex, *.json) portent des noms fixes écrasés sans avertissement — l'écrasement attesté par l'incohérence em_run.log/em_results.csv.

**Preuve :** run_simulations.py:65 : `DEFAULT_OUT_DIR = pathlib.Path("data") / "experiments"` (relatif) ; make_figures.py:70-71 : `REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]; DEFAULT_IN = REPO_ROOT / "data" / ...` (absolu) ; run_simulations.py:309 : `df.to_csv(out_path, index=False)` sans garde d'écrasement.

**Suggestion :** Ancrer aussi les répertoires de sortie des runners sur REPO_ROOT, horodater ou hasher les sorties (ou refuser d'écraser sans --force), et fournir un Makefile/driver qui encode l'ordre des étapes.

**Ajustement de sévérité proposé par les vérificateurs :** medium (inchangé) — pas un bug de calcul, mais hygiène de pipeline avec conséquence déjà matérialisée : em_results.csv du dépôt provient d'un run partiel M1-only incohérent avec em_run.log, et les tables §6.4 du papier seraient régénérées depuis ces données sans aucun avertissement. | none — medium is appropriate

### ✅ [LOW] make_tab_bic re-pénalise la MÊME log-vraisemblance (modèle K=2) pour K=1..4 : K=1 gagne mécaniquement ; la table générée contient des \ph{XX%} littéraux

`prg/experiments/make_figures.py:297-343` — statut : confirmed (1 vote(s)) — catégorie : Correction / table BIC factice

La « sélection d'ordre » réutilise les log-vraisemblances du filtre h5_exact sous le VRAI modèle K=2 pour tous les K testés, en ne changeant que la pénalité d_H5(K) : le BIC est alors strictement croissant en K et K=1 a toujours le BIC minimal (1770.8 < 1846.8 < 1938.1 < 2044.5 dans tab_bic.tex), ce qui contredirait la sélection du vrai K=2. Le docstring l'assume (« only a rough BIC approximation ») et le papier n'\input{} pas tab_bic.tex (il ne cite que BIC̄₂=1846.8 et défère la vraie sélection), mais le fichier généré dans paper/figures/generated/ contient des cellules \ph{XX\%} et une colonne pct_sel jamais calculée — risque d'inclusion accidentelle d'une table trompeuse.

**Preuve :** make_figures.py:332 : `bic_vals = sub["log_lik"].dropna().apply(lambda ll: d * np.log(N_val) - 2.0 * ll)` (même ll pour tous K_test) ; :338-339 : `r"\ph{XX\%}"` ; tab_bic.tex : « 1 & 1770.8 & \ph{XX\%} ».

**Suggestion :** Supprimer la génération de tab_bic.tex tant que les fits EM par K ne sont pas disponibles, ou la marquer explicitement DRAFT dans le nom de fichier.

**Ajustement de sévérité proposé par les vérificateurs :** keep low — accurate finding, but it is a self-documented placeholder not referenced by the paper; the only live risk is accidental future inclusion of a stale artifact

### ✅ [LOW] compute_ljung_box retourne p=1.0 (test « réussi ») si le test échoue sur toutes les composantes ; côté tables, les NaN comptent comme « rejet »

`prg/experiments/metrics.py:209-225` — statut : confirmed (1 vote(s)) — catégorie : Correction des métriques / cas dégénérés

min_pval est initialisé à 1.0 ; si acorr_ljungbox lève une exception ou produit NaN pour chaque composante, la fonction renvoie 1.0 au lieu de NaN — un run pathologique est compté comme parfaitement blanc dans « LB pass % ». Inversement, dans make_figures.py:223 et fill_placeholders.py:108, `(g["lb_pval"] > 0.05).mean()` compte les NaN (trials échoués) comme rejets, dégonflant le taux de passage au lieu de les exclure. Les deux conventions opposées coexistent. Aussi : pour s>1 le min des p-values composante par composante sans correction de multiplicité rend le test plus sévère pour M2 que pour les modèles scalaires (documenté, mais affecte la comparabilité inter-modèles de la colonne LB%).

**Preuve :** metrics.py:209 : `min_pval = 1.0` puis retour inconditionnel :225 ; make_figures.py:223 : `lb_pass = float((g["lb_pval"] > 0.05).mean() * 100)` sans dropna().

**Suggestion :** Retourner NaN si aucune composante n'a produit de p-value finie ; utiliser `g["lb_pval"].dropna()` dans les agrégations et rapporter le nombre de runs exclus.

**Ajustement de sévérité proposé par les vérificateurs :** Keep low, but note it sits at the high end: combined with the confirmed imm_general divergence bug, diverged runs returning p=1.0 can silently inflate the LB pass % of the buggy mode in paper-reported tables, masking the divergence; if that mode's results appear in the paper, medium would also be defensible.

### ✅ [LOW] Accuracy de régime alignée par la permutation optimale choisie SUR le test : borne inférieure mécanique à 0.5 pour K=2, valeurs absolues gonflées

`scripts/e3_bw_em.py:90-105` — statut : confirmed (1 vote(s)) — catégorie : Métriques / label-switching

Le label-switching est bien géré (permutation brute-force pour acc, ARI invariant par permutation), mais la permutation est choisie pour MAXIMISER l'accuracy sur l'ensemble de test lui-même (e3_bw_em.py:_best_regime_alignment, run_real_data.py:230-239 best_perm_acc_ari, e3_add_hamilton.py:43-53) : pour K=2, acc = max(a, 1-a) ≥ 0.5 par construction (les 0.505-0.58 de table3.json sont indistinguables du hasard), et pour K=3 l'alignement oracle ajoute un biais optimiste. Le biais est appliqué identiquement à toutes les méthodes (Hamilton compris), donc le classement reste comparable, mais les valeurs absolues d'accuracy publiées (tab_enso_em « after optimal regime-permutation alignment ») sont optimistes. Alternative plus propre : fixer la permutation sur le train, ou ne rapporter que l'ARI.

**Preuve :** e3_bw_em.py:99-104 : boucle sur permutations maximisant `np.mean(perm_arr[r_hat] == r_true)` sur r_true_test ; run_real_data.py:231-238 idem.

**Suggestion :** Aligner les régimes via les paramètres estimés sur le train (ex. ordonner par variance ou par moyenne de Y) ou documenter explicitement la borne 1/K… max dans les légendes.

**Ajustement de sévérité proposé par les vérificateurs :** keep low — uniform across methods, rankings unaffected, alignment disclosed in the caption; only the absolute accuracy values are optimistically biased

### ✅ [LOW] e3_add_hamilton mute results/e3/table3.json en place et réécrit table3.tex avec un format différent de celui d'e3_bw_em

`scripts/e3_add_hamilton.py:114-118` — statut : confirmed (1 vote(s)) — catégorie : Reproductibilité / ordre d'exécution

Le script exige que e3_bw_em.py ait été exécuté avant (crash sinon sur read_text), modifie table3.json en place (dédoublonnage par nom puis append), et regénère table3.tex avec un layout différent (séparateur \midrule avant Hamilton) de celui produit par e3_bw_em.py:259-277 — le contenu final de table3.tex dépend donc de QUEL script a tourné en dernier. Aucun des deux fichiers n'est versionné. Dépendance d'ordre implicite typique signalée dans l'audit (point 1).

**Preuve :** e3_add_hamilton.py:114 : `data = json.loads(args.table.read_text(...))` (pas de garde d'existence) ; :116-118 : filtrage + append + write_text sur le même fichier ; deux émetteurs concurrents de table3.tex.

**Suggestion :** Faire de l'ajout Hamilton une option de e3_bw_em.py (un seul émetteur de table3.*), ou écrire un fichier séparé table3_hamilton.tex.

**Ajustement de sévérité proposé par les vérificateurs :** keep low

### ✅ [LOW] Téléchargements non épinglés : fin yfinance exclusive (le « 2004-2024 » s'arrête au 2024-12-30), VIX FRED et NOAA sans version ; l'ONI historique change avec les mises à jour de période de base

`scripts/fetch_sp500_vix.py:51-57` — statut : confirmed (1 vote(s)) — catégorie : Données / versionnage

(1) yf.download(end=...) est EXCLUSIF : avec --end 2024-12-31 la dernière séance est le 2024-12-30, alors que la doc et le papier annoncent 2004-2024 inclus — écart d'un jour bénin mais incohérent avec « --end ». (2) Aucune version/date de téléchargement n'est enregistrée dans les CSV produits ; les valeurs ajustées Yahoo et les fichiers NOAA évoluent (l'ONI est recalculé rétroactivement à chaque changement de période de base ERSSTv5, tous les ~5 ans : les labels de régime historiques de build_enso_csv.py peuvent changer entre deux téléchargements). Le risque est mitigé par le fait que sp500_vix.csv, enso_sst.csv et les .txt NOAA sont commités, et que build_enso_csv ne retélécharge que si absent — mais un --force-download silencieux peut décaler tous les régimes sans alerte.

**Preuve :** fetch_sp500_vix.py:51-57 : `yf.download("^GSPC", start=start, end=end, ...)` (end exclusif) ; build_enso_csv.py:52-60 : download si absent, aucune empreinte de version ; commentaire fetch_sp500_vix.py:27 : « The downloaded CSV is meant to be committed for reproducibility » (les sources NOAA n'ont pas l'équivalent).

**Suggestion :** Enregistrer date de téléchargement + sha256 des fichiers bruts dans un manifeste commité ; ajouter un jour à end pour yfinance ou documenter l'exclusivité ; afficher un diff de labels de régime après re-téléchargement NOAA.

**Ajustement de sévérité proposé par les vérificateurs :** Keep at low — correctly calibrated. The off-by-one is benign (as the finding admits), and the unpinned-download risk is largely mitigated by committed data files; the residual risk only materializes on an explicit --force-download or fresh fetch. Drop the "le papier annonce 2004-2024" portion of the description: the paper makes no such claim, only script docstrings/comments do.

### ✅ [LOW] Docstrings périmés référencant l'ancienne API de projections ('b', 'a', 'su') et d'anciens protocoles

`prg/experiments/run_supervised.py:197-199` — statut : confirmed (1 vote(s)) — catégorie : Documentation périmée

run_supervised_trial documente « Elements are None, 'b', 'a', or 'su' » alors que la CLI (choices=["none","ab"]) et le code n'acceptent que None/'ab' ; supervised_run.log montre que les anciennes projections ont réellement servi pour des CSV antérieurs. De même, le docstring de run_em.py (« n_inits=10 restarts », N_RUNS=100) ne décrit pas le protocole du papier (5 restarts, 10 runs), et figure3_em_restarts.py documente encore les valeurs legacy (~-66k pour V3) irréproductibles avec l'API actuelle. Ces docstrings périmés masquent la dérive d'API détectée dans la trouvaille critique sur e3_bw_em.py.

**Preuve :** run_supervised.py:197-199 : « H5 projection choices ... Elements are None, 'b', 'a', or 'su'. » vs :519-524 : `choices=["none", "ab"]` ; figure3_em_restarts.py:10-13 : « its absolute log-likelihood values fall on a very different scale (~ -66k) ».

**Suggestion :** Mettre à jour tous les docstrings vers l'API {None,'ab'} et le protocole effectivement publié.

**Ajustement de sévérité proposé par les vérificateurs :** Keep at low. Documentation-only, but correctly flagged: the legacy constraint strings are silently ignored by the current implementation (no error), so the stale docstrings can lead users to believe a projection was applied when it was not — low is appropriate, not info.

### ✅ [INFO] Toutes les métriques test de la Table tab_enso_em (E3) transitent par le filtre mode='imm_general'

`prg/experiments/run_real_data.py:266-279` — statut : confirmed (1 vote(s)) — catégorie : Comparabilité / dépendance au bug imm_general

Sans re-rapporter le bug déjà identifié en vague 1 (Γ(j,k) non-PSD du mode imm_general), il faut noter la conséquence côté expériences : dans run_e3, les test_LL, test_MSE, acc et ARI des trois variantes EM (V0/V1/V2) du papier §7 sont tous calculés via GSSFilter(params, mode="imm_general") avec warnings supprimés (warnings.simplefilter("ignore")). Si le bug du mode imm_general affecte la vraisemblance ou π_n, toute la Table tab_enso_em (et regime_trace.csv → make_figures_real) en hérite. La comparaison entre variantes reste interne au même filtre, mais les valeurs absolues et la comparaison avec kalman_k1/h5_exact de la Table tab_enso_filter peuvent être contaminées.

**Preuve :** run_real_data.py:268 : `f = GSSFilter(params_from_dict(params), mode="imm_general")` dans run_e3, sous `warnings.simplefilter("ignore")`.

**Suggestion :** Une fois le bug imm_general corrigé, regénérer e3_table.json/tab_enso_em.tex et regime_trace.csv ; envisager d'évaluer aussi les variantes EM via h5_exact lorsque les paramètres satisfont (H5).

**Ajustement de sévérité proposé par les vérificateurs :** None — [info] is well calibrated. It is a propagation/consequence note conditional on the already-reported imm_general bug, not an independent defect; flagging it as info-level guidance to regenerate artifacts after the fix is exactly right.

### ✅ [INFO] E2 S&P500 : h5_exact est évalué avec des paramètres OLS non contraints (incompatibles H5), contrairement au protocole ENSO du papier

`scripts/e2_filter_comparison.py:157-169` — statut : confirmed (1 vote(s)) — catégorie : Comparabilité des baselines

Dans e2_filter_comparison.py, un seul fit OLS non contraint sert aux deux modes : GSSFilter(params, mode="h5_exact") tourne donc sur un modèle qui viole (H5) (warning RuntimeWarning supprimé ligne 166-168), ce qui dégrade artificiellement la méthode du papier sur ce dataset — le contraire du biais habituel, mais incohérent avec run_real_data.run_e2 (ENSO, §7) qui donne au mode h5_exact les paramètres H5-projetés et compare aussi imm_general sur les deux jeux de paramètres. Les résultats S&P500 (results/e2, non versionnés) ne sont pas dans le papier, mais toute comparaison croisée entre les deux scripts E2 serait trompeuse. Le log-lik test des trois filtres reste calculé sur la même variable (Y), avec initialisations différentes (priors moments d'échantillon pour kalman_k1 vs priors du modèle pour GSS) — comparables au premier ordre.

**Preuve :** e2_filter_comparison.py:157 : `fit = fit_supervised(..., constraint=None)` puis :166-169 : `gss_h5 = GSSFilter(params, mode="h5_exact")` sur ces mêmes params ; run_real_data.py:190-207 : fit_raw ET fit_h5, h5_exact servi par p_h5.

**Suggestion :** Aligner e2_filter_comparison.py sur le protocole de run_real_data (ajouter le fit constraint='ab' et la ligne h5_exact_h5fit) ou retirer le script s'il est obsolète.

**Ajustement de sévérité proposé par les vérificateurs :** Keep at info. The script's docstring openly documents the bias ("biased if B≠0"), results are not in the paper and not versioned, so no published claim is affected; but the missing h5_exact-on-H5-fit arm and the divergence from the ENSO protocol justify the recommendation to align the script on run_real_data.run_e2 or retire it.

### ✅ [INFO] Fig. 3 (convergence EM) : la courbe moyenne est tronquée à la longueur du run le plus court

`prg/experiments/make_figures.py:563-579` — statut : confirmed (1 vote(s)) — catégorie : Figures / agrégation

make_fig_em_convergence empile les historiques LL de tous les seeds et coupe à min_len avant de moyenner : si un run converge tôt (peu d'itérations), la courbe moyenne de TOUTES les graines s'arrête là, masquant la fin de convergence des autres. Avec le protocole actuel (presque tous les runs épuisent le budget de 50 itérations, cf. texte du papier), l'effet est faible, mais la moyenne deviendrait trompeuse avec une tolérance plus serrée. Idem, seuls les n_curves=5 premiers seeds sont tracés en fin (« Thin lines = individual runs ») sans le préciser comme sous-échantillon dans la légende du papier (« 10 MC runs »).

**Preuve :** make_figures.py:571-573 : `min_len = min(len(v) for v in ll_mat); ll_arr = np.array([v[:min_len] for v in ll_mat])` puis mean(axis=0).

**Suggestion :** Prolonger chaque historique par sa dernière valeur (padding « converged ») avant de moyenner, et indiquer dans la légende que seules 5 trajectoires individuelles sont montrées.

**Ajustement de sévérité proposé par les vérificateurs :** None — [info] is correctly calibrated. The averaging artifact is currently sub-iteration-scale and the caption mismatch is a minor presentation issue; it would only become misleading with a tighter tolerance or harder models, which the finding itself states.

### ✅ [INFO] E2 ENSO : traces x_hat/π collectées puis jetées ; E1 sans garde n<5 dans e1_supervised_h5 ; divers petits écarts

`prg/experiments/run_real_data.py:211-217` — statut : confirmed (1 vote(s)) — catégorie : Code mort / sorties non utilisées

Petits points relevés en lecture intégrale : (1) run_e2 collecte `traces` (x_hat, π par filtre) mais main() ne les écrit jamais (seul E3 écrit regime_trace.csv) — la comparaison visuelle des filtres E2 est impossible à reproduire sans modifier le code ; (2) e1_supervised_h5.fisher_test_B_zero n'a pas la garde `n < 5` présente dans run_real_data.fisher_B_zero:111 (risque de F-stat sur 3-4 points si un label est rare) ; (3) run_em.py:243 `except (RuntimeError, Exception)` est redondant ; (4) fill_placeholders écrase la clé « EM rmse oracle N{N} » à chaque variante (PH puis GEM) — sans conséquence car la valeur est identique, mais fragile.

**Preuve :** run_real_data.py:196-209 : `traces = {}` passé à run_filter puis retourné dans e2 mais jamais persisté (main:498 ne sérialise que K et scores) ; e1_supervised_h5.py:74-96 : pas de seuil minimal n.

**Suggestion :** Écrire traces E2 dans results/enso/e2_trace.csv ; harmoniser la garde n<5 ; nettoyer l'except redondant.

**Ajustement de sévérité proposé par les vérificateurs :** None — [info] is correctly calibrated. These are code-hygiene and reproducibility nits with no runtime impact; the only minor imprecision is that the n=3 case in claim (2) is in fact caught by the df_resid guard, so the missing guard only matters at exactly n=4 (never reached on the real datasets).

## Hygiène du dépôt

_Audit d'hygiène du dépôt exactIMM (HEAD = af19b48, 10 commits après le tag v0.13.1) : lecture de README.md, Makefile, pyproject.toml, config.toml, .gitignore, des 4 workflows GitHub (lint, tests, build, audit) + dependabot.yml, survol du wiki (8 pages) et inspection git (ls-files, check-ignore, ls-files -i, tags) des dossiers data/, results/, logs/, docs/, paper/ et exactIMM.egg-info. L'hygiène git pure est bonne : aucun artefact suivi à tort (egg-info, caches, LaTeX, PDF de test du dossier paper/ sont tous ignorés et non trackés), les données réelles nécessaires sont présentes, et les versions affichées (pyproject = CITATION.cff = README = wiki = tag v0.13.1) sont cohérentes. Les vrais problèmes sont opérationnels : le job Security audit échoue chaque semaine depuis 3+ semaines (pip-audit --strict bute sur le paquet editable local — correctif d'une ligne), le parcours d'installation documenté README/Makefile plante sur machine vierge (pytest-qt sans PyQt6, prouvé par un run CI), « make check » ne reproduit pas le format-check de la CI (ça a déjà cassé un build), et les 17 tests GUI ne tournent dans aucun job CI. S'y ajoutent de la doc dérivante (3 comptes de tests différents : 204/209/219), 8 options config.toml mortes sur 11 avec leurs règles .gitignore orphelines, une dépendance hmmlearn jamais importée, et le risque structurel assumé mais notable que les sources LaTeX du papier (paper/) ne soient sous aucune gestion de version._

### ✅ [HIGH] Le job 'Security audit' (pip-audit) échoue à chaque exécution hebdomadaire depuis au moins 3 semaines

`.github/workflows/audit.yml:28` — statut : confirmed (2 vote(s)) — catégorie : CI

pip-audit --strict est lancé après 'pip install -e .[dev,gui,paper]'. Le paquet local installé en editable (exactimm 0.13.1) n'existe pas sur PyPI, et --strict transforme ce skip en erreur. Les 3 derniers runs planifiés (25 mai, 1er juin, 8 juin 2026) sont tous en failure : le signal sécurité est rouge en permanence, une vraie CVE dans numpy/PyQt6/yfinance passerait inaperçue.

**Preuve :** Log du run 27133143048 : « ERROR:pip_audit._cli:exactIMM: Dependency not found on PyPI and could not be audited: exactimm (0.13.1) » → exit code 1. gh run list --workflow=audit.yml : failure / failure / failure (2026-06-08, 2026-06-01, 2026-05-25).

**Suggestion :** Remplacer la ligne 28 par « pip-audit --strict --skip-editable » (pip-audit ignore alors le paquet local installé en -e), ou auditer un export figé (pip freeze --exclude-editable > req.txt && pip-audit -r req.txt --strict).

**Ajustement de sévérité proposé par les vérificateurs :** Keep high. The repo's only automated vulnerability-scanning gate has been non-functional for its entire existence (5 consecutive failed runs), meaning real CVEs would go unnoticed; the fix is trivial but the silenced security signal justifies high severity. | Keep high. It is a broken security control rather than an exploitable vulnerability, so not critical, but with Dependabot alerts disabled this is the repo's only CVE scanner and it has been non-functional since inception (5/5 runs failed, never one success), producing zero security signal plus alarm fatigue. Trivial one-line fix.

### ✅ [HIGH] Le parcours d'installation documenté (README quick-start et Makefile) est cassé sur machine vierge : pytest plante au démarrage sans PyQt6

`Makefile:35-45 (et README.md:111-137, 668-671)` — statut : confirmed (2 vote(s)) — catégorie : Documentation/Build

README (« pip install -e \".[dev]\" » puis « pytest ») et Makefile (« make install » = .[dev] seulement, puis « make test » = pytest sans options) installent pytest-qt (extra dev) mais pas PyQt6 (extra gui). Or pytest-qt fait planter pytest au démarrage s'il n'y a aucun binding Qt — exactement l'erreur qui a cassé la CI le 9 juin. La CI a été contournée (tests.yml ajoute -p no:pytest-qt) mais README et Makefile n'ont pas été corrigés ; ça ne se voit pas en local uniquement parce que le .venv du développeur contient déjà PyQt6.

**Preuve :** Run CI 27215432660 (commit a1b8487, avant le contournement) : « ERROR: pytest-qt requires either PySide6, PyQt5 or PyQt6 installed. ##[error]Process completed with exit code 4 ». Commentaire dans tests.yml : « Disable the pytest-qt plugin (it errors at startup without a binding) ». Makefile:35 « install: $(PIP) install -e \".[dev]\" » ; Makefile:44-45 « test: $(PY) -m pytest ».

**Suggestion :** Au choix : (a) ajouter PyQt6 à l'extra [dev] (pytest-qt sans binding est incohérent), (b) documenter « pip install -e \".[dev,gui]\" » dans README/Makefile, ou (c) faire que make test passe -p no:pytest-qt --ignore=tests/*_gui.py quand PyQt6 est absent (import-test dans le Makefile).

**Ajustement de sévérité proposé par les vérificateurs :** Lower from high to medium. The breakage is real and deterministic on a fresh machine, but it is a documentation/packaging defect, not a correctness bug: pytest fails loudly at startup with a self-explanatory message naming the missing packages, and the fix is a one-line pip install. It cannot cause silent wrong results. Medium (rather than low) because the documented quick-start path fails completely for any new contributor or paper reviewer, and README:660-662 explicitly stakes the paper's reproducibility claims on being able to run this suite. | high est défendable et peut être conservé: le parcours d'installation documenté échoue totalement (exit code 4, 0 test) sur toute machine vierge, pour un artefact de recherche dont le README présente la suite de tests comme la validation de reproductibilité du papier. Au pire, un rubric strict réservant « high » aux échecs silencieux/de correction justifierait medium-high, car l'erreur est bruyante et indique elle-même le correctif.

### ✅ [MEDIUM] « make check » prétend la parité CI mais omet « ruff format --check » — ça a déjà cassé la CI

`Makefile:50-59` — statut : confirmed (2 vote(s)) — catégorie : CI/Build

La cible check (« lint + typecheck + tests (CI parity) ») n'exécute que « ruff check », alors que le workflow Lint exécute aussi « ruff format --check . ». L'écart a déjà produit un faux vert local : le commit a1b8487 a fait échouer le job Lint (run 27215432034) et a dû être rattrapé par d86c3c5 « style: apply ruff format (was failing CI format-check) ». S'ajoute une dérive de version : la CI épingle ruff==0.15.12 (lint.yml) alors que [dev] exige seulement ruff>=0.6, et dependabot (écosystème pip) ne met pas à jour un pin inline dans un workflow.

**Preuve :** Makefile:59 « check: lint typecheck test ## Run lint + typecheck + tests (CI parity) » ; lint.yml : « pip install ruff==0.15.12 » puis « ruff format --check . » ; gh run list : Lint failure sur a1b8487, corrigé par d86c3c5.

**Suggestion :** Ajouter « $(PY) -m ruff format --check . » à la cible lint (et « ruff format . » à fmt), et aligner la version : soit épingler ruff==0.15.12 dans [dev], soit installer ruff via l'extra dev dans lint.yml pour n'avoir qu'une source de vérité.

**Ajustement de sévérité proposé par les vérificateurs :** medium is appropriate. Pure dev-tooling issue (argues for low), but the target explicitly advertises "CI parity" and the gap has demonstrably broken CI once already (run 27215432034), with the format/version drift making recurrence likely. Keep medium. | keep medium

### ✅ [MEDIUM] paper/ (sources LaTeX du papier) et docs/ totalement hors gestion de version ; règles LaTeX globales (*.log, *.out) redondantes et risquées

`.gitignore:8, 11, 144-172` — statut : confirmed (2 vote(s)) — catégorie : Hygiène git / Risque de perte

Tout paper/ est ignoré (« kept out of the repo », .gitignore:11) : paper.tex, paper.bib, sections/, plus localement 8 triplets test*.{tex?,aux,log,pdf} et tous les artefacts latexmk — rien n'est suivi (vérifié : git ls-files ne contient aucun fichier paper/, et git check-ignore confirme paper/test2.pdf, paper/paper.log). Conséquences : (1) le manuscrit de recherche n'a AUCUN historique de version ni sauvegarde via ce dépôt — point de défaillance unique pour le livrable principal ; (2) README:630-635 et wiki/Paper-Reproduce.md:54-80 disent que make_figures écrit dans paper/figures/generated/ et que fill_placeholders met à jour paper/sections/06_experiments.tex — sur un clone frais ce fichier n'existe pas (fill_placeholders.py:278 imprime ERROR et rend 0). Par ailleurs la section LaTeX du .gitignore (lignes 144-172) est presque entièrement redondante puisque paper/ et docs/ sont déjà ignorés en bloc, et ses motifs globaux *.log (l.147) et *.out (l.148) ignorent silencieusement TOUT fichier .log/.out n'importe où dans le dépôt (rend la ligne 26 data/experiments/*.log et les lignes 41-42 logs/* partiellement redondantes ; un futur fichier de données .log serait invisible pour git sans avertissement).

**Preuve :** git ls-files | grep paper/ → vide ; git check-ignore -v paper/test2.pdf → « .gitignore:11:paper/ » ; ls paper/ → paper.tex, paper.aux, paper.log, paper.pdf, test2..test8.{aux,log,pdf}, test_table.* ; git check-ignore -v data/experiments/em_run.log → « .gitignore:147:*.log ».

**Suggestion :** Si l'exclusion de paper/ est volontaire, s'assurer qu'il a son propre dépôt git (Overleaf ou autre) et le dire dans le README (« paper/ est un dépôt séparé, non inclus ») ; sinon versionner les .tex/.bib et ne garder que les artefacts dans .gitignore. Restreindre *.log/*.out au scope LaTeX (paper/**/*.log) ou les supprimer puisque paper/ et docs/ sont déjà ignorés en bloc.

**Ajustement de sévérité proposé par les vérificateurs :** Keep medium. The paper/ exclusion is explicitly intentional and external backups (e.g. Overleaf, Time Machine) cannot be ruled out from the repo, which caps it below high; but the documented reproduction workflow silently failing on a fresh clone plus the global *.log/*.out ignore footgun make it more than low/cosmetic. | Keep medium. For a research repo whose stated purpose is to accompany the paper, an unversioned primary deliverable plus documented repro steps that silently exit 0 on a fresh clone justify medium. Downgrade to low only if the author confirms paper/ is managed in a separate repo (e.g. Overleaf) — there is no on-disk evidence of that (no paper/.git, no README mention).

### ✅ [MEDIUM] Angles morts CI : les 17 tests GUI ne tournent dans aucun job, et prg/experiments/ + scripts/ n'ont aucun test

`.github/workflows/tests.yml:36-40` — statut : confirmed (2 vote(s)) — catégorie : CI

Au-delà du skip documenté (-p no:pytest-qt --ignore des 2 fichiers GUI), il n'existe aucun job qui installe [gui] : les 17 tests pytest-qt (tests/test_main_window_gui.py : 10, tests/test_param_panel_gui.py : 7) ne s'exécutent que sur la machine du développeur — une régression GUI ne sera jamais détectée par la CI alors que ces tests ont justement été ajoutés en v0.13.1 comme garde-fous de la contrainte AB. Par ailleurs aucun fichier de tests ne couvre prg/experiments/ (pipelines de reproduction §6/§7 du papier) ni scripts/ ; la couverture CI mesurée sur prg inclut donc gui/ et experiments/ comme zones mortes. Enfin la matrice ne teste que 3.14 (cohérent avec requires-python, simple constat).

**Preuve :** tests.yml:40 « pytest -p no:pytest-qt --ignore=tests/test_param_panel_gui.py --ignore=tests/test_main_window_gui.py ... » avec le commentaire « They still run locally » ; grep -c "def test" → 10 + 7 ; ls tests/ : aucun test_experiments*/test_scripts*.

**Suggestion :** Ajouter un job (ou une entrée de matrice) qui fait « pip install -e .[dev,gui] » avec QT_QPA_PLATFORM=offscreen (et libegl1 sur ubuntu-latest) pour exécuter les 2 fichiers GUI. Ajouter au minimum un test de fumée import+--help pour prg/experiments et scripts.

**Ajustement de sévérité proposé par les vérificateurs :** Keep medium. It is a test-infrastructure gap rather than a runtime bug, but the skipped tests were written specifically as regression guards for the central AB/H5 constraint UI, and the gap is silent (CI stays green). Medium is well calibrated; do not raise or lower. | keep medium (note: rephrase "aucun job n'installe [gui]" to "aucun job n'exécute les tests GUI" — audit.yml installs the gui extra but only runs pip-audit)

### ✅ [LOW] Trois comptes de tests différents dans la doc : 209 (README/Makefile), 204 (wiki), 219 (réel)

`README.md:95, 134, 660, 670, 674 (+ Makefile:11, wiki/Installation.md:46 et 56)` — statut : confirmed (1 vote(s)) — catégorie : Documentation

La suite collecte 219 tests (pytest --collect-only : « 219 tests collected »). Le README annonce « 209 tests » à 5 endroits, le Makefile aussi (ligne 11), et wiki/Installation.md dit « ≈ 45 s, 204 tests » puis « If pytest reports 204 passed ». Le wiki promet même un critère de vérification d'installation qui est faux.

**Preuve :** pytest --collect-only -q → « 219 tests collected in 1.79s » ; README.md:670 « pytest # 209 tests, ~45 s » ; wiki/Installation.md « Run the test suite (≈ 45 s, 204 tests) ».

**Suggestion :** Supprimer le nombre exact des docs (écrire « la suite pytest ») ou le générer ; au minimum aligner README/Makefile/wiki sur 219. Le garde-fou tests/test_no_stale_refs.py pourrait vérifier ce compte.

**Ajustement de sévérité proposé par les vérificateurs :** None — [low] is correctly calibrated: docs-only inconsistency with no code impact, though the wiki's false install-verification criterion ("204 passed") could mildly confuse new users.

### ✅ [LOW] 8 des 11 options de config.toml ne sont lues nulle part (section [output] entière, project_name, random_seed, 3 chemins)

`config.toml:5-20` — statut : confirmed (1 vote(s)) — catégorie : Configuration morte

Seules general.log_level, paths.logs et paths.data_simulated sont lues (prg/simulate.py:251-256, prg/filter/main.py:210-213). Sont mortes : general.project_name, general.random_seed, paths.data_output, paths.data_plot, paths.history_tracker, et toute la section [output] (save_interval, plot_format, dpi — le dpi est codé en dur à 150 dans make_figures.py, make_figures_real.py et plot_panel.py). L'en-tête « Edit this file to tune simulation and I/O parameters » promet des réglages sans effet ; random_seed=42 en particulier laisse croire à un seed par défaut qui n'existe pas (le seed CLI par défaut est None).

**Preuve :** grep cfg dans prg/ : seuls cfg[general][log_level], paths[logs], paths[data_simulated] sont consommés ; grep save_interval|plot_format|history_tracker|data_output|data_plot|random_seed dans prg/, scripts/, tests/ : zéro lecture (seuls des dpi=150 codés en dur).

**Suggestion :** Réduire config.toml aux 3 clés réellement lues (c'est d'ailleurs ce que montre la section Configuration du README, lignes 684-693), ou brancher les options restantes. Supprimer aussi les règles .gitignore associées (voir trouvaille dédiée).

**Ajustement de sévérité proposé par les vérificateurs :** Severity "low" is correctly calibrated: dead configuration and a misleading header/seed value, no functional impact.

### ✅ [LOW] Règles .gitignore mortes : data/output/, data/historyTracker/, data/plot/ n'existent pas (ni leurs .gitkeep négés)

`.gitignore:17-22` — statut : confirmed (1 vote(s)) — catégorie : Hygiène git

Les lignes 17-22 ignorent trois répertoires absents du dépôt et ré-incluent des .gitkeep qui n'existent pas (data/ ne contient que experiments/, real/, simulated/). Ces règles sont le pendant des chemins morts de config.toml.

**Preuve :** .gitignore:17-22 « data/output/* / !/data/output/.gitkeep / data/historyTracker/* / ... » ; ls data/ → experiments, real, simulated uniquement ; git ls-files data/ → aucun .gitkeep pour ces trois chemins.

**Suggestion :** Supprimer ces 6 lignes en même temps que les clés config.toml correspondantes.

**Ajustement de sévérité proposé par les vérificateurs :** None — [low] is correctly calibrated for dead .gitignore housekeeping with no functional impact.

### ✅ [LOW] hmmlearn déclaré dans l'extra [paper] mais jamais importé — dépendance morte compilée à chaque run CI

`pyproject.toml:33-40` — statut : confirmed (1 vote(s)) — catégorie : Dépendances

grep sur prg/, scripts/ et tests/ : aucun « import hmmlearn / from hmmlearn ». Les autres paquets de l'extra sont bien utilisés (yfinance et requests dans scripts/fetch_sp500_vix.py, statsmodels dans prg/experiments/metrics.py:197 et scripts/baselines/hamilton_msar.py, sklearn dans prg/experiments/run_real_data.py et scripts/e3_*.py). hmmlearn n'a pas de wheel cp314 : le job audit.yml le compile depuis les sources à chaque run (« Building wheel for hmmlearn... ~8 s ») et il élargit la surface pip-audit pour rien.

**Preuve :** grep -rln "import hmmlearn|from hmmlearn" prg/ scripts/ tests/ → vide ; log CI audit : « Building wheel for hmmlearn (pyproject.toml): started ... finished » ; pyproject.toml:38 « hmmlearn>=0.3 ».

**Suggestion :** Retirer hmmlearn de l'extra [paper] (le baseline HMM est implémenté maison via scripts/e3_bw_em.py). Mettre aussi à jour la table des extras de wiki/Installation.md, qui en plus décrit dev comme « pytest, pytest-cov » alors qu'il contient pytest-qt, mypy et ruff.

**Ajustement de sévérité proposé par les vérificateurs :** Keep [low] — correctly calibrated. Dead optional dependency with small CI cost (weekly audit job only) and a stale doc table; proposed fix (drop hmmlearn from [paper], update wiki/Installation.md extras table) is correct and low-risk.

### ✅ [LOW] Section [Unreleased] vide malgré 10 commits depuis v0.13.1 ; 16 versions dans le CHANGELOG mais seulement 3 tags

`CHANGELOG.md:8 (et tags git)` — statut : confirmed (1 vote(s)) — catégorie : Versionnement

Les numéros eux-mêmes sont cohérents (pyproject 0.13.1 = CITATION.cff 0.13.1 du 2026-05-06 = README « Software v0.13.1 » = wiki/Citing.md = tag v0.13.1). En revanche : (1) HEAD est 10 commits après v0.13.1 avec des changements visibles utilisateur (correctifs GUI 2ed780a/0d26177, perfs filtre a1b8487, reformatage) et la section [Unreleased] du CHANGELOG est vide ; (2) le CHANGELOG documente 16 versions (0.1.0 → 0.13.1) mais seuls v0.10.1, v0.11.0 et v0.13.1 sont tagués — README:27 dit « Since v0.13.0 » alors qu'aucun tag v0.13.0 n'existe.

**Preuve :** git describe --tags → v0.13.1-10-gaf19b48 ; sed CHANGELOG : « ## [Unreleased] » suivi directement de « ## [0.13.1] — 2026-05-06 » ; git tag -l → v0.10.1, v0.11.0, v0.13.1 ; grep "^## \[" CHANGELOG.md → 16 releases.

**Suggestion :** Documenter les correctifs post-0.13.1 sous [Unreleased] (ou tagger un v0.13.2), et soit tagger rétroactivement les versions du CHANGELOG, soit accepter la politique « tags jalons » et la noter dans le CHANGELOG.

**Ajustement de sévérité proposé par les vérificateurs :** low (inchangée) — problème d'hygiène de release/documentation sans impact sur le code ; le sous-point README v0.13.0 est cosmétique puisque 0.13.0 est une version CHANGELOG légitime.

### ✅ [LOW] Wiki Installation en décalage avec pyproject : Python « 3.13 likely works », venv créé avec python3 nu, flag pytest inexistant

`wiki/Installation.md:5-6, 17-18, 77` — statut : confirmed (1 vote(s)) — catégorie : Documentation

(1) Le wiki dit « Python 3.14+ (3.13 likely works) » alors que requires-python = ">=3.14" fait refuser l'installation par pip sous 3.13 — c'est précisément l'erreur que le README documente longuement (lignes 141-173). (2) Le quick-install du wiki utilise « python3 -m venv .venv » : si python3 système est 3.11 (cas Homebrew décrit par le README), l'installation échoue ; le README et le Makefile utilisent correctement python3.14. (3) La table « Common issues » conseille « pytest --rerun-failures 2 » : le plugin pytest-rerunfailures n'est pas dans [dev] et son flag réel est --reruns ; la commande échoue donc deux fois.

**Preuve :** wiki/Installation.md:5-6 « 3.13 likely works » vs pyproject.toml:14 « requires-python = ">=3.14" » ; wiki:17 « python3 -m venv .venv » vs README:113 « python3.14 -m venv .venv » ; wiki:77 « rerun with pytest --rerun-failures 2 » vs pyproject [dev] sans pytest-rerunfailures.

**Suggestion :** Aligner le wiki sur le README (python3.14 explicite, exigence stricte 3.14), corriger ou supprimer le conseil --rerun-failures (ajouter pytest-rerunfailures à [dev] si la flakiness Apple Silicon est réelle).

**Ajustement de sévérité proposé par les vérificateurs :** none — low is correctly calibrated (documentation drift, but the quick-install path genuinely fails on the platform the README itself describes)

### ✅ [INFO] actions/checkout@v4 et setup-python@v5 sur Node 20 déprécié — bascule forcée Node 24 le 16 juin 2026 (dans 5 jours)

`.github/workflows/build.yml:16-21 (idem lint.yml, tests.yml, audit.yml)` — statut : confirmed (1 vote(s)) — catégorie : CI

Tous les workflows utilisent checkout@v4 et setup-python@v5. Les runners GitHub affichent l'avertissement de dépréciation Node 20 et forceront Node 24 par défaut à partir du 16/06/2026 (suppression le 16/09/2026). dependabot github-actions est configuré mais en cadence mensuelle, donc la mise à jour peut arriver après l'échéance.

**Preuve :** Log du run 27133143048 : « ##[warning]Node.js 20 actions are deprecated... actions/checkout@v4, actions/setup-python@v5 ... forced to run with Node.js 24 by default starting June 16th, 2026 » ; .github/dependabot.yml : github-actions « interval: monthly ».

**Suggestion :** Bumper checkout et setup-python vers leurs majeures actuelles dans les 4 workflows (un seul commit), ou passer la cadence dependabot github-actions en weekly.

**Ajustement de sévérité proposé par les vérificateurs :** Keep [info]. The severity is correctly calibrated: the forced Node 24 default on 2026-06-16 is unlikely to break checkout@v4/setup-python@v5 immediately (they run under Node 24, with a documented opt-out), and hard removal of Node 20 is only on 2026-09-16. Do not escalate; the one-commit bump (checkout@v5, setup-python@v6 in all four workflows) is still worth doing soon to silence warnings and stay ahead of the September removal.

### ✅ [INFO] Affirmation fausse : enso_sst.csv « regenerated by the test harness if removed » — aucun test n'y touche

`README.md:652-654` — statut : confirmed (1 vote(s)) — catégorie : Documentation

Le README affirme que data/real/enso_sst.csv est régénéré par le harnais de tests s'il est supprimé. Aucun fichier de tests/ ne référence « enso » ; seul scripts/build_enso_csv.py (exécution manuelle) reconstruit le CSV, et prg/experiments/run_real_data.py se contente de le lire. Les données réelles nécessaires sont par ailleurs bien présentes et suivies (data/real/{enso_sst.csv, nino12, nino34, oni, sp500_vix.csv}).

**Preuve :** grep -rn enso tests/ → vide ; README.md:652-654 « the unified CSV data/real/enso_sst.csv is regenerated by the test harness if removed » ; git ls-files data/real/ → les 5 fichiers suivis.

**Suggestion :** Reformuler : « régénérable via python scripts/build_enso_csv.py », ou ajouter réellement une fixture de régénération.

**Ajustement de sévérité proposé par les vérificateurs :** None — [info] is correctly calibrated: pure documentation inaccuracy with no runtime impact since the data files are present and git-tracked.

### ✅ [INFO] Aucun artefact suivi à tort : egg-info, caches, paper/, docs/, .DS_Store, code-workspace tous correctement ignorés et non trackés

`.gitignore:global` — statut : confirmed (1 vote(s)) — catégorie : Hygiène git (constat positif)

Contrôle croisé sain : « git ls-files -i -c --exclude-standard » ne retourne rien (aucun fichier suivi ne viole les règles d'ignore). exactIMM.egg-info/ (règle *.egg-info/), .pytest_cache/, .ruff_cache/, .DS_Store, exactIMM.code-workspace, docs/ et paper/ sont présents sur le disque mais non suivis. Les artefacts résiduels locaux (data/simulated/plotsSimuWojciech.pdf, data/experiments/*.log, CSVs horodatés GUI) sont tous couverts par les règles. Les résultats suivis dans results/enso/ (7 fichiers JSON/tex, petits) sont un choix raisonnable et documenté dans le .gitignore lui-même ; data/experiments/.gitkeep, logs/.gitkeep et results/.gitkeep préservent l'arborescence attendue par les scripts.

**Preuve :** git ls-files -i -c --exclude-standard → vide ; git check-ignore -v exactIMM.egg-info → « .gitignore:75:*.egg-info/ » ; git status --short → seul « ?? audit/ » (notes d'audit de session).

**Suggestion :** Rien à faire — point de référence pour le rapport. Seule trace de travail non suivie : le dossier audit/ de la session en cours.

**Ajustement de sévérité proposé par les vérificateurs :** None — [info] is correct for a hygiene-baseline finding with no action required.

---
_Généré automatiquement ; justifications complètes des votes dans `raw/02-code-periphery-result.json`._