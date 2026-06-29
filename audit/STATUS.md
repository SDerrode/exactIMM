# Audit complet exactIMM — état d'avancement

Démarré le 2026-06-10 sur le commit `af19b48` (main).
Demande : audit complet du code + du papier (`paper/`).
Contrainte : interruptions par limites d'utilisation → checkpoints après chaque vague.

## Plan et avancement

| Vague | Périmètre | Statut | Checkpoint |
|-------|-----------|--------|------------|
| 1 | Code cœur scientifique : `prg/filter/`, `prg/utils/h5_constraint.py`, `prg/classes/`, `prg/learning/` | FAIT (2026-06-11) — 48 trouvailles, dont 3 HIGH confirmées | `01-code-core.md` |
| 2 | Code périphérie : `prg/gui/`, `tests/`, `scripts/`, `prg/experiments/`, hygiène dépôt (CI, pyproject, Makefile) | FAIT (2026-06-11) — 89 trouvailles, 2 CRITICAL + 12 HIGH confirmées | `02-code-periphery.md` |
| 3 | Papier — maths : sections 02–05, annexes A–E, cohérence des notations | FAIT (2026-06-12) — 80 trouvailles, 4 CRITICAL + 18 HIGH confirmées | `03-paper-math.md` |
| 4 | Papier — cohérence papier↔code↔résultats + éditorial/biblio/LaTeX | FAIT (2026-06-13) — 4a+4b+4c, 72 trouvailles | `04a/04b/04c-*.md` |
| 5 | Synthèse finale | **FAIT (2026-06-13)** — 289 trouvailles, 268 confirmées (3 crit / 29 high / 72 med eff.) | `AUDIT_REPORT.md` |

> **AUDIT TERMINÉ.** Lire `AUDIT_REPORT.md` (synthèse). Toutes les vagues sont closes et persistées.

## Méthode
- Chaque vague = workflow multi-agents : finders par dimension → vérification adversariale
  de chaque trouvaille (2 vérificateurs pour critical/high/medium, 1 pour low/info).
- Statuts des trouvailles : confirmé / incertain / réfuté (les réfutés sont conservés en annexe).
- En cas de reprise après interruption : lire ce fichier, reprendre à la première vague non « FAIT ».

## Reprise des workflows (en cas d'interruption)
Relancer avec `Workflow({scriptPath, resumeFromRunId})` — les agents terminés reviennent du cache.
- Vague 1 : run `wf_a0c95e2c-68b` — TERMINÉ, résultat persisté dans `raw/01-code-core-result.json`.
- Vague 2 : run `wf_61178f24-945` — TERMINÉ, résultat persisté dans `raw/02-code-periphery-result.json`.
- Vague 3 : run `wf_1ee834ca-798` (relance à neuf, le 1er essai `wf_0ff443a3-2b5` n'avait rien de réutilisable),
  script `.../workflows/scripts/audit-paper-math-wf_0ff443a3-2b5.js`
  (sous /Users/MacBook_Derrode/.claude/projects/-Users-MacBook-Derrode-Documents-ProjetsRecherche-Markov-FofGss-exactIMM--claude-worktrees-adoring-turing-8b6f51/ae181375-195d-4aea-b218-bdd1feed984a/)
  ÉTAT 2026-06-12 : le monolithe a été abandonné après 3 reprises ; les résultats ont été EXTRAITS À LA MAIN
  des transcripts vers `raw/03-paper-math-extracted.json` (80 trouvailles / 6 dimensions + 64 verdicts).
  Les 66 vérifications manquantes sont traitées par 3 mini-lots (`raw/03-verify-batch{1,2,3}.json`, 22 items chacun)
  via le script `workflows/scripts/verify-batch-wf_c290b13a-8ba.js` (workflow `verify-batch`, prend
  args={batchFile, count} — patché pour parser les args stringifiés).
  - Lot 1 (filtrage + contrainte) : run `wf_c0228168-40f` EN COURS → fusionner les verdicts dans
    `raw/03-paper-math-extracted.json` (clé verdicts, format {title,lens,verdict}), puis lancer lot 2 puis lot 3.
  - Après les 3 lots : générer `03-paper-math.md` (statuts: confirmed/uncertain/refuted selon votes), MAJ ce STATUS, vague 4.
- Vague 4 — NOUVELLE MÉTHODE (demande utilisateur 2026-06-12) : PROGRESSIVE, 3 mini-workflows
  indépendants de ~30-40 agents chacun, persistés individuellement dès leur fin :
  - 4a `eq-vs-code` (algorithmes papier ↔ code) : finders FAITS (run `wf_89cd0f89-6ee`, 28 trouvailles →
    `raw/04a-findings.json`). Vérif via `verify-batch` (script `verify-batch-wf_c290b13a-8ba.js`,
    args={batchFile,count}) sur `raw/04a-verify-batch{1,2}.json` :
      · batch1 (21) FAIT → fusionné dans `raw/04a-extracted.json` (21 verdicts).
      · batch2 (21) FAIT (run `wf_b2d4b824-79b`). 4a COMPLET : 42 verdicts dans `raw/04a-extracted.json`,
        28 trouvailles toutes confirmées (2 high, 12 medium, 10 low, 4 info) → `04a-eq-vs-code.md`. FAIT.
    NB générateur markdown réutilisable : `raw/gen_md.py` (args: extracted.json out.md "Titre" "sous-titre" ;
    lit data['_dimtitles'] pour les libellés de dimensions). `verify-batch` script = `verify-batch-wf_c290b13a-8ba.js`.
  - 4b `experiments-vs-results` : FAIT. finders run `wf_397fc904-434` (18 trouvailles), vérif runs
    `wf_1ed49426-fda`+`wf_03c227fc-29b` → `raw/04b-extracted.json` (31 verdicts), `04b-experiments.md`.
    Bilan : 6 high (la plupart = artefacts orphelins hors-papier, rétrogradés), 6 medium, 1 low confirmées ;
    point saillant : label leakage ENSO (ONI centré, HIGH maintenu) ; le PDF compilé est numériquement correct.
  - 4c `editorial` : abstract/intro/conclusion vs corps, paper.bib, hygiène LaTeX, fichiers test*.pdf parasites
    → `raw/04c-findings.json` + `raw/04c-extracted.json` + `04c-editorial.md`. EN COURS (finders run `wf_<voir ci-dessous>`).
    Process identique à 4a/4b : finders → construire `raw/04c-verify-batch{1,2}.json` → `verify-batch` → fusion → gen_md → MAJ STATUS → vague 5.
  Lancer 4a, persister, puis 4b, persister, puis 4c, persister. JAMAIS les trois en parallèle.
- Vague 5 (synthèse) : inline depuis les fichiers de audit/, AUCUN agent (insensible aux limites).
  → `AUDIT_REPORT.md` : résumé exécutif, top des trouvailles par sévérité (code et papier séparés),
  tableau complet, méthodo, stats (agents/tokens), recommandations priorisées.

## Repères
- Dépôt principal : /Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM
- Code : prg/ (~16,7k LOC), tests/ (3,5k, 219 tests), scripts/ (2,9k)
- Papier : paper/sections/*.tex (9), paper/appendix/*.tex (5), ~2 370 lignes, PDF compilé du 7 mai
- Venv : .venv (réparé le 2026-06-09, shebangs OK)
