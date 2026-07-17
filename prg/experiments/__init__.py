"""
prg/experiments
===============
Monte-Carlo simulation study (§6 of the paper).

Sub-modules
-----------
models_paper     — reference models M1, M2, M3 with AB-constrained B
                   (get_params_M1 / get_params_M2 / get_params_M3 / get_params)
metrics          — scalar performance metrics:
                     dof_ab            free-parameter count d_{AB}(K,q,s)
                     compute_rmse      root mean squared filtering error
                     compute_nees      ANEES (expected 1 for consistent filter)
                     compute_ljung_box Ljung-Box p-value (innovation whiteness)
                     compute_jarque_bera Jarque-Bera p-value (normality)
                     compute_bic       BIC for AB-constrained GSS models
run_simulations  — §6.2: Monte-Carlo filter benchmark (NGH-MSM-KF vs GPB2):
                     run_one_trial     single (model, N, seed, mode) trial
                     run_all           full study → data/experiments/mc_results.csv
                   CLI: python -m prg.experiments.run_simulations [--help]
run_supervised   — §6.3: supervised OLS MC study (4 projections × 100 seeds):
                     run_supervised_trial   single trial → list of metric dicts
                     run_supervised_all     full study → supervised_results.csv
                   CLI: python -m prg.experiments.run_supervised [--help]
run_em           — §6.4: semi-supervised Baum-Welch EM MC study (PH vs GEM):
                     run_em_trial           single trial → (rows, history_rows)
                     run_em_all             full study → em_results.csv +
                                             em_ll_history.csv
                   CLI: python -m prg.experiments.run_em [--help]
make_figures     — post-processing: reads all *_results.csv files and writes
                     paper/figures/generated/fig_rmse_vs_N.pdf        (§6.2)
                     paper/figures/generated/fig_supervised_rmse.pdf   (§6.3)
                     paper/figures/generated/fig_em_convergence.pdf    (§6.4)
                     paper/figures/generated/tab_filter_M1.tex         (§6.2)
                     paper/figures/generated/tab_filter_M2M3.tex       (§6.2)
                     paper/figures/generated/tab_supervised_M1.tex     (§6.3)
                     paper/figures/generated/tab_em_basin.tex          (§6.4)
                     paper/figures/generated/tab_em_restarts.tex       (§6.4)
                   CLI: python -m prg.experiments.make_figures [--help]
fill_placeholders — final step: fills \\ph{...} placeholders in
                     paper/sections/06_experiments.tex with numerical values
                     computed from the *_results.csv files.
                   CLI: python -m prg.experiments.fill_placeholders [--help]
                         python -m prg.experiments.fill_placeholders --dry-run

Recommended workflow
--------------------
1. python -m prg.experiments.run_simulations   # §6.2  (~30 min)
2. python -m prg.experiments.run_supervised    # §6.3  (~15 min)
3. python -m prg.experiments.run_em            # §6.4  (~2 hours)
4. python -m prg.experiments.make_figures      # generate PDF figs + .tex tables
5. python -m prg.experiments.fill_placeholders # fill \\ph{} in 06_experiments.tex
"""
