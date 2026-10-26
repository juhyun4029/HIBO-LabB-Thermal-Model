# Curated result summaries

This directory contains compact publication outputs from completed run
`20261024_084010_3074c42a`. Raw chains, checkpoints, execution logs and full
candidate records are intentionally excluded. The root README reports every
accepted merge and its convergence and temperature metrics.

| File | Contents |
|---|---|
| `selected_path.csv` | All nine accepted merges, original component membership, model sizes, train/validation RMSE, convergence, candidate counts and Pareto distances |
| `initial_parameter_summary.csv` | All 59 reported initial-model marginals |
| `initial_convergence.json` | Initial physical and ordered diagnostics, representative-draw identity and temperature scores |
| `final_model.json` | Final 5R2C structure, coefficients, allocation fractions and fixed noise parameters |
| `final_parameter_summary.csv` | Ten local parameters from the authoritative final rescue attempt |
| `numeric_rules.csv` | All 30 reported numeric subgroup rules |
| `readable_rules.csv` | Readable conditions and their recorded support and effect summaries |
| `rules_summary.json` | Rule conditions, summary statistics, search settings and interpretation limits |
| `provenance.json` | Original archive hashes and source locations for the curated outputs |

The initial fit passed the prescribed R-hat and ESS criteria. Reduction adopted
nine merges, evaluating 165 distinct pairs with 160 PASS and five UNRESOLVED
outcomes. Final validation RMSE was 0.585278°C for 5R2C; 14R7C had the lowest
validation RMSE along the adopted path, 0.354658°C.

Reduced-model uncertainty is conditional on fixed parent parameters. Validation
was reused for acceptance and selection. Numeric subgroup rules are exploratory
and have not been independently validated. These summaries retain the recorded
values; no new fitting was performed when preparing them.
