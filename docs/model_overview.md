# Model-source overview

The package implements five model scenarios across three source modules:

- A0 — No AI: `ikkem.model.no_ai`;
- A1 — Fixed: the internal S0 branch of `ikkem.model.planning_runtime`;
- A2 — Planning-only: the internal S1 branch of `ikkem.model.planning_runtime`;
- A3 — Runtime-only: `ikkem.model.runtime_only`;
- A4 — Joint: the internal S4 branch of `ikkem.model.planning_runtime`.

All three modules use the same generic external AI-workload interface and transmission-neighbour helpers. They require Gurobi for optimization and a configuration provider for input loading.

The source modules retain their inherited internal scenario identifiers for implementation compatibility. Public documentation and scientific interpretation should use A0–A4 as model-scenario labels. Contrasts between them should be described as matched scenario comparisons, not as empirical causal effects.
