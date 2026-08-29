# IKKEM National Energy System Model

This repository contains the Python implementation of the IKKEM model and its generic data interfaces.

## Included

- core model implementations for the No-AI (A0), Fixed (A1), Planning-only (A2), Runtime-only (A3), and Joint (A4) model scenarios;
- transmission-network helper functions;
- generic external AI-workload readers and a generic workload transformation utility;
- a provider interface for configuration and data-loading logic;
- generic result-loading and numerical aggregation utilities;
- package metadata and documentation.

## Layout

```text
.
├── docs/                         Model and interface documentation
├── examples/                     Configuration-free interface example
├── src/ikkem/
│   ├── config_contract.py        Configuration-provider adapter
│   ├── model/                    Core optimization source
│   ├── postprocessing/           Generic result I/O utilities
│   └── workload/                 Generic AI-workload interfaces
└── pyproject.toml                Python package metadata
```

## Local installation

Python 3.10 or newer is required. Gurobi and a valid Gurobi licence are required to solve the optimization model.

```bash
python -m pip install -e .
```

The generic interfaces can be imported directly:

```python
from ikkem.workload.ai_load_interface import load_external_ai_load
```

Running a model requires a configuration provider named by `IKKEM_CONFIG_PROVIDER`. The provider interface is documented in [Configuration provider interface](docs/configuration_contract.md).

```bash
IKKEM_CONFIG_PROVIDER=your_package.provider \
python -m ikkem.model.planning_runtime
```

## Model-scenario source mapping

| Model scenario | Source module | Role |
|---|---|---|
| A0 — No AI | `ikkem.model.no_ai` | Dedicated No-AI counterfactual |
| A1 — Fixed | `ikkem.model.planning_runtime` | Fixed workload placement and execution |
| A2 — Planning-only | `ikkem.model.planning_runtime` | Planning allocation without Runtime shifting |
| A3 — Runtime-only | `ikkem.model.runtime_only` | Runtime shifting with the Fixed planning anchor |
| A4 — Joint | `ikkem.model.planning_runtime` | Joint Planning allocation and Runtime execution |

These labels describe the roles of the source modules.
