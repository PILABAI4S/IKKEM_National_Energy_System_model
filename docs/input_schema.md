# Generic AI-workload input schema

`ikkem.workload.ai_load_interface.load_external_ai_load` accepts a CSV table with these minimum fields:

| Field | Meaning |
|---|---|
| `province` or `province_code` | Province identifier |
| `fixed_ai_load_mw` | Fixed AI demand in MW |
| `flexible_ai_load_mw` | Flexible AI demand in MW |

An optional hour field may be named `hour_index`, `hour`, `h`, or `time_index`. If no hour field exists, the loader treats each provincial value as a mean-MW constant and expands it to the requested number of hours.

Optional selectors are `scenario`, `allocation_version`, and `year`. Optional operational metadata accepted by the loader includes hosting capacity, hosting penalty, local-retention, maximum-host-share, and hosting-upper-bound fields. See the module docstrings for the exact accepted aliases.

The source-cluster loader expects a directory containing:

- `province_ai_load_hourly.csv`;
- `source_cluster_arrival_hourly.csv`;
- `source_cluster_members.csv`;
- `destination_capacity.csv`;
- optional `metadata.json`.
