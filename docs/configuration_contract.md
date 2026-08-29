# Configuration provider interface

The core modules delegate configuration loading to an external Python module. Set the environment variable `IKKEM_CONFIG_PROVIDER` to the dotted import name of that module.

The provider must implement these callables:

| Function | Expected role |
|---|---|
| `get_config()` | Return the argument/configuration object consumed by the selected model module |
| `get_load_demand_path()` | Return the load-demand file path |
| `get_param_yaml_path()` | Return the parameter-YAML path |
| `get_power_system_data_dir()` | Return the power-system input directory |
| `load_lcoe()` | Load and return the model's LCOE inputs |
| `get_trans_data()` | Load and return transmission data |
| `get_pro_underground(params)` | Return underground-resource information for the supplied parameters |

Model modules should be launched with package semantics so their relative imports resolve:

```bash
IKKEM_CONFIG_PROVIDER=your_package.provider \
python -m ikkem.model.planning_runtime
```
