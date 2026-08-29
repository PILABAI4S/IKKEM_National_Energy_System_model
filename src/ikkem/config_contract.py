"""Configuration-provider interface for the model package.

Set ``IKKEM_CONFIG_PROVIDER`` to an importable module that implements every
function listed in ``REQUIRED_FUNCTIONS`` before running a model entry point.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Any


CONFIG_PROVIDER_ENV = "IKKEM_CONFIG_PROVIDER"
REQUIRED_FUNCTIONS = (
    "get_config",
    "get_load_demand_path",
    "get_param_yaml_path",
    "get_power_system_data_dir",
    "load_lcoe",
    "get_trans_data",
    "get_pro_underground",
)

_provider_module: ModuleType | None = None


class ConfigurationProviderError(RuntimeError):
    """Raised when the configuration provider is unavailable."""


def _provider() -> ModuleType:
    global _provider_module
    if _provider_module is not None:
        return _provider_module

    module_name = os.environ.get(CONFIG_PROVIDER_ENV, "").strip()
    if not module_name:
        raise ConfigurationProviderError(
            f"Set {CONFIG_PROVIDER_ENV} to an importable provider module."
        )

    module = importlib.import_module(module_name)
    missing = [name for name in REQUIRED_FUNCTIONS if not callable(getattr(module, name, None))]
    if missing:
        raise ConfigurationProviderError(
            f"Configuration provider {module_name!r} is missing callable(s): {missing}"
        )
    _provider_module = module
    return module


def _call(name: str, *args: Any, **kwargs: Any) -> Any:
    return getattr(_provider(), name)(*args, **kwargs)


def get_config() -> Any:
    return _call("get_config")


def get_load_demand_path() -> str:
    return _call("get_load_demand_path")


def get_param_yaml_path() -> str:
    return _call("get_param_yaml_path")


def get_power_system_data_dir() -> str:
    return _call("get_power_system_data_dir")


def load_lcoe() -> Any:
    return _call("load_lcoe")


def get_trans_data() -> Any:
    return _call("get_trans_data")


def get_pro_underground(params: Any) -> Any:
    return _call("get_pro_underground", params)
