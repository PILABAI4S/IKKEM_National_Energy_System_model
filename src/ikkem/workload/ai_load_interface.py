"""External AI/data-centre load interface for the power-system model.

The main model keeps the legacy ``ai_ratio`` path as a fallback. When
``USE_EXTERNAL_AI_LOAD=true`` is set, this helper reads a province-level AI
load table and returns fixed AI demand, a flexible AI pool, and hosting
capacity limits in units compatible with the existing model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


PROVINCE_NAME_TO_CODE = {
    "Anhui": "AH",
    "Beijing": "BJ",
    "Fujian": "FJ",
    "Gansu": "GS",
    "Guangdong": "GD",
    "Guangxi": "GX",
    "Guizhou": "GZ",
    "Hainan": "HI",
    "Hebei": "HE",
    "Henan": "HA",
    "Heilongjiang": "HL",
    "Hubei": "HB",
    "Hunan": "HN",
    "Jilin": "JL",
    "Jiangxi": "JX",
    "Jiangsu": "JS",
    "Liaoning": "LN",
    "Inner Mongolia": "NM",
    "Ningxia": "NX",
    "Qinghai": "QH",
    "Sichuan": "SC",
    "Shaanxi": "SN",
    "Shandong": "SD",
    "Shanghai": "SH",
    "Shanxi": "SX",
    "Tianjin": "TJ",
    "Xinjiang": "XJ",
    "Tibet": "XZ",
    "Yunnan": "YN",
    "Zhejiang": "ZJ",
    "Chongqing": "CQ",
}

AI_LOAD_COLUMNS = {"fixed_ai_load_mw", "flexible_ai_load_mw"}
PROVINCE_COLUMNS = {"province", "province_code"}
FILTER_COLUMNS = {"scenario", "allocation_version", "year"}
HOUR_COLUMNS = ("hour_index", "hour", "h", "time_index")
PROVINCE_METADATA_COLUMNS = {
    "ai_hosting_capacity_mw",
    "ai_host_power_cap_gw",
    "ai_host_energy_cap_gwh_year",
    "ai_hosting_class",
    "host_cap_method",
    "ai_existing_locked_gw",
    "ai_committed_gw",
    "ai_new_gw",
    "ai_new_plannable_gw",
    "ai_runtime_flexible_gw",
    "new_plannable_share_by_year",
    "runtime_flexible_share_by_year",
    "ai_hosting_penalty_rmb_per_mwh",
    "host_penalty_rmb_per_mwh",
    "ai_local_retention_min",
    "local_retention_min",
    "ai_max_host_share",
    "max_host_share",
    "ai_hosting_upper_bound_mw",
    "scenario_northwest_max_host_share",
    "scenario_southwest_max_host_share",
    "scenario_coastal_demand_max_host_share",
}


@dataclass
class ExternalAILoad:
    fixed_ai_load_mw: dict[str, np.ndarray]
    flexible_ai_load_mw: dict[str, np.ndarray]
    flexible_pool_gw: float
    ai_hosting_capacity_gw: dict[str, float]
    ai_hosting_penalty_rmb_per_mwh: dict[str, float]
    ai_local_retention_min: dict[str, float]
    ai_max_host_share: dict[str, float]
    ai_hosting_upper_bound_gw: dict[str, float]
    metadata: dict[str, object]


@dataclass
class SourceClusterInterface:
    province_load: ExternalAILoad
    source_cluster_arrival_mw: dict[str, np.ndarray]
    source_cluster_profile: dict[str, np.ndarray]
    source_cluster_mean_mw: dict[str, float]
    source_cluster_zone: dict[str, str]
    source_cluster_type: dict[str, str]
    origin_reconstruction_weight: dict[tuple[str, str], float]
    destination_host_power_cap_gw: dict[str, float]
    destination_host_energy_cap_gwh_year: dict[str, float]
    metadata: dict[str, object]
    # ---- Phase 2a dual-pool (siting / runtime) fields (§5.3a) ----
    # When the data layer provides explicit siting/runtime columns these hold
    # the two pools. When those columns are absent (legacy single-pool
    # interface), the loader sets runtime := siting := the legacy arrival, so
    # downstream code that reads these fields degrades to old behaviour.
    source_cluster_siting_mean_mw: dict[str, float] = field(default_factory=dict)
    source_cluster_runtime_mean_mw: dict[str, float] = field(default_factory=dict)
    source_cluster_siting_arrival_mw: dict[str, np.ndarray] = field(default_factory=dict)
    source_cluster_runtime_arrival_mw: dict[str, np.ndarray] = field(default_factory=dict)
    source_cluster_runtime_profile: dict[str, np.ndarray] = field(default_factory=dict)
    siting_pool_gw: float = 0.0
    runtime_pool_gw: float = 0.0
    dual_pool: bool = False


def str_to_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _province_code(value: object) -> str:
    text = str(value).strip()
    return PROVINCE_NAME_TO_CODE.get(text, text)


def _ensure_province_code(df: pd.DataFrame) -> pd.DataFrame:
    if "province_code" in df.columns:
        df["province_code"] = df["province_code"].map(_province_code)
    elif "province" in df.columns:
        df["province_code"] = df["province"].map(_province_code)
    else:
        raise ValueError("AI load interface file missing province/province_code column.")
    return df


def _read_csv_selected(path: Path, wanted: set[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    available = [col for col in header.columns if col in wanted]
    return pd.read_csv(path, usecols=available)


def _prepare_operational_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = _ensure_province_code(out)
    if "ai_hosting_capacity_mw" not in out.columns and "ai_host_power_cap_gw" in out.columns:
        out["ai_hosting_capacity_mw"] = (
            pd.to_numeric(out["ai_host_power_cap_gw"], errors="coerce").fillna(0.0)
            * 1000.0
        )
    if (
        "ai_hosting_penalty_rmb_per_mwh" not in out.columns
        and "host_penalty_rmb_per_mwh" in out.columns
    ):
        out["ai_hosting_penalty_rmb_per_mwh"] = out["host_penalty_rmb_per_mwh"]
    if "ai_local_retention_min" not in out.columns and "local_retention_min" in out.columns:
        out["ai_local_retention_min"] = out["local_retention_min"]
    if "ai_max_host_share" not in out.columns and "max_host_share" in out.columns:
        out["ai_max_host_share"] = out["max_host_share"]
    defaults = {
        "ai_hosting_capacity_mw": 0.0,
        "ai_hosting_penalty_rmb_per_mwh": 0.0,
        "ai_local_retention_min": np.nan,
        "ai_max_host_share": np.nan,
        "ai_hosting_upper_bound_mw": np.nan,
    }
    for column, value in defaults.items():
        if column not in out.columns:
            out[column] = value
    return out


def _first_optional(row: pd.Series, columns: tuple[str, ...], default=np.nan):
    for column in columns:
        if column in row and pd.notna(row[column]):
            return row[column]
    return default


def _first_optional_numeric(row: pd.Series, columns: tuple[str, ...], default=np.nan):
    value = _first_optional(row, columns, default=default)
    if pd.isna(value):
        return default
    return float(value)


def _sum_numeric_dict(values: dict[str, float | None]) -> float:
    return float(sum(float(v) for v in values.values() if v is not None and pd.notna(v)))


def _filter_table(
    df: pd.DataFrame,
    scenario: str | None,
    allocation_version: str | None,
    year: int | None,
) -> pd.DataFrame:
    out = df.copy()
    if scenario and "scenario" in out.columns:
        out = out[out["scenario"].astype(str) == scenario]
    if allocation_version and "allocation_version" in out.columns:
        out = out[out["allocation_version"].astype(str) == allocation_version]
    if year is not None and "year" in out.columns:
        year_rows = out[out["year"].astype(int) == int(year)]
        if not year_rows.empty:
            out = year_rows
    return out


def _constant_hourly(value_mw: float, hours: int) -> np.ndarray:
    return np.full(hours, float(value_mw), dtype=float)


def _fit_hourly_series(values: np.ndarray, hours: int, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) == hours:
        return arr
    if len(arr) < hours and hours % len(arr) == 0:
        return np.tile(arr, hours // len(arr))
    if len(arr) > hours:
        return arr[:hours]
    raise ValueError(
        f"Hourly AI load series {label} has length {len(arr)}, "
        f"which cannot be fitted to requested hours={hours}."
    )


def load_external_ai_load(
    file_path: str | Path,
    provinces: list[str],
    hours: int,
    scenario: str | None = None,
    allocation_version: str | None = None,
    year: int | None = None,
    default_capacity_multiplier: float = 2.0,
    metadata_file_path: str | Path | None = None,
) -> ExternalAILoad:
    """Load stage-1 external AI inputs.

    Expected minimum columns:
    ``province`` or ``province_code``, plus ``fixed_ai_load_mw`` and
    ``flexible_ai_load_mw``.

    If an ``hour_index``/``hour``/``h`` column is present, fixed and flexible
    load columns are treated as hourly long-format series. Otherwise they are
    treated as province-level mean MW constants and expanded to all hours.

    Optional columns, either in ``file_path`` or ``metadata_file_path``:
    ``ai_hosting_capacity_mw, ai_hosting_penalty_rmb_per_mwh,
    ai_local_retention_min, ai_max_host_share, ai_hosting_upper_bound_mw,
    scenario, allocation_version, year, evidence_level``.
    The reader intentionally loads only operational columns needed by the
    model, so large hourly CSVs can keep rich audit metadata out-of-band.
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"AI load interface file not found: {path}")
    metadata_path = Path(metadata_file_path) if metadata_file_path else None
    if metadata_path is not None and not metadata_path.exists():
        raise FileNotFoundError(f"AI load metadata file not found: {metadata_path}")

    header = pd.read_csv(path, nrows=0)
    columns = set(header.columns)
    missing_load = sorted(AI_LOAD_COLUMNS - columns)
    if missing_load:
        raise ValueError(f"AI load interface file missing columns: {missing_load}")
    if not (PROVINCE_COLUMNS & columns):
        raise ValueError("AI load interface file missing province/province_code column.")

    load_columns = AI_LOAD_COLUMNS | PROVINCE_COLUMNS | FILTER_COLUMNS
    load_columns |= {col for col in HOUR_COLUMNS if col in columns}
    if metadata_path is None:
        load_columns |= PROVINCE_METADATA_COLUMNS
    df = _read_csv_selected(path, load_columns)

    df = _filter_table(df, scenario, allocation_version, year)
    if df.empty:
        raise ValueError(
            "AI load interface has no rows after filtering by "
            f"scenario={scenario!r}, allocation_version={allocation_version!r}, year={year!r}"
        )

    df = _ensure_province_code(df.copy())

    if metadata_path is not None:
        metadata_header = pd.read_csv(metadata_path, nrows=0)
        metadata_columns = set(metadata_header.columns)
        if not (PROVINCE_COLUMNS & metadata_columns):
            raise ValueError("AI load metadata file missing province/province_code column.")
        meta_df = _read_csv_selected(
            metadata_path,
            PROVINCE_COLUMNS | FILTER_COLUMNS | PROVINCE_METADATA_COLUMNS,
        )
        meta_df = _filter_table(meta_df, scenario, allocation_version, year)
        if meta_df.empty:
            raise ValueError(
                "AI load metadata file has no rows after filtering by "
                f"scenario={scenario!r}, allocation_version={allocation_version!r}, year={year!r}"
            )
    else:
        meta_df = df
    meta_df = _prepare_operational_metadata(meta_df)

    region_max_host_share = {}
    region_columns = {
        "northwest": "scenario_northwest_max_host_share",
        "southwest": "scenario_southwest_max_host_share",
        "coastal_demand": "scenario_coastal_demand_max_host_share",
    }
    for region, column in region_columns.items():
        if column not in meta_df.columns:
            continue
        values = pd.to_numeric(meta_df[column], errors="coerce").dropna().unique()
        if len(values):
            region_max_host_share[region] = float(values[0])
    hour_column = None
    for candidate in ("hour_index", "hour", "h", "time_index"):
        if candidate in df.columns:
            hour_column = candidate
            break

    if hour_column is None:
        province_df = df.groupby("province_code", as_index=False).agg(
            fixed_ai_load_mw=("fixed_ai_load_mw", "sum"),
            flexible_ai_load_mw=("flexible_ai_load_mw", "sum"),
        )
    else:
        df[hour_column] = pd.to_numeric(df[hour_column], errors="raise").astype(int)
        hourly_df = (
            df.groupby(["province_code", hour_column], as_index=False)
            .agg(
                fixed_ai_load_mw=("fixed_ai_load_mw", "sum"),
                flexible_ai_load_mw=("flexible_ai_load_mw", "sum"),
            )
            .sort_values(["province_code", hour_column])
        )
        province_df = hourly_df.groupby("province_code", as_index=False).agg(
            fixed_ai_load_mw=("fixed_ai_load_mw", "mean"),
            flexible_ai_load_mw=("flexible_ai_load_mw", "mean"),
        )

    metadata_df = meta_df.groupby("province_code", as_index=False).agg(
        ai_hosting_capacity_mw=("ai_hosting_capacity_mw", "first"),
        ai_hosting_penalty_rmb_per_mwh=("ai_hosting_penalty_rmb_per_mwh", "mean"),
        ai_local_retention_min=("ai_local_retention_min", "mean"),
        ai_max_host_share=("ai_max_host_share", "mean"),
        ai_hosting_upper_bound_mw=("ai_hosting_upper_bound_mw", "first"),
    )
    province_df = province_df.merge(metadata_df, on="province_code", how="left")
    province_df = _prepare_operational_metadata(province_df)
    hourly_by_province = {}
    if hour_column is not None:
        hourly_by_province = {
            pro: group
            for pro, group in hourly_df.groupby("province_code", sort=False)
        }

    fixed = {pro: np.zeros(hours, dtype=float) for pro in provinces}
    flexible = {pro: np.zeros(hours, dtype=float) for pro in provinces}
    capacity = {pro: 0.0 for pro in provinces}
    penalty = {pro: 0.0 for pro in provinces}
    local_retention = {}
    max_host_share = {}
    hosting_upper_bound = {}
    ai_existing_locked_gw = {}
    ai_committed_gw = {}
    ai_new_gw = {}
    ai_new_plannable_gw = {}
    ai_runtime_flexible_gw = {}
    ai_hosting_class = {}
    host_cap_method = {}
    new_plannable_share_by_year = {}
    runtime_flexible_share_by_year = {}

    seen = set()
    for _, row in province_df.iterrows():
        pro = str(row["province_code"])
        if pro not in fixed:
            continue
        if hour_column is None:
            fixed_mw = float(row["fixed_ai_load_mw"])
            flex_mw = float(row["flexible_ai_load_mw"])
            fixed_series = _constant_hourly(fixed_mw, hours)
            flexible_series = _constant_hourly(flex_mw, hours)
        else:
            pro_hourly = hourly_by_province.get(pro)
            if pro_hourly is None or pro_hourly.empty:
                continue
            fixed_mw = float(row["fixed_ai_load_mw"])
            flex_mw = float(row["flexible_ai_load_mw"])
            fixed_series = _fit_hourly_series(
                pro_hourly["fixed_ai_load_mw"].to_numpy(dtype=float),
                hours,
                f"{pro}.fixed_ai_load_mw",
            )
            flexible_series = _fit_hourly_series(
                pro_hourly["flexible_ai_load_mw"].to_numpy(dtype=float),
                hours,
                f"{pro}.flexible_ai_load_mw",
            )
        cap_mw = float(row["ai_hosting_capacity_mw"])
        if cap_mw <= 0:
            cap_mw = default_capacity_multiplier * (fixed_mw + flex_mw)
        fixed[pro] = fixed_series
        flexible[pro] = flexible_series
        capacity[pro] = cap_mw / 1000.0
        penalty[pro] = float(row["ai_hosting_penalty_rmb_per_mwh"])
        local_value = row.get("ai_local_retention_min", np.nan)
        share_value = row.get("ai_max_host_share", np.nan)
        upper_value = row.get("ai_hosting_upper_bound_mw", np.nan)
        if pd.notna(local_value):
            local_retention[pro] = float(local_value)
        if pd.notna(share_value):
            max_host_share[pro] = float(share_value)
        if pd.notna(upper_value) and float(upper_value) > 0:
            hosting_upper_bound[pro] = float(upper_value) / 1000.0
        existing_locked = _first_optional_numeric(row, ("ai_existing_locked_gw",))
        committed = _first_optional_numeric(row, ("ai_committed_gw",))
        new_capacity = _first_optional_numeric(row, ("ai_new_gw",))
        new_plannable = _first_optional_numeric(row, ("ai_new_plannable_gw",))
        runtime_flexible = _first_optional_numeric(row, ("ai_runtime_flexible_gw",))
        if pd.notna(existing_locked):
            ai_existing_locked_gw[pro] = float(existing_locked)
        if pd.notna(committed):
            ai_committed_gw[pro] = float(committed)
        if pd.notna(new_capacity):
            ai_new_gw[pro] = float(new_capacity)
        if pd.notna(new_plannable):
            ai_new_plannable_gw[pro] = float(new_plannable)
        if pd.notna(runtime_flexible):
            ai_runtime_flexible_gw[pro] = float(runtime_flexible)
        hosting_class = _first_optional(row, ("ai_hosting_class",))
        if pd.notna(hosting_class):
            ai_hosting_class[pro] = str(hosting_class)
        method = _first_optional(row, ("host_cap_method",))
        if pd.notna(method):
            host_cap_method[pro] = str(method)
        plannable_share = _first_optional_numeric(
            row, ("new_plannable_share_by_year",)
        )
        runtime_share = _first_optional_numeric(
            row, ("runtime_flexible_share_by_year",)
        )
        if pd.notna(plannable_share):
            new_plannable_share_by_year[pro] = float(plannable_share)
        if pd.notna(runtime_share):
            runtime_flexible_share_by_year[pro] = float(runtime_share)
        seen.add(pro)

    missing_provinces = [pro for pro in provinces if pro not in seen]
    flexible_pool_gw = sum(float(np.mean(arr)) for arr in flexible.values()) / 1000.0
    ai_load_decomposition = {
        "unit": "GW average",
        "existing_locked": ai_existing_locked_gw,
        "committed": ai_committed_gw,
        "new": ai_new_gw,
        "new_plannable": ai_new_plannable_gw,
        "runtime_flexible": ai_runtime_flexible_gw,
        "new_plannable_share_by_year": new_plannable_share_by_year,
        "runtime_flexible_share_by_year": runtime_flexible_share_by_year,
        "totals": {
            "existing_locked_gw": _sum_numeric_dict(ai_existing_locked_gw),
            "committed_gw": _sum_numeric_dict(ai_committed_gw),
            "new_gw": _sum_numeric_dict(ai_new_gw),
            "new_plannable_gw": _sum_numeric_dict(ai_new_plannable_gw),
            "runtime_flexible_gw": _sum_numeric_dict(ai_runtime_flexible_gw),
            "fixed_ai_load_gw": sum(float(np.mean(arr)) for arr in fixed.values())
            / 1000.0,
            "flexible_ai_load_gw": flexible_pool_gw,
        },
        "note": (
            "Optional interface decomposition. Empty dicts mean the selected "
            "input file did not provide explicit existing/committed/new/runtime "
            "AI load columns."
        ),
    }
    metadata = {
        "file_path": str(path),
        "metadata_file_path": str(metadata_path) if metadata_path else None,
        "scenario": scenario,
        "allocation_version": allocation_version,
        "year": year,
        "hours": hours,
        "load_rows_used": int(len(df)),
        "province_rows_used": int(len(province_df)),
        "hourly_load_format": hour_column is not None,
        "hour_column": hour_column,
        "load_columns_read": sorted(df.columns),
        "metadata_columns_read": sorted(meta_df.columns),
        "matched_provinces": sorted(seen),
        "missing_provinces": missing_provinces,
        "flexible_pool_gw": flexible_pool_gw,
        "province_local_retention_override_count": len(local_retention),
        "province_max_host_share_override_count": len(max_host_share),
        "province_hosting_upper_bound_override_count": len(hosting_upper_bound),
        "region_max_host_share_from_interface": region_max_host_share,
        "ai_load_decomposition": ai_load_decomposition,
        "destination_hosting_class_from_interface": ai_hosting_class,
        "host_cap_method_by_province": host_cap_method,
        "host_cap_method": (
            sorted(set(host_cap_method.values()))[0]
            if len(set(host_cap_method.values())) == 1 and host_cap_method
            else "unspecified"
        ),
        "note": "Stage-1 interface uses average hourly MW. Optional per-province hosting constraints separate demand-source retention from hosting-destination feasibility.",
    }
    return ExternalAILoad(
        fixed,
        flexible,
        flexible_pool_gw,
        capacity,
        penalty,
        local_retention,
        max_host_share,
        hosting_upper_bound,
        metadata,
    )


def load_source_cluster_interface(
    interface_dir: str | Path,
    provinces: list[str],
    hours: int,
    scenario: str | None = None,
    allocation_version: str | None = None,
    year: int | None = None,
    default_capacity_multiplier: float = 2.0,
) -> SourceClusterInterface:
    """Load the S4-OD data-layer interface directory.

    The directory is expected to contain:
    ``province_ai_load_hourly.csv``, ``source_cluster_arrival_hourly.csv``,
    ``source_cluster_members.csv``, ``destination_capacity.csv``, and
    optionally ``metadata.json``.
    """

    root = Path(interface_dir)
    province_file = root / "province_ai_load_hourly.csv"
    cluster_file = root / "source_cluster_arrival_hourly.csv"
    members_file = root / "source_cluster_members.csv"
    capacity_file = root / "destination_capacity.csv"
    metadata_json = root / "metadata.json"
    for path in [province_file, cluster_file, members_file, capacity_file]:
        if not path.exists():
            raise FileNotFoundError(f"Missing source-cluster interface file: {path}")

    province_load = load_external_ai_load(
        province_file,
        provinces,
        hours,
        scenario=scenario,
        allocation_version=allocation_version,
        year=year,
        default_capacity_multiplier=default_capacity_multiplier,
        metadata_file_path=capacity_file,
    )

    cluster_df = pd.read_csv(cluster_file)
    required_cluster_cols = {
        "cluster_id",
        "zone",
        "hour_index",
        "cluster_arrival_mw",
        "cluster_profile",
    }
    missing = sorted(required_cluster_cols - set(cluster_df.columns))
    if missing:
        raise ValueError(f"source_cluster_arrival_hourly.csv missing columns: {missing}")
    cluster_df = _filter_table(cluster_df, scenario, allocation_version, year)
    cluster_df["hour_index"] = pd.to_numeric(cluster_df["hour_index"], errors="raise").astype(int)

    # §5.3b dual-pool detection. New data layer emits explicit siting/runtime
    # arrival columns; legacy single-pool files do not. When absent we set
    # runtime := siting := the legacy cluster_arrival_mw so all downstream
    # consumers degrade to old single-pool behaviour with zero regression.
    has_siting_col = "cluster_siting_arrival_mw" in cluster_df.columns
    has_runtime_col = "cluster_runtime_arrival_mw" in cluster_df.columns
    dual_pool = bool(has_siting_col and has_runtime_col)

    arrival: dict[str, np.ndarray] = {}
    profile: dict[str, np.ndarray] = {}
    mean_mw: dict[str, float] = {}
    cluster_zone: dict[str, str] = {}
    cluster_type: dict[str, str] = {}
    siting_arrival: dict[str, np.ndarray] = {}
    runtime_arrival: dict[str, np.ndarray] = {}
    runtime_profile: dict[str, np.ndarray] = {}
    siting_mean_mw: dict[str, float] = {}
    runtime_mean_mw: dict[str, float] = {}
    for gid, group in cluster_df.groupby("cluster_id", sort=False):
        group = group.sort_values("hour_index")
        gid = str(gid)
        arrival[gid] = _fit_hourly_series(
            group["cluster_arrival_mw"].to_numpy(dtype=float),
            hours,
            f"{gid}.cluster_arrival_mw",
        )
        profile[gid] = _fit_hourly_series(
            group["cluster_profile"].to_numpy(dtype=float),
            hours,
            f"{gid}.cluster_profile",
        )
        mean_mw[gid] = float(np.mean(arrival[gid]))
        cluster_zone[gid] = str(group["zone"].iloc[0])
        cluster_type[gid] = str(group.get("cluster_type", pd.Series([""])).iloc[0])

        # Siting pool: explicit column if present, else the legacy arrival.
        if has_siting_col:
            siting_arrival[gid] = _fit_hourly_series(
                group["cluster_siting_arrival_mw"].to_numpy(dtype=float),
                hours,
                f"{gid}.cluster_siting_arrival_mw",
            )
        else:
            siting_arrival[gid] = arrival[gid]
        # Runtime pool: explicit column if present, else degrade to siting.
        if has_runtime_col:
            runtime_arrival[gid] = _fit_hourly_series(
                group["cluster_runtime_arrival_mw"].to_numpy(dtype=float),
                hours,
                f"{gid}.cluster_runtime_arrival_mw",
            )
        else:
            runtime_arrival[gid] = siting_arrival[gid]
        if "cluster_runtime_profile" in group.columns:
            runtime_profile[gid] = _fit_hourly_series(
                group["cluster_runtime_profile"].to_numpy(dtype=float),
                hours,
                f"{gid}.cluster_runtime_profile",
            )
        else:
            rt_mean = float(np.mean(runtime_arrival[gid]))
            runtime_profile[gid] = (
                runtime_arrival[gid] / rt_mean
                if rt_mean > 1e-9
                else np.zeros_like(runtime_arrival[gid])
            )

        # Prefer explicit mean columns when supplied, else recompute from series.
        if "source_cluster_siting_mean_mw" in group.columns and pd.notna(
            group["source_cluster_siting_mean_mw"].iloc[0]
        ):
            siting_mean_mw[gid] = float(group["source_cluster_siting_mean_mw"].iloc[0])
        else:
            siting_mean_mw[gid] = float(np.mean(siting_arrival[gid]))
        if "source_cluster_runtime_mean_mw" in group.columns and pd.notna(
            group["source_cluster_runtime_mean_mw"].iloc[0]
        ):
            runtime_mean_mw[gid] = float(group["source_cluster_runtime_mean_mw"].iloc[0])
        else:
            runtime_mean_mw[gid] = float(np.mean(runtime_arrival[gid]))

    # §5.3b subset guard: runtime must not exceed siting at the cluster level.
    subset_violations = {
        gid: runtime_mean_mw[gid] - siting_mean_mw[gid]
        for gid in siting_mean_mw
        if runtime_mean_mw[gid] - siting_mean_mw[gid] > 1e-6
    }
    if subset_violations:
        raise ValueError(
            "Source-cluster interface violates runtime <= siting at clusters: "
            f"{subset_violations}. Check the data-layer dual-pool generation."
        )

    siting_pool_gw = sum(siting_mean_mw.values()) / 1000.0
    runtime_pool_gw = sum(runtime_mean_mw.values()) / 1000.0

    members_df = pd.read_csv(members_file)
    required_member_cols = {"cluster_id", "province_code", "origin_weight"}
    missing = sorted(required_member_cols - set(members_df.columns))
    if missing:
        raise ValueError(f"source_cluster_members.csv missing columns: {missing}")
    members_df = _filter_table(members_df, scenario, allocation_version, year)
    members_df = _ensure_province_code(members_df)
    origin_weight = {
        (str(row["province_code"]), str(row["cluster_id"])): float(row["origin_weight"])
        for _, row in members_df.iterrows()
    }

    capacity_df = pd.read_csv(capacity_file)
    capacity_df = _filter_table(capacity_df, scenario, allocation_version, year)
    capacity_df = _ensure_province_code(capacity_df)
    destination_power_cap = {}
    destination_energy_cap = {}
    for _, row in capacity_df.iterrows():
        pro = str(row["province_code"])
        if "ai_host_power_cap_gw" in row and pd.notna(row["ai_host_power_cap_gw"]):
            destination_power_cap[pro] = float(row["ai_host_power_cap_gw"])
        else:
            destination_power_cap[pro] = float(row.get("ai_hosting_capacity_mw", 0.0)) / 1000.0
        if "ai_host_energy_cap_gwh_year" in row and pd.notna(row["ai_host_energy_cap_gwh_year"]):
            destination_energy_cap[pro] = float(row["ai_host_energy_cap_gwh_year"])

    metadata = {
        "interface_dir": str(root),
        "province_file": str(province_file),
        "cluster_file": str(cluster_file),
        "members_file": str(members_file),
        "capacity_file": str(capacity_file),
        "metadata_file": str(metadata_json) if metadata_json.exists() else None,
        "cluster_count": len(arrival),
        "cluster_ids": sorted(arrival),
        "source_cluster_interface": True,
        "province_load_metadata": province_load.metadata,
        # §5.3c dual-pool audit, exposed for S0/S1/S4 reconciliation.
        "dual_pool": dual_pool,
        "dual_pool_columns_present": {
            "cluster_siting_arrival_mw": has_siting_col,
            "cluster_runtime_arrival_mw": has_runtime_col,
            "cluster_runtime_profile": "cluster_runtime_profile" in cluster_df.columns,
        },
        "siting_pool_gw": siting_pool_gw,
        "runtime_pool_gw": runtime_pool_gw,
        "runtime_over_siting_ratio": (
            runtime_pool_gw / siting_pool_gw if siting_pool_gw > 1e-9 else 0.0
        ),
        "dual_pool_degraded_to_single": (not dual_pool),
        "dual_pool_note": (
            "When dual_pool is False the data layer did not provide explicit "
            "siting/runtime columns, so runtime was set equal to siting "
            "(legacy single-pool behaviour)."
        ),
    }
    if metadata_json.exists():
        metadata["metadata_json"] = json.loads(metadata_json.read_text())

    return SourceClusterInterface(
        province_load=province_load,
        source_cluster_arrival_mw=arrival,
        source_cluster_profile=profile,
        source_cluster_mean_mw=mean_mw,
        source_cluster_zone=cluster_zone,
        source_cluster_type=cluster_type,
        origin_reconstruction_weight=origin_weight,
        destination_host_power_cap_gw=destination_power_cap,
        destination_host_energy_cap_gwh_year=destination_energy_cap,
        metadata=metadata,
        source_cluster_siting_mean_mw=siting_mean_mw,
        source_cluster_runtime_mean_mw=runtime_mean_mw,
        source_cluster_siting_arrival_mw=siting_arrival,
        source_cluster_runtime_arrival_mw=runtime_arrival,
        source_cluster_runtime_profile=runtime_profile,
        siting_pool_gw=siting_pool_gw,
        runtime_pool_gw=runtime_pool_gw,
        dual_pool=dual_pool,
    )
