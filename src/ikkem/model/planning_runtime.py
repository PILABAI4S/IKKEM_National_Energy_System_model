import argparse
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from numpy.ma.core import count
from omegaconf import OmegaConf
import pandas as pd
import pickle
from ikkem.config_contract import (
    get_config,
    get_load_demand_path,
    get_param_yaml_path,
    get_power_system_data_dir,
    load_lcoe,
    get_trans_data,
    get_pro_underground,
)
from .transmission_helpers import (
    get_ac_in_neighbors,
    get_ac_out_neighbors,
    get_dc_in_neighbors,
    get_dc_out_neighbors,
)
from ikkem.workload.ai_load_interface import (
    load_external_ai_load,
    load_source_cluster_interface,
    str_to_bool,
)
import copy

AI_HUB_PROVINCES = {"HE", "JS", "GD", "SC", "CQ", "NM", "GZ", "GS", "NX", "XJ", "YN"}
AI_NORTHWEST_PROVINCES = {"NM", "XJ", "GS", "QH", "NX"}
AI_SOUTHWEST_PROVINCES = {"SC", "CQ", "GZ", "YN"}
AI_COASTAL_DEMAND_PROVINCES = {"BJ", "SH", "TJ", "GD", "JS", "ZJ", "SD", "FJ"}
AI_OPERATIONAL_SCENARIOS = {"S0", "S1", "S4"}
H2_SCENARIOS = {"NONE", "AG", "UG"}
EXTREME_CF_SCENARIOS = {"none", "conservative", "normal", "aggressive"}
EXTREME_CF_EVENT_TYPES = {"wind", "both"}
EXTREME_CF_FACTOR_KINDS = {"mean"}

# Policy-facing hosting classes. Network tier remains a reachability filter;
# hosting class controls per-destination flexible-AI concentration caps.
AI_CLASS_A_WESTERN_NONREALTIME = {"GZ", "NM", "GS", "NX"}
AI_CLASS_B_NATIONAL_CLUSTER = {"HE", "AH", "GD", "SC", "CQ"}
AI_CLASS_C_EASTERN_DEMAND = {"BJ", "SH", "JS", "ZJ", "TJ"}
AI_CLASS_D_RESOURCE_CANDIDATE = {"XJ", "YN", "QH", "SD", "FJ", "SN"}
AI_CLASS_E_LOCAL_SERVICE = {
    "GX", "HI", "HA", "HB", "HN", "HL", "JL", "JX", "LN", "SX", "XZ"
}

AI_SCENARIO_DEST_CAPS = {
    "conservative": {"A": 0.15, "B": 0.10, "C": 0.05, "D": 0.03, "E": 0.02},
    "central": {"A": 0.25, "B": 0.15, "C": 0.08, "D": 0.05, "E": 0.03},
    "liberal": {"A": 0.35, "B": 0.25, "C": 0.12, "D": 0.10, "E": 0.05},
    "extreme_upper_bound": {"A": 0.45, "B": 0.35, "C": 0.15, "D": 0.15, "E": 0.05},
    "legacy_tier": {},
}
AI_SCENARIO_OD_CAPS = {
    "conservative": {"A": 0.05, "B": 0.03, "C": 0.02, "D": 0.01, "E": 0.005},
    "central": {"A": 0.08, "B": 0.05, "C": 0.03, "D": 0.015, "E": 0.01},
    "liberal": {"A": 0.12, "B": 0.08, "C": 0.05, "D": 0.03, "E": 0.02},
    "extreme_upper_bound": {"A": 0.20, "B": 0.15, "C": 0.08, "D": 0.05, "E": 0.03},
    "legacy_tier": {},
}


def set_small_values_to_zero(data_dict, threshold=1e-3):
    """将字典中所有NumPy数组内绝对值小于阈值的元素设为0"""
    for key in data_dict:
        # print(data_dict[key])
        arr = np.array(data_dict[key])
        # 创建绝对值小于阈值的掩码

        mask = np.abs(arr) < threshold
        # 使用掩码将符合条件的值设为0
        arr[mask] = 0
        # 可选：更新字典中的数组（实际已原地修改）
        data_dict[key] = arr.tolist()

    return data_dict


def CRF(discount_rate: float = 0.05, lifetime: int = 30):
    crf = (discount_rate * (1 + discount_rate) ** lifetime) / (
        (1 + discount_rate) ** lifetime - 1
    )
    return crf


def _get_arg(args, name, default):
    if OmegaConf.is_config(args):
        return OmegaConf.select(args, name, default=default)
    return getattr(args, name, default)


def classify_energy_gap(
    abs_gap_gwh,
    rel_gap,
    strict_abs_gwh,
    strict_rel,
    warn_abs_gwh,
    warn_rel,
    review_abs_gwh,
    review_rel,
    fail_abs_gwh,
    fail_rel,
):
    """Tiered energy-conservation audit with structural hard-fail thresholds."""
    if abs_gap_gwh <= strict_abs_gwh or rel_gap <= strict_rel:
        return "PASS_STRICT"
    if abs_gap_gwh <= warn_abs_gwh or rel_gap <= warn_rel:
        return "WARN_NUMERICAL_RESIDUAL"
    if abs_gap_gwh <= review_abs_gwh or rel_gap <= review_rel:
        return "WARN_REVIEW_REQUIRED"
    if abs_gap_gwh > fail_abs_gwh and rel_gap > fail_rel:
        return "FAIL_STRUCTURAL"
    return "WARN_REVIEW_REQUIRED"


def classify_temporal_gap(gap_gwh, strict_gwh, warn_gwh, review_gwh, fail_gwh):
    """Tiered temporal audit for no-advance and deadline constraints."""
    if gap_gwh <= strict_gwh:
        return "PASS_STRICT"
    if gap_gwh <= warn_gwh:
        return "WARN_NUMERICAL_RESIDUAL"
    if gap_gwh <= review_gwh:
        return "WARN_REVIEW_REQUIRED"
    if gap_gwh > fail_gwh:
        return "FAIL_STRUCTURAL"
    return "WARN_REVIEW_REQUIRED"


def classify_power_gap(gap_gw, strict_gw, warn_gw, review_gw, fail_gw):
    """Tiered power-cap audit."""
    if gap_gw <= strict_gw:
        return "PASS_STRICT"
    if gap_gw <= warn_gw:
        return "WARN_NUMERICAL_RESIDUAL"
    if gap_gw <= review_gw:
        return "WARN_REVIEW_REQUIRED"
    if gap_gw > fail_gw:
        return "FAIL_STRUCTURAL"
    return "WARN_REVIEW_REQUIRED"


def combine_audit_status(statuses):
    active = [s for s in statuses if s not in {None, "NOT_APPLICABLE"}]
    if "FAIL_STRUCTURAL" in active:
        return "FAIL_STRUCTURAL"
    if "WARN_REVIEW_REQUIRED" in active:
        return "WARN_REVIEW_REQUIRED"
    if "WARN_NUMERICAL_RESIDUAL" in active:
        return "WARN_NUMERICAL_RESIDUAL"
    return "PASS_STRICT"


def _province_zone_map(params, provinces):
    zones = list(getattr(params, "Province_z", []))
    if len(zones) < len(params.Province):
        raise ValueError(
            f"Province_z has {len(zones)} entries but Province has {len(params.Province)}"
        )
    return {
        str(pro): str(zones[idx])
        for idx, pro in enumerate(params.Province)
        if str(pro) in set(map(str, provinces))
    }


def _default_ai_hosting_class(province):
    """Policy-facing destination hosting class."""
    province = str(province)
    if province in AI_CLASS_A_WESTERN_NONREALTIME:
        return "A"
    if province in AI_CLASS_B_NATIONAL_CLUSTER:
        return "B"
    if province in AI_CLASS_C_EASTERN_DEMAND:
        return "C"
    if province in AI_CLASS_D_RESOURCE_CANDIDATE:
        return "D"
    return "E"


def _default_ai_network_tier(province):
    """Destination network tier = workload-reachability filter only (NOT hosting cap).

    Explicit tiers decoupled from the legacy AI_HUB_PROVINCES set so that
    resource-candidate provinces (e.g. XJ, YN) are no longer treated as
    strong low-latency nodes. Hosting concentration is governed separately by
    hosting class (see _default_ai_hosting_class).
    """
    province = str(province)
    # Tier 3: strong low-latency network / demand-side nodes
    if province in {"BJ", "SH", "TJ", "JS", "ZJ", "GD", "HE", "AH"}:
        return 3
    # Tier 2: national or regional hubs (near-real-time / offline feasible)
    if province in {"SC", "CQ", "GZ", "NM", "GS", "NX", "SD", "FJ"}:
        return 2
    # Tier 1: offline-only / resource candidates (XJ, YN, QH, ...) / local-service
    return 1


def _default_destination_network_tier(province):
    """Backward-compatible wrapper for legacy callers."""
    return _default_ai_network_tier(province)


def _required_min_tier_for_cluster(cluster_type):
    """Minimum network tier required by workload type."""
    cluster_type = str(cluster_type or "").lower()
    if cluster_type in {"realtime", "interactive", "latency_sensitive"}:
        return 3
    if cluster_type in {"near_realtime", "serving"}:
        return 2
    if cluster_type in {"batch", "training", "offline", "flexible"}:
        return 1
    return 1


def _build_zone_adjacency(trans_data, province_zone):
    """Derive zone adjacency from transmission topology.

    Returns zone_neighbors (dict of zone -> set of neighbor zones including self),
    adjacency_source, isolated_zones, and total_adjacent_pairs.
    """
    all_zones = sorted(set(province_zone.values()))
    zone_neighbors = {z: {z} for z in all_zones}
    all_pairs = set()
    for category_key in trans_data:
        cat = trans_data.get(category_key, {})
        if not isinstance(cat, dict):
            continue
        pairs = cat.get("pair", [])
        for p1, p2 in pairs:
            all_pairs.add((str(p1), str(p2)))
    for p1, p2 in all_pairs:
        if p1 not in province_zone or p2 not in province_zone:
            continue
        z1, z2 = province_zone[p1], province_zone[p2]
        if z1 != z2:
            zone_neighbors[z1].add(z2)
            zone_neighbors[z2].add(z1)
    isolated = [z for z in all_zones
                if len(zone_neighbors.get(z, {z}) - {z}) <= 1]
    total_pairs = sum(len(v - {z}) for z, v in zone_neighbors.items()) // 2
    return zone_neighbors, "topology", isolated, total_pairs


def _load_od_network_policy(policy_file):
    """Optional CSV policy file for OD network tiers and share caps."""
    od_network_score = {}
    od_network_tier = {}
    od_share_cap = {}
    if not policy_file:
        return od_network_score, od_network_tier, od_share_cap
    if not os.path.exists(policy_file):
        raise FileNotFoundError(f"AI_OD_NETWORK_POLICY_FILE not found: {policy_file}")

    df = pd.read_csv(policy_file)
    required_cols = {"source_cluster", "destination_province", "network_tier"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"AI_OD_NETWORK_POLICY_FILE missing columns: {sorted(missing)}"
        )

    for _, row in df.iterrows():
        gid = str(row["source_cluster"])
        dest = str(row["destination_province"])
        key = (gid, dest)
        od_network_tier[key] = int(row["network_tier"])
        if "network_score" in df.columns and not pd.isna(row["network_score"]):
            od_network_score[key] = float(row["network_score"])
        if "od_share_cap" in df.columns and not pd.isna(row["od_share_cap"]):
            od_share_cap[key] = float(row["od_share_cap"])

    return od_network_score, od_network_tier, od_share_cap


def _build_s4_od_parameters(
    source_cluster_arrival_mw,
    source_cluster_profile,
    source_cluster_mean_mw,
    source_cluster_zone,
    source_cluster_type,
    origin_reconstruction_weight,
    destination_host_power_cap_gw,
    destination_host_energy_cap_gwh_year,
    provinces,
    hours,
    flexible_pool_gw,
    province_zone,
    use_network_tier_caps=False,
    od_network_score=None,
    od_network_tier=None,
    od_share_cap=None,
    destination_network_tier=None,
    destination_hosting_class=None,
    destination_host_share_cap=None,
    od_class_share_cap=None,
    use_destination_tier_fallback=True,
    default_missing_tier=0,
    host_cap_scenario="legacy_tier",
    source_cluster_runtime_mean_mw=None,
    source_cluster_runtime_profile=None,
    runtime_pool_gw=None,
    zone_neighbors=None,
):
    """Normalize source-cluster interface data into model-ready S4-OD parameters.

    When zone_neighbors is provided, OD arcs are allowed to destinations in the
    same zone OR any adjacent zone (derived from transmission topology).
    When None (default), only same-zone destinations are allowed (backward compat).
    """

    source_clusters = sorted(str(g) for g in source_cluster_arrival_mw)
    destination_provinces = [str(p) for p in provinces]
    if not source_clusters:
        raise ValueError("S4-OD source-cluster interface has no source clusters.")
    od_network_score = od_network_score or {}
    od_network_tier = od_network_tier or {}
    od_share_cap = od_share_cap or {}
    destination_network_tier = destination_network_tier or {}
    destination_hosting_class = destination_hosting_class or {}
    destination_host_share_cap = destination_host_share_cap or {}
    od_class_share_cap = od_class_share_cap or {}
    host_cap_scenario = str(host_cap_scenario or "legacy_tier").strip().lower()
    if host_cap_scenario not in AI_SCENARIO_DEST_CAPS:
        raise ValueError(
            f"AI_HOST_CAP_SCENARIO must be one of {sorted(AI_SCENARIO_DEST_CAPS)}, "
            f"got {host_cap_scenario!r}"
        )

    cluster_arrival_gw = {}
    cluster_profile = {}
    cluster_mean_gw = {}
    cluster_total_arrival_gwh = {}
    cluster_zone = {}
    cluster_type = {}
    profile_mean_abs_error = {}
    profile_input_max_abs_diff = {}
    cluster_mean_input_actual_abs_diff = {}
    cluster_mean_input_actual_rel_diff = {}
    for gid in source_clusters:
        arrival = np.asarray(source_cluster_arrival_mw[gid], dtype=float)
        raw_profile = np.asarray(source_cluster_profile[gid], dtype=float)
        if len(arrival) != hours:
            raise ValueError(
                f"S4-OD cluster {gid} arrival length {len(arrival)} != hours={hours}"
            )
        if len(raw_profile) != hours:
            raise ValueError(
                f"S4-OD cluster {gid} profile length {len(raw_profile)} != hours={hours}"
            )
        input_mean_mw = float(source_cluster_mean_mw.get(gid, np.mean(arrival)))
        if input_mean_mw <= 1e-9:
            raise ValueError(
                f"S4-OD cluster {gid} has non-positive input mean arrival."
            )
        actual_mean_mw = float(np.mean(arrival))
        if actual_mean_mw <= 1e-9:
            raise ValueError(
                f"S4-OD cluster {gid} has non-positive actual mean arrival "
                "computed from hourly source_cluster_arrival_mw."
            )
        mean_mw = actual_mean_mw
        profile = arrival / mean_mw
        cluster_mean_input_actual_abs_diff[gid] = abs(input_mean_mw - actual_mean_mw)
        cluster_mean_input_actual_rel_diff[gid] = (
            abs(input_mean_mw - actual_mean_mw) / actual_mean_mw
            if actual_mean_mw > 1e-9
            else 0.0
        )
        raw_profile_mean = float(np.mean(raw_profile))
        if raw_profile_mean <= 1e-9:
            raise ValueError(f"S4-OD cluster {gid} has non-positive raw profile mean.")
        raw_profile_norm = raw_profile / raw_profile_mean
        profile_input_max_abs_diff[gid] = float(
            np.max(np.abs(raw_profile_norm - profile))
        )
        cluster_arrival_gw[gid] = arrival / 1000.0
        cluster_profile[gid] = profile
        cluster_mean_gw[gid] = mean_mw / 1000.0
        cluster_total_arrival_gwh[gid] = float(np.sum(arrival) / 1000.0)
        cluster_zone[gid] = str(source_cluster_zone.get(gid, ""))
        cluster_type[gid] = str(source_cluster_type.get(gid, ""))
        profile_mean_abs_error[gid] = abs(float(np.mean(profile)) - 1.0)

    origin_weight_by_cluster = {gid: {} for gid in source_clusters}
    origin_cluster_by_province = {}
    for (province, gid), weight in origin_reconstruction_weight.items():
        province = str(province)
        gid = str(gid)
        if gid not in origin_weight_by_cluster:
            continue
        origin_weight_by_cluster[gid][province] = float(weight)
        origin_cluster_by_province[province] = gid

    origin_weight_sum_abs_error = {}
    for gid in source_clusters:
        weight_sum = sum(origin_weight_by_cluster[gid].values())
        if not origin_weight_by_cluster[gid]:
            raise ValueError(f"S4-OD cluster {gid} has no origin reconstruction weights.")
        origin_weight_sum_abs_error[gid] = abs(weight_sum - 1.0)

    missing_destination_caps = [
        p for p in destination_provinces if p not in destination_host_power_cap_gw
    ]
    if missing_destination_caps:
        raise ValueError(
            "S4-OD destination power caps missing provinces: "
            f"{missing_destination_caps}"
        )
    destination_power_cap_gw = {
        p: float(destination_host_power_cap_gw[p]) for p in destination_provinces
    }
    missing_destination_energy_caps = [
        p for p in destination_provinces if p not in destination_host_energy_cap_gwh_year
    ]
    if missing_destination_energy_caps:
        raise ValueError(
            "S4-OD destination energy caps missing provinces: "
            f"{missing_destination_energy_caps}"
        )
    destination_energy_cap_gwh_year = {
        p: float(destination_host_energy_cap_gwh_year[p])
        for p in destination_provinces
    }
    destination_hosting_class_by_province = {
        p: str(destination_hosting_class.get(p, _default_ai_hosting_class(p)))
        for p in destination_provinces
    }
    allowed_destinations_by_cluster = {}
    clusters_by_destination = {p: [] for p in destination_provinces}
    network_tier_by_arc = {}
    network_score_by_arc = {}
    od_share_cap_by_arc = {}
    cross_zone_arcs = set()  # tracked when zone_neighbors is provided
    # Diagnostic switch: force all arc share_caps to 1.0 (per-arc unconstrained)
    unconstrained_arcs = str_to_bool(
        os.environ.get("AI_OD_UNCONSTRAINED_ARCS", "false"), default=False
    )
    if unconstrained_arcs:
        print("WARNING: AI_OD_UNCONSTRAINED_ARCS=true — diagnostic mode, "
              "per-arc share caps set to 1.0. Do NOT use for manuscript results.")
    tier_default_share_cap = {
        0: 0.0,
        1: float(os.environ.get("AI_OD_TIER1_SHARE_CAP", "0.05")),
        2: float(os.environ.get("AI_OD_TIER2_SHARE_CAP", "0.12")),
        3: float(os.environ.get("AI_OD_TIER3_SHARE_CAP", "0.30")),
    }
    for gid in source_clusters:
        zone = cluster_zone[gid]
        if zone == "":
            raise ValueError(f"S4-OD cluster {gid} has empty zone.")
        allowed = []
        for dest in destination_provinces:
            dest_zone = str(province_zone.get(dest, ""))
            # Allow same-zone OR adjacent-zone destinations (when zone_neighbors provided)
            if zone_neighbors is not None:
                if dest_zone != zone and dest_zone not in zone_neighbors.get(zone, {zone}):
                    continue
            elif dest_zone != zone:
                continue
            # Track cross-zone arcs for migration cost and audit
            is_cross_zone = (dest_zone != zone)
            if use_network_tier_caps:
                if (gid, dest) in od_network_tier:
                    tier = int(od_network_tier[(gid, dest)])
                elif use_destination_tier_fallback:
                    tier = int(destination_network_tier.get(dest, default_missing_tier))
                else:
                    tier = int(default_missing_tier)
                min_tier = _required_min_tier_for_cluster(cluster_type.get(gid, ""))
                share_cap = float(
                    od_share_cap.get((gid, dest), tier_default_share_cap.get(tier, 0.0))
                )
            else:
                tier = 1
                min_tier = 1
                share_cap = 1.0
            if tier < min_tier:
                continue
            host_class = destination_hosting_class_by_province.get(dest, "E")
            if host_cap_scenario != "legacy_tier":
                share_cap = float(
                    od_class_share_cap.get(
                        dest,
                        AI_SCENARIO_OD_CAPS[host_cap_scenario].get(host_class, 0.0),
                    )
                )
            if share_cap <= 0:
                continue
            allowed.append(dest)
            network_tier_by_arc[(gid, dest)] = int(tier)
            if (gid, dest) in od_network_score:
                network_score_by_arc[(gid, dest)] = float(od_network_score[(gid, dest)])
            od_share_cap_by_arc[(gid, dest)] = (
                1.0 if unconstrained_arcs else float(share_cap)
            )
            if is_cross_zone:
                cross_zone_arcs.add((gid, dest))
        if not allowed:
            raise ValueError(
                f"S4-OD cluster {gid} in zone {zone} has no allowed destination provinces."
            )
        allowed_destinations_by_cluster[gid] = allowed
        for dest in allowed:
            clusters_by_destination[dest].append(gid)
    cluster_profile_cum = {
        gid: np.cumsum(cluster_profile[gid]) for gid in source_clusters
    }

    # ---- Phase 2b dual-pool: runtime (time-shiftable) subset of siting ----
    # cluster_mean_gw above is the SITING mean (= legacy flexible). The runtime
    # mean is a per-cluster subset. When the interface does not provide runtime
    # means (single-pool / legacy callers), runtime := siting so behaviour is
    # unchanged. The runtime cumulative profile reuses the (shared) cluster
    # profile, scaled to the runtime mean.
    runtime_cluster_mean_gw = {}
    for gid in source_clusters:
        siting_mean_gw = float(cluster_mean_gw[gid])
        if source_cluster_runtime_mean_mw and gid in source_cluster_runtime_mean_mw:
            rt_gw = float(source_cluster_runtime_mean_mw[gid]) / 1000.0
        else:
            rt_gw = siting_mean_gw  # degrade: runtime == siting
        # Guard: runtime must not exceed siting at any cluster.
        if rt_gw > siting_mean_gw + 1e-9:
            raise ValueError(
                f"S4-OD cluster {gid} runtime mean {rt_gw} GW exceeds siting mean "
                f"{siting_mean_gw} GW. Check dual-pool interface generation."
            )
        runtime_cluster_mean_gw[gid] = max(0.0, rt_gw)
    runtime_cluster_profile = {}
    for gid in source_clusters:
        if source_cluster_runtime_profile and gid in source_cluster_runtime_profile:
            rt_profile = np.asarray(source_cluster_runtime_profile[gid], dtype=float)
            if len(rt_profile) != hours:
                raise ValueError(
                    f"S4-OD cluster {gid} runtime profile length {len(rt_profile)} "
                    f"!= hours={hours}"
                )
            rt_mean = float(np.mean(rt_profile))
            if rt_mean <= 1e-9:
                raise ValueError(
                    f"S4-OD cluster {gid} has non-positive runtime profile mean."
                )
            runtime_cluster_profile[gid] = rt_profile / rt_mean
        else:
            runtime_cluster_profile[gid] = cluster_profile[gid]
    runtime_cluster_profile_cum = {
        gid: np.cumsum(runtime_cluster_profile[gid]) for gid in source_clusters
    }
    # Per-cluster runtime fraction theta_g = runtime / siting. Stored separately
    # so the model can build runtime-only no-advance / deadline constraints.
    runtime_fraction_by_cluster = {
        gid: (runtime_cluster_mean_gw[gid] / cluster_mean_gw[gid])
        if cluster_mean_gw[gid] > 1e-12 else 0.0
        for gid in source_clusters
    }
    runtime_pool_gw_computed = sum(runtime_cluster_mean_gw.values())
    if runtime_pool_gw is None:
        runtime_pool_gw = runtime_pool_gw_computed

    cluster_pool_gw = sum(cluster_mean_gw.values())
    pool_abs_error = abs(cluster_pool_gw - float(flexible_pool_gw))
    max_profile_mean_abs_error = max(profile_mean_abs_error.values(), default=0.0)
    max_profile_input_abs_diff = max(profile_input_max_abs_diff.values(), default=0.0)
    profile_input_diff_warn_tol = float(
        os.environ.get("AI_CLUSTER_PROFILE_INPUT_DIFF_WARN_TOL", "1e-6")
    )
    if max_profile_input_abs_diff > profile_input_diff_warn_tol:
        print(
            "WARNING: source_cluster_profile differs from arrival/mean normalized profile. "
            f"max_profile_input_abs_diff={max_profile_input_abs_diff}, "
            f"warn_tol={profile_input_diff_warn_tol}. "
            "The model uses arrival/mean as the authoritative cluster profile."
        )
    max_origin_weight_sum_abs_error = max(
        origin_weight_sum_abs_error.values(), default=0.0
    )
    if pool_abs_error > 1e-6:
        raise ValueError(
            f"S4-OD cluster pool {cluster_pool_gw} GW != province flexible pool "
            f"{flexible_pool_gw} GW."
        )
    if max_profile_mean_abs_error > 1e-8:
        raise ValueError(
            f"S4-OD cluster profile mean audit failed: {max_profile_mean_abs_error}"
        )
    if max_origin_weight_sum_abs_error > 1e-8:
        raise ValueError(
            "S4-OD origin reconstruction weight audit failed: "
            f"{max_origin_weight_sum_abs_error}"
        )

    return {
        "source_clusters": source_clusters,
        "destination_provinces": destination_provinces,
        "cluster_arrival_gw": cluster_arrival_gw,
        "cluster_profile": cluster_profile,
        "cluster_mean_gw": cluster_mean_gw,
        "cluster_total_arrival_gwh": cluster_total_arrival_gwh,
        "cluster_zone": cluster_zone,
        "cluster_type": cluster_type,
        "origin_weight_by_cluster": origin_weight_by_cluster,
        "origin_cluster_by_province": origin_cluster_by_province,
        "allowed_destinations_by_cluster": allowed_destinations_by_cluster,
        "clusters_by_destination": clusters_by_destination,
        "od_network_tier": network_tier_by_arc,
        "od_network_score": network_score_by_arc,
        "od_share_cap": od_share_cap_by_arc,
        "cross_zone_arcs": cross_zone_arcs,
        "zone_neighbors": zone_neighbors,
        "destination_network_tier": destination_network_tier,
        "cluster_profile_cum": cluster_profile_cum,
        "runtime_cluster_profile": runtime_cluster_profile,
        "runtime_cluster_profile_cum": runtime_cluster_profile_cum,
        "runtime_cluster_mean_gw": runtime_cluster_mean_gw,
        "runtime_fraction_by_cluster": runtime_fraction_by_cluster,
        "runtime_pool_gw": float(runtime_pool_gw),
        "siting_pool_gw": float(cluster_pool_gw),
        "destination_power_cap_gw": destination_power_cap_gw,
        "destination_energy_cap_gwh_year": destination_energy_cap_gwh_year,
        "destination_hosting_class_by_province": destination_hosting_class_by_province,
        "host_cap_scenario": host_cap_scenario,
        "audit": {
            "source_cluster_count": len(source_clusters),
            "destination_province_count": len(destination_provinces),
            "cluster_pool_gw": float(cluster_pool_gw),
            "province_flexible_pool_gw": float(flexible_pool_gw),
            "cluster_pool_abs_error_gw": float(pool_abs_error),
            "max_profile_mean_abs_error": float(max_profile_mean_abs_error),
            "max_profile_input_abs_diff": float(max_profile_input_abs_diff),
            "profile_input_max_abs_diff_by_cluster": profile_input_max_abs_diff,
            "max_cluster_mean_input_actual_abs_diff_mw": float(
                max(cluster_mean_input_actual_abs_diff.values(), default=0.0)
            ),
            "max_cluster_mean_input_actual_rel_diff": float(
                max(cluster_mean_input_actual_rel_diff.values(), default=0.0)
            ),
            "cluster_mean_input_actual_abs_diff_mw_by_cluster": (
                cluster_mean_input_actual_abs_diff
            ),
            "cluster_mean_input_actual_rel_diff_by_cluster": (
                cluster_mean_input_actual_rel_diff
            ),
            "max_origin_weight_sum_abs_error": float(
                max_origin_weight_sum_abs_error
            ),
            "destination_power_cap_sum_gw": float(
                sum(destination_power_cap_gw.values())
            ),
            "destination_energy_cap_sum_gwh_year": float(
                sum(destination_energy_cap_gwh_year.values())
            ),
            "network_tier_policy_enabled": bool(use_network_tier_caps),
            "use_destination_tier_fallback": bool(use_destination_tier_fallback),
            "default_missing_tier": int(default_missing_tier),
            "od_share_cap_min": float(min(od_share_cap_by_arc.values(), default=0.0)),
            "od_share_cap_max": float(max(od_share_cap_by_arc.values(), default=0.0)),
            "network_tier_counts_allowed_arcs": {
                str(t): int(
                    sum(1 for v in network_tier_by_arc.values() if int(v) == t)
                )
                for t in [0, 1, 2, 3]
            },
            "od_share_cap_by_arc_count": int(len(od_share_cap_by_arc)),
            "destination_network_tier_counts": {
                str(t): int(
                    sum(1 for v in destination_network_tier.values() if int(v) == t)
                )
                for t in [0, 1, 2, 3]
            },
            "destination_hosting_class_by_province": destination_hosting_class_by_province,
            "host_cap_scenario": host_cap_scenario,
            "allowed_od_arc_count": int(
                sum(len(v) for v in allowed_destinations_by_cluster.values())
            ),
            "cross_zone_arc_count": int(len(cross_zone_arcs)),
            "cross_zone_arc_share": float(
                len(cross_zone_arcs)
                / max(1, sum(len(v) for v in allowed_destinations_by_cluster.values()))
            ),
            "max_allowed_destinations_per_cluster": int(
                max(len(v) for v in allowed_destinations_by_cluster.values())
            ),
            "zero_incoming_destination_count": int(
                sum(1 for v in clusters_by_destination.values() if len(v) == 0)
            ),
        },
    }


def add_cluster_od_planning_constraints(
    model,
    AI_OD_FLOW,
    PRO_AI_LOAD,
    ai_cluster_params,
    prefix,
    unmet_siting=None,
):
    """Add shared source-cluster OD planning constraints for S1 and S4-OD."""
    source_clusters = ai_cluster_params["source_clusters"]
    destination_provinces = ai_cluster_params["destination_provinces"]
    allowed_destinations_by_cluster = ai_cluster_params[
        "allowed_destinations_by_cluster"
    ]
    clusters_by_destination = ai_cluster_params["clusters_by_destination"]
    cluster_mean_gw = ai_cluster_params["cluster_mean_gw"]
    destination_energy_cap_gwh_year = ai_cluster_params[
        "destination_energy_cap_gwh_year"
    ]
    status = {
        "enabled": True,
        "source_balance_constraints": 0,
        "destination_plan_link_constraints": 0,
        "destination_energy_quota_constraints": 0,
    }

    for gid in source_clusters:
        unmet_term = (
            unmet_siting[gid]
            if unmet_siting is not None and gid in unmet_siting
            else 0.0
        )
        model.addConstr(
            gp.quicksum(
                AI_OD_FLOW[gid][dest]
                for dest in allowed_destinations_by_cluster[gid]
            )
            + unmet_term
            == float(cluster_mean_gw[gid]),
            name=f"{prefix}_source_balance_{gid}",
        )
        status["source_balance_constraints"] += 1

    for dest in destination_provinces:
        incoming_clusters = clusters_by_destination[dest]
        planned_dest = gp.quicksum(
            AI_OD_FLOW[gid][dest]
            for gid in incoming_clusters
            if gid in AI_OD_FLOW and dest in AI_OD_FLOW[gid]
        )
        model.addConstr(
            PRO_AI_LOAD[dest] == planned_dest,
            name=f"{prefix}_destination_plan_link_{dest}",
        )
        status["destination_plan_link_constraints"] += 1
        model.addConstr(
            planned_dest <= float(destination_energy_cap_gwh_year[dest]) / 8760.0,
            name=f"{prefix}_destination_energy_quota_{dest}",
        )
        status["destination_energy_quota_constraints"] += 1

    return status


def add_destination_hosting_share_constraints(
    model,
    PRO_AI_LOAD,
    destination_provinces,
    hourly_AI_load,
    destination_hosting_class,
    class_share_cap,
    prefix,
):
    """Limit total flexible AI hosted by each destination hosting class."""
    status = {
        "enabled": True,
        "constraints": 0,
        "class_share_cap": {
            str(k): float(v) for k, v in class_share_cap.items()
        },
        "destination_hosting_class": {
            str(k): str(v) for k, v in destination_hosting_class.items()
        },
        "by_destination": {},
    }

    for dest in destination_provinces:
        dest = str(dest)
        host_class = str(destination_hosting_class.get(dest, "E"))
        share_cap = float(class_share_cap.get(host_class, 0.0))
        cap_gw = share_cap * float(hourly_AI_load)
        model.addConstr(
            PRO_AI_LOAD[dest] <= cap_gw,
            name=f"{prefix}_destination_hosting_share_cap_{dest}",
        )
        status["constraints"] += 1
        status["by_destination"][dest] = {
            "hosting_class": str(host_class),
            "share_cap": float(share_cap),
            "cap_gw": float(cap_gw),
        }

    return status


def add_destination_network_share_constraints(
    model,
    PRO_AI_LOAD,
    destination_provinces,
    hourly_AI_load,
    destination_network_tier,
    tier_share_cap,
    prefix,
):
    """Backward-compatible wrapper for legacy callers."""
    destination_hosting_class = {
        str(dest): _default_ai_hosting_class(str(dest)) for dest in destination_provinces
    }
    class_share_cap = {
        "A": float(tier_share_cap.get(1, 0.0)),
        "B": float(tier_share_cap.get(2, 0.0)),
        "C": float(tier_share_cap.get(2, 0.0)),
        "D": float(tier_share_cap.get(3, 0.0)),
        "E": float(tier_share_cap.get(1, 0.0)),
    }
    return add_destination_hosting_share_constraints(
        model=model,
        PRO_AI_LOAD=PRO_AI_LOAD,
        destination_provinces=destination_provinces,
        hourly_AI_load=hourly_AI_load,
        destination_hosting_class=destination_hosting_class,
        class_share_cap=class_share_cap,
        prefix=prefix,
    )


def build_s0_cluster_fixed_load_mw(ai_cluster_params, provinces, hours):
    """Reconstruct fixed-origin flexible AI load from source clusters."""
    provinces = [str(p) for p in provinces]
    s0_load_mw = {p: np.zeros(hours, dtype=float) for p in provinces}

    for gid in ai_cluster_params.get("source_clusters", []):
        cluster_mean_gw = float(ai_cluster_params["cluster_mean_gw"][gid])
        cluster_profile = np.asarray(
            ai_cluster_params["cluster_profile"][gid], dtype=float
        )[:hours]
        cluster_hourly_mw = cluster_mean_gw * 1000.0 * cluster_profile
        for origin, weight in ai_cluster_params["origin_weight_by_cluster"][
            gid
        ].items():
            origin = str(origin)
            if origin in s0_load_mw:
                s0_load_mw[origin] += float(weight) * cluster_hourly_mw

    return s0_load_mw


def audit_s0_cluster_reconstruction(
    flexible_ai_load,
    s0_cluster_fixed_flexible_load_mw,
    provinces,
    hours,
):
    """Audit consistency between province-level flexible AI load and cluster reconstruction."""
    audit = {
        "units": "MW for hourly differences; MWh for energy differences",
        "hours": int(hours),
        "applied_to_load_demand": False,
        "note": (
            "This audit compares the province-level flexible_ai_load with the "
            "source-cluster reconstructed fixed-origin flexible load. When "
            "S0_USE_CLUSTER_RECONSTRUCTION=true, the reconstructed series is used "
            "in LOAD_DEMAND for the S0 scenario."
        ),
        "max_abs_diff_mw": 0.0,
        "max_abs_diff_gw": 0.0,
        "total_original_energy_mwh": 0.0,
        "total_reconstructed_energy_mwh": 0.0,
        "total_energy_diff_mwh": 0.0,
        "total_abs_energy_diff_mwh": 0.0,
        "national_hourly_max_abs_diff_mw": 0.0,
        "national_hourly_max_abs_diff_gw": 0.0,
        "by_province": {},
    }
    national_original = np.zeros(hours, dtype=float)
    national_reconstructed = np.zeros(hours, dtype=float)

    for pro in provinces:
        pro = str(pro)
        original = np.asarray(
            flexible_ai_load.get(pro, np.zeros(hours)), dtype=float
        )[:hours]
        reconstructed = np.asarray(
            s0_cluster_fixed_flexible_load_mw.get(pro, np.zeros(hours)),
            dtype=float,
        )[:hours]
        diff = reconstructed - original
        max_abs_diff_mw = float(np.max(np.abs(diff))) if len(diff) else 0.0
        original_energy_mwh = float(np.sum(original))
        reconstructed_energy_mwh = float(np.sum(reconstructed))
        energy_diff_mwh = reconstructed_energy_mwh - original_energy_mwh

        audit["by_province"][pro] = {
            "original_energy_mwh": original_energy_mwh,
            "reconstructed_energy_mwh": reconstructed_energy_mwh,
            "energy_diff_mwh": energy_diff_mwh,
            "max_abs_diff_mw": max_abs_diff_mw,
        }
        audit["max_abs_diff_mw"] = max(audit["max_abs_diff_mw"], max_abs_diff_mw)
        audit["total_original_energy_mwh"] += original_energy_mwh
        audit["total_reconstructed_energy_mwh"] += reconstructed_energy_mwh
        audit["total_abs_energy_diff_mwh"] += abs(energy_diff_mwh)
        national_original += original
        national_reconstructed += reconstructed

    audit["max_abs_diff_gw"] = audit["max_abs_diff_mw"] / 1000.0
    national_hourly_diff = national_reconstructed - national_original
    audit["national_hourly_max_abs_diff_mw"] = (
        float(np.max(np.abs(national_hourly_diff)))
        if len(national_hourly_diff)
        else 0.0
    )
    audit["national_hourly_max_abs_diff_gw"] = (
        audit["national_hourly_max_abs_diff_mw"] / 1000.0
    )
    audit["total_energy_diff_mwh"] = (
        audit["total_reconstructed_energy_mwh"]
        - audit["total_original_energy_mwh"]
    )
    original_total = audit["total_original_energy_mwh"]
    audit["total_energy_relative_gap"] = (
        audit["total_energy_diff_mwh"] / original_total
        if abs(original_total) > 1e-9
        else 0.0
    )
    return audit


def compute_ai_hosting_concentration_audit(
    results_AI_load,
    province_hosting_class,
    hourly_AI_load,
    class_share_cap=None,
    binding_tol=0.99,
):
    """Post-solve concentration audit for destination-hosted flexible AI.

    When ``class_share_cap`` (the per-hosting-class share cap dict for the
    active scenario) is provided, also reports each destination's hosting cap
    and which destinations are cap-binding (flow / cap >= ``binding_tol``).
    """
    flows = {
        str(pro): float(np.asarray(value, dtype=float).reshape(-1)[0])
        for pro, value in results_AI_load.items()
    }
    total = float(sum(flows.values()))
    ranked = sorted(flows.items(), key=lambda item: item[1], reverse=True)
    by_class = {}
    for pro, flow in flows.items():
        host_class = str(province_hosting_class.get(pro, "E"))
        by_class[host_class] = by_class.get(host_class, 0.0) + flow

    def top_share(n):
        if total <= 1e-12:
            return 0.0
        return float(sum(flow for _, flow in ranked[:n]) / total)

    # Herfindahl-Hirschman index of hosting shares (1/N to 1; higher = concentrated).
    hhi = float(
        sum((flow / total) ** 2 for flow in flows.values())
    ) if total > 1e-12 else 0.0

    class_share_cap = class_share_cap or {}
    binding_destinations = []
    if class_share_cap:
        for pro, flow in ranked:
            host_class = str(province_hosting_class.get(pro, "E"))
            share_cap = float(class_share_cap.get(host_class, 0.0))
            cap_gw = share_cap * float(hourly_AI_load)
            binding_ratio = float(flow / cap_gw) if cap_gw > 1e-12 else None
            binding_destinations.append(
                {
                    "province": pro,
                    "hosting_class": host_class,
                    "hosted_gw": float(flow),
                    "cap_gw": cap_gw,
                    "binding_ratio": binding_ratio,
                    "is_binding": bool(
                        binding_ratio is not None and binding_ratio >= binding_tol
                    ),
                }
            )

    return {
        "enabled": True,
        "unit": "GW average flexible AI hosted by destination province",
        "total_hosted_gw": total,
        "expected_flexible_pool_gw": float(hourly_AI_load),
        "total_gap_gw": float(total - float(hourly_AI_load)),
        "top1_hosting_share": top_share(1),
        "top3_hosting_share": top_share(3),
        "top5_hosting_share": top_share(5),
        "top10_hosting_share": top_share(10),
        "hhi_hosting": hhi,
        "top_destinations": [
            {
                "province": pro,
                "hosting_class": str(province_hosting_class.get(pro, "E")),
                "hosted_gw": float(flow),
                "share": float(flow / total if total > 1e-12 else 0.0),
            }
            for pro, flow in ranked[:10]
        ],
        "by_hosting_class_gw": {k: float(v) for k, v in sorted(by_class.items())},
        "by_hosting_class_share": {
            k: float(v / total if total > 1e-12 else 0.0)
            for k, v in sorted(by_class.items())
        },
        "binding_destinations": binding_destinations,
        "binding_count": int(sum(1 for d in binding_destinations if d["is_binding"])),
    }


def compute_od_concentration_audit(AI_OD_FLOW, ai_cluster_params):
    """Post-solve concentration audit for source-cluster OD allocation arcs."""
    source_top_share = {}
    source_top_destination = {}
    arc_records = []
    all_arc_total = 0.0
    for gid in AI_OD_FLOW:
        flows = {
            str(dest): float(getattr(AI_OD_FLOW[gid][dest], "x", AI_OD_FLOW[gid][dest]))
            for dest in AI_OD_FLOW[gid]
        }
        source_total = float(sum(flows.values()))
        ranked = sorted(flows.items(), key=lambda item: item[1], reverse=True)
        source_top_share[str(gid)] = (
            float(ranked[0][1] / source_total)
            if source_total > 1e-12 and ranked
            else 0.0
        )
        source_top_destination[str(gid)] = ranked[0][0] if ranked else None
        all_arc_total += source_total
        for dest, flow in ranked:
            arc_records.append(
                {
                    "source_cluster": str(gid),
                    "destination_province": str(dest),
                    "flow_gw": float(flow),
                    "share_of_source": float(
                        flow / source_total if source_total > 1e-12 else 0.0
                    ),
                    "destination_hosting_class": str(
                        ai_cluster_params.get(
                            "destination_hosting_class_by_province", {}
                        ).get(str(dest), "E")
                    ),
                    "network_tier": int(
                        ai_cluster_params.get("od_network_tier", {}).get(
                            (str(gid), str(dest)), 0
                        )
                    ),
                }
            )
    ranked_arcs = sorted(arc_records, key=lambda item: item["flow_gw"], reverse=True)

    return {
        "enabled": True,
        "unit": "GW average flexible AI OD allocation",
        "total_od_flow_gw": float(all_arc_total),
        "source_cluster_count": int(len(AI_OD_FLOW)),
        "arc_count": int(len(arc_records)),
        "max_source_top1_destination_share": float(
            max(source_top_share.values(), default=0.0)
        ),
        "source_top1_destination_share": source_top_share,
        "source_top_destination": source_top_destination,
        "top10_arcs": ranked_arcs[:10],
    }


def _resolve_extreme_cf_input_dir(args):
    """Return the power-system input directory for the selected CF stress scenario."""
    scenario = str(
        _get_arg(args, "extreme_cf_scenario", os.environ.get("EXTREME_CF_SCENARIO", "none"))
        or "none"
    ).strip().lower()
    event_type = str(
        _get_arg(args, "extreme_cf_event_type", os.environ.get("EXTREME_CF_EVENT_TYPE", "both"))
        or "both"
    ).strip().lower()
    factor_kind = str(
        _get_arg(args, "extreme_cf_factor_kind", os.environ.get("EXTREME_CF_FACTOR_KIND", "mean"))
        or "mean"
    ).strip().lower()
    if scenario not in EXTREME_CF_SCENARIOS:
        raise ValueError(
            f"extreme_cf_scenario must be one of {sorted(EXTREME_CF_SCENARIOS)}, got {scenario!r}"
        )
    if event_type not in EXTREME_CF_EVENT_TYPES:
        raise ValueError(
            f"extreme_cf_event_type must be one of {sorted(EXTREME_CF_EVENT_TYPES)}, got {event_type!r}"
        )
    if factor_kind not in EXTREME_CF_FACTOR_KINDS:
        raise ValueError(
            f"extreme_cf_factor_kind must be one of {sorted(EXTREME_CF_FACTOR_KINDS)}, got {factor_kind!r}"
        )
    if scenario == "none":
        path = get_power_system_data_dir()
    else:
        base_dir = str(
            _get_arg(args, "extreme_cf_base_dir", os.environ.get("EXTREME_CF_BASE_DIR", ""))
            or ""
        ).strip()
        if not base_dir:
            base_dir = os.path.join(
                PROJECT_ROOT,
                "inputs",
                "power_system_cf_modified",
            )
        path = os.path.join(
            os.path.abspath(os.path.expanduser(base_dir)),
            f"gfdl_ssp126_{event_type}_{scenario}_{factor_kind}",
        )
    required = [
        "wind_cf.pkl",
        "pv_cf.pkl",
        "wind_cell.pkl",
        "pv_cell.pkl",
        "wind_cap.pkl",
        "pv_cap.pkl",
        "wind_lcoe.pkl",
        "pv_lcoe.pkl",
    ]
    missing = [name for name in required if not os.path.exists(os.path.join(path, name))]
    if missing:
        raise FileNotFoundError(
            f"Extreme CF input directory {path!r} is missing required files: {missing}"
        )
    return path, {
        "extreme_cf_scenario": scenario,
        "extreme_cf_event_type": event_type,
        "extreme_cf_factor_kind": factor_kind,
        "power_system_data_dir": path,
        "metadata_file": os.path.join(path, "cf_modifier_metadata.json"),
        "audit_file": os.path.join(path, "cf_modifier_audit.csv"),
    }


class LCOEModel:
    def __init__(self, popt):
        self.popt = popt

    def __call__(self, x):
        """
        使得实例可以像函数一样被调用: model(2060)
        """
        dt = np.array(x) - 2022
        a, b1, b2, c = self.popt
        return a * np.exp(-b1 * dt) + (1 - a - c) * np.exp(-b2 * dt) + c


def National_energy_model(args):

    path, extreme_cf_metadata = _resolve_extreme_cf_input_dir(args)
    os.environ["POWER_SYSTEM_DATA_DIR"] = path
    print(
        "Using power-system data directory:",
        json.dumps(extreme_cf_metadata, ensure_ascii=False),
    )
    lcoe = load_lcoe()
    trans_data = get_trans_data()
    Params = OmegaConf.load(get_param_yaml_path())
    if OmegaConf.is_config(args):
        vre_min_utilization = OmegaConf.select(args, "vre_min_utilization", default=0.0)
    else:
        vre_min_utilization = getattr(args, "vre_min_utilization", 0.0)
    vre_min_utilization = float(vre_min_utilization)

    # China Energy Outlook 2060 (2024) electricity supply path, 100 million kWh.
    # This replaces the old compounding growth list, which implied 2060/2020 ~= 3.97.
    load_growth_years = np.array([2020, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060])
    load_growth_values = np.array([77792, 98346, 117808, 134330, 146700, 157485, 166360, 171465, 175000])
    load_growth_factors = load_growth_values / load_growth_values[0]
    load_reshapping = float(np.interp(args.test_years, load_growth_years, load_growth_factors))
    print("China Energy Outlook 2060 load growth factors:", dict(zip(load_growth_years, load_growth_factors)))
    print(load_reshapping, "load growth factor")
    other_new_installed_cap = (1 + args.other_conf) ** (args.test_years - 2020)
    hydro_new_install_cap_conf = (1 + 0.015) ** (args.test_years - 2020)
    nuclear_new_install_cap_conf = (1 + 0.042) ** (args.test_years - 2020)
    # Coal variable cost year multiplier (conservative: fuel escalation only, no carbon price)
    COAL_PRICE_YEAR_MULTIPLIER = {
        2020: 1.00,
        2025: 1.15,
        2030: 1.20,
        2035: 1.25,
        2040: 1.30,
        2045: 1.35,
        2050: 1.40,
        2055: 1.45,
        2060: 1.50,
    }
    if args.test:
        print("start test mode")
        HOURS = args.test_hours
        PRO_NUMS = args.test_pro_num
    else:
        print("start normal mode")
        HOURS = Params.Hours
        PRO_NUMS = Params.PRO_NUMS
    HORIZON_SCALE = HOURS / 8760.0
    Province = Params.Province[:PRO_NUMS]
    ai_operational_scenario = str(
        _get_arg(
            args,
            "ai_operational_scenario",
            os.environ.get("AI_OPERATIONAL_SCENARIO", "S1"),
        )
    ).strip().upper()
    if ai_operational_scenario not in AI_OPERATIONAL_SCENARIOS:
        raise ValueError(
            "Main manuscript AI scenarios are restricted to S0, S1 and S4. "
            f"Got {ai_operational_scenario!r}. S2/S3 have been removed from the main model."
        )
    ai_batch_delay_hours = int(
        _get_arg(args, "ai_batch_delay_hours", os.environ.get("AI_BATCH_DELAY_HOURS", 6))
    )
    ai_batch_delay_hours = max(0, ai_batch_delay_hours)
    ai_batch_power_cap_multiplier = float(
        _get_arg(
            args,
            "ai_batch_power_cap_multiplier",
            os.environ.get("AI_BATCH_POWER_CAP_MULTIPLIER", 1.5),
        )
    )
    ai_batch_power_cap_multiplier = max(1.0, ai_batch_power_cap_multiplier)
    h2_scenario_default = "NONE"
    h2_scenario = str(
        _get_arg(args, "h2_scenario", os.environ.get("H2_SCENARIO", h2_scenario_default))
        or h2_scenario_default
    ).strip().upper()
    if h2_scenario in {"NOH2", "NO_H2", "NO-H2", "NONE"}:
        h2_scenario = "NONE"
    if h2_scenario not in H2_SCENARIOS:
        raise ValueError(
            f"h2_scenario must be one of {sorted(H2_SCENARIOS)}, got {h2_scenario!r}"
        )
    h2_aboveground_allowed = h2_scenario in {"AG", "UG"}
    h2_underground_allowed = h2_scenario == "UG"
    province_zone = _province_zone_map(Params, Province)
    # load demand data
    load_demand = pd.read_csv(
        get_load_demand_path(),
        delimiter=",",
    ).to_numpy()[:, 1:]
    offwind_loce = pd.read_csv(
        os.path.join(path, "offshorewind_lcoe.csv"),
        delimiter=",",
        header=None,
    ).to_numpy()
    offwind_cf = pd.read_csv(
        os.path.join(path, "offshore_cf.csv"),
        delimiter=",",
        header=None,
    ).to_numpy()
    offwind_pro = ["FJ", "GD", "GX", "HI", "HE", "JS", "LN", "SD", "SH", "TJ", "ZJ"]

    load_demand = np.asarray(load_demand, dtype=float)
    load_index = []
    all_AI_load = 0
    non_ai_base_load_mw = {}
    non_ai_base_energy_mwh = 0.0
    fixed_ai_energy_mwh = 0.0
    flexible_ai_energy_mwh = 0.0
    ai_total_energy_mwh = 0.0
    total_with_ai_energy_mwh = 0.0
    ai_to_non_ai_base_ratio = None
    ai_share_of_total_after_addition = None
    flexible_share_of_ai = None
    energy_accounting_mwh = {}
    ai_interface_metadata = {
        "use_external_ai_load": False,
        "extreme_cf": dict(extreme_cf_metadata),
    }
    fixed_ai_load = {}
    flexible_ai_load = {}
    ai_hosting_capacity_gw = {}
    ai_hosting_penalty_rmb_per_mwh = {}
    ai_flexible_load_gw = {}
    ai_hosting_lb_gw = {}
    ai_hosting_ub_gw = {}
    ai_batch_power_cap_gw = {}
    use_source_cluster_ai_interface = False
    ai_source_cluster_arrival_mw = {}
    ai_source_cluster_profile = {}
    ai_source_cluster_mean_mw = {}
    ai_source_cluster_runtime_mean_mw = {}
    ai_source_cluster_runtime_profile = {}
    ai_source_cluster_siting_mean_mw = {}
    ai_interface_siting_pool_gw = 0.0
    ai_interface_runtime_pool_gw = 0.0
    ai_interface_dual_pool = False
    ai_source_cluster_zone = {}
    ai_source_cluster_type = {}
    ai_origin_reconstruction_weight = {}
    ai_destination_host_power_cap_gw = {}
    ai_destination_host_energy_cap_gwh_year = {}
    ai_s4_od_params = {}
    use_network_tier_caps = str_to_bool(
        os.environ.get("AI_USE_NETWORK_TIER_CAPS", "true"), default=True
    )
    ai_od_network_policy_file = os.environ.get("AI_OD_NETWORK_POLICY_FILE", "")
    ai_od_default_missing_tier = int(
        os.environ.get("AI_OD_DEFAULT_MISSING_TIER", "0")
    )
    ai_use_destination_tier_fallback = str_to_bool(
        os.environ.get("AI_USE_DESTINATION_TIER_FALLBACK", "true"),
        default=True,
    )
    ai_od_tier1_share_cap = float(os.environ.get("AI_OD_TIER1_SHARE_CAP", "0.05"))
    ai_od_tier2_share_cap = float(os.environ.get("AI_OD_TIER2_SHARE_CAP", "0.12"))
    ai_od_tier3_share_cap = float(os.environ.get("AI_OD_TIER3_SHARE_CAP", "0.30"))
    ai_dest_tier1_share_cap = float(os.environ.get("AI_DEST_TIER1_SHARE_CAP", "0.05"))
    ai_dest_tier2_share_cap = float(os.environ.get("AI_DEST_TIER2_SHARE_CAP", "0.12"))
    ai_dest_tier3_share_cap = float(os.environ.get("AI_DEST_TIER3_SHARE_CAP", "0.30"))
    ai_host_cap_scenario = str(
        _get_arg(args, "AI_HOST_CAP_SCENARIO", os.environ.get("AI_HOST_CAP_SCENARIO", "central"))
    ).strip().lower()
    if ai_host_cap_scenario not in AI_SCENARIO_DEST_CAPS:
        raise ValueError(
            f"AI_HOST_CAP_SCENARIO must be one of {sorted(AI_SCENARIO_DEST_CAPS)}, "
            f"got {ai_host_cap_scenario!r}"
        )
    if ai_host_cap_scenario == "legacy_tier":
        print(
            "WARNING: AI_HOST_CAP_SCENARIO=legacy_tier is a diagnostic/legacy mode "
            "and must NOT be used for main manuscript results. Use "
            "conservative/central/liberal/extreme_upper_bound instead."
        )
    s0_cluster_fixed_flexible_load_mw = {}
    s0_reconstruction_audit = None
    s0_use_cluster_reconstruction = str_to_bool(
        os.environ.get("S0_USE_CLUSTER_RECONSTRUCTION"), default=True
    )
    strict_cluster_ai_interface = str_to_bool(
        os.environ.get("STRICT_CLUSTER_AI_INTERFACE", "true"), default=True
    )
    strict_postsolve_ai_audit = str_to_bool(
        os.environ.get("STRICT_POSTSOLVE_AI_AUDIT", "true"), default=True
    )
    raise_on_structural_ai_audit_fail = str_to_bool(
        os.environ.get("RAISE_ON_STRUCTURAL_AI_AUDIT_FAIL", "false"),
        default=False,
    )
    ai_energy_audit_abs_tol_gwh = float(
        os.environ.get(
            "AI_ENERGY_AUDIT_ABS_TOL_GWH",
            os.environ.get(
                "AI_ENERGY_AUDIT_TOL_GWH",
                os.environ.get("AI_AUDIT_TOL_GWH", "0.01"),
            ),
        )
    )
    ai_energy_audit_rel_tol = float(os.environ.get("AI_ENERGY_AUDIT_REL_TOL", "1e-6"))
    ai_temporal_audit_tol_gwh = float(
        os.environ.get("AI_TEMPORAL_AUDIT_TOL_GWH", "1e-4")
    )
    ai_power_audit_tol_gw = float(os.environ.get("AI_POWER_AUDIT_TOL_GW", "1e-6"))
    ai_energy_warn_abs_tol_gwh = float(
        os.environ.get("AI_ENERGY_WARN_ABS_TOL_GWH", "0.02")
    )
    ai_energy_warn_rel_tol = float(os.environ.get("AI_ENERGY_WARN_REL_TOL", "5e-6"))
    ai_energy_review_abs_tol_gwh = float(
        os.environ.get("AI_ENERGY_REVIEW_ABS_TOL_GWH", "0.1")
    )
    ai_energy_review_rel_tol = float(
        os.environ.get("AI_ENERGY_REVIEW_REL_TOL", "2e-5")
    )
    ai_energy_fail_abs_tol_gwh = float(
        os.environ.get("AI_ENERGY_FAIL_ABS_TOL_GWH", "0.1")
    )
    ai_energy_fail_rel_tol = float(os.environ.get("AI_ENERGY_FAIL_REL_TOL", "5e-5"))
    ai_temporal_warn_tol_gwh = float(
        os.environ.get("AI_TEMPORAL_WARN_TOL_GWH", "0.001")
    )
    ai_temporal_review_tol_gwh = float(
        os.environ.get("AI_TEMPORAL_REVIEW_TOL_GWH", "0.1")
    )
    ai_temporal_fail_tol_gwh = float(
        os.environ.get("AI_TEMPORAL_FAIL_TOL_GWH", "0.1")
    )
    ai_power_warn_tol_gw = float(os.environ.get("AI_POWER_WARN_TOL_GW", "1e-4"))
    ai_power_review_tol_gw = float(
        os.environ.get("AI_POWER_REVIEW_TOL_GW", "1e-2")
    )
    ai_power_fail_tol_gw = float(os.environ.get("AI_POWER_FAIL_TOL_GW", "1e-2"))
    ai_audit_tol_gwh = ai_energy_audit_abs_tol_gwh
    s0_reconstruction_energy_tol = float(
        os.environ.get("S0_CLUSTER_RECONSTRUCTION_ENERGY_TOL", "1e-6")
    )
    s0_reconstruction_hourly_tol_gw = float(
        os.environ.get("S0_CLUSTER_RECONSTRUCTION_HOURLY_TOL_GW", "1e-4")
    )
    for i in range(len(Params.Province)):
        try:
            # print(Params.load_demand_pro_index.index(Params.Province[i]),Params.Province[i])
            load_index.append(Params.load_demand_pro_index.index(Params.Province[i]))
        except:
            load_index.append(-1)
            # print('error1',Params.Province[i])
    LOAD_DEMAND = {}
    use_external_ai_load = str_to_bool(os.environ.get("USE_EXTERNAL_AI_LOAD"), default=False)
    include_ai_penalty_in_objective = str_to_bool(
        os.environ.get("INCLUDE_AI_PENALTY_IN_OBJECTIVE", "false"),
        default=False,
    )
    compute_ai_penalty_diagnostic = str_to_bool(
        os.environ.get("COMPUTE_AI_PENALTY_DIAGNOSTIC", "true"),
        default=True,
    )
    # ---- Phase C: migration cost parameters (scan variable, no hardcoded real value) ----
    ai_migration_ctilde = float(os.environ.get("AI_MIGRATION_CTILDE", "0.0"))
    ai_migration_scale_per_gw_year = float(
        os.environ.get("AI_MIGRATION_SCALE_PER_GW_YEAR", "0.0")
    )
    c_mig_per_gw_year = ai_migration_ctilde * ai_migration_scale_per_gw_year
    ai_s4_zone_runtime_transfer = str_to_bool(
        os.environ.get("AI_S4_ZONE_RUNTIME_TRANSFER"), default=False
    )
    ai_rt_zone_route_mode = str(
        os.environ.get("AI_RT_ZONE_ROUTE_MODE", "adjacent")
    ).strip().lower()
    if ai_rt_zone_route_mode not in {"none", "adjacent", "all"}:
        raise ValueError(
            f"AI_RT_ZONE_ROUTE_MODE must be one of none/adjacent/all, got "
            f"{ai_rt_zone_route_mode!r}"
        )
    # Phase C: zone-neighbor mode switch for Run-NoMig vs Run-FreeMig.
    # "adjacent" (default): same-zone + adjacent-zone OD.
    # "none": same-zone only (Run-NoMig baseline for 0.1b).
    ai_zone_neighbors_mode = str(
        os.environ.get("AI_ZONE_NEIGHBORS", "adjacent")
    ).strip().lower()
    if ai_zone_neighbors_mode not in {"adjacent", "none"}:
        raise ValueError(
            f"AI_ZONE_NEIGHBORS must be 'adjacent' or 'none', got "
            f"{ai_zone_neighbors_mode!r}"
        )
    # Convert to million yuan per GW-year for objective (objective unit = 百万 元)
    if strict_cluster_ai_interface and not use_external_ai_load:
        raise ValueError(
            "Main S0/S1/S4 experiments require USE_EXTERNAL_AI_LOAD=true "
            "under STRICT_CLUSTER_AI_INTERFACE=true. Refusing to enter the legacy AI branch. "
            "Set STRICT_CLUSTER_AI_INTERFACE=false only for legacy diagnostic runs."
        )
    if use_external_ai_load:
        ai_load_file = os.environ.get("AI_LOAD_FILE")
        ai_load_metadata_file = os.environ.get("AI_LOAD_METADATA_FILE")
        ai_load_scenario = os.environ.get("AI_LOAD_SCENARIO")
        ai_allocation_version = os.environ.get("AI_ALLOCATION_VERSION")
        ai_load_year_text = os.environ.get("AI_LOAD_YEAR")
        ai_load_year = int(ai_load_year_text) if ai_load_year_text else args.test_years
        if not ai_load_file:
            raise ValueError("USE_EXTERNAL_AI_LOAD=true requires AI_LOAD_FILE.")
        if os.path.isdir(ai_load_file):
            source_cluster_interface = load_source_cluster_interface(
                interface_dir=ai_load_file,
                provinces=list(Province),
                hours=HOURS,
                scenario=ai_load_scenario,
                allocation_version=ai_allocation_version,
                year=ai_load_year,
            )
            external_ai = source_cluster_interface.province_load
            use_source_cluster_ai_interface = True
            ai_source_cluster_arrival_mw = (
                source_cluster_interface.source_cluster_arrival_mw
            )
            ai_source_cluster_profile = source_cluster_interface.source_cluster_profile
            ai_source_cluster_mean_mw = source_cluster_interface.source_cluster_mean_mw
            ai_source_cluster_zone = source_cluster_interface.source_cluster_zone
            ai_source_cluster_type = source_cluster_interface.source_cluster_type
            ai_origin_reconstruction_weight = (
                source_cluster_interface.origin_reconstruction_weight
            )
            ai_destination_host_power_cap_gw = (
                source_cluster_interface.destination_host_power_cap_gw
            )
            ai_destination_host_energy_cap_gwh_year = (
                source_cluster_interface.destination_host_energy_cap_gwh_year
            )
            # ---- Phase 2b dual-pool: capture runtime/siting pool info ----
            ai_source_cluster_runtime_mean_mw = getattr(
                source_cluster_interface, "source_cluster_runtime_mean_mw", {}
            ) or {}
            ai_source_cluster_runtime_profile = getattr(
                source_cluster_interface, "source_cluster_runtime_profile", {}
            ) or {}
            ai_source_cluster_siting_mean_mw = getattr(
                source_cluster_interface, "source_cluster_siting_mean_mw", {}
            ) or {}
            ai_interface_siting_pool_gw = float(
                getattr(source_cluster_interface, "siting_pool_gw", 0.0) or 0.0
            )
            ai_interface_runtime_pool_gw = float(
                getattr(source_cluster_interface, "runtime_pool_gw", 0.0) or 0.0
            )
            ai_interface_dual_pool = bool(
                getattr(source_cluster_interface, "dual_pool", False)
            )
        else:
            external_ai = load_external_ai_load(
                file_path=ai_load_file,
                provinces=list(Province),
                hours=HOURS,
                scenario=ai_load_scenario,
                allocation_version=ai_allocation_version,
                year=ai_load_year,
                metadata_file_path=ai_load_metadata_file,
            )
        fixed_ai_load = external_ai.fixed_ai_load_mw
        flexible_ai_load = external_ai.flexible_ai_load_mw
        ai_hosting_capacity_gw = external_ai.ai_hosting_capacity_gw
        if ai_destination_host_power_cap_gw:
            ai_hosting_capacity_gw.update(ai_destination_host_power_cap_gw)
        ai_hosting_penalty_rmb_per_mwh = external_ai.ai_hosting_penalty_rmb_per_mwh
        ai_local_retention_min = external_ai.ai_local_retention_min
        ai_max_host_share = external_ai.ai_max_host_share
        ai_hosting_upper_bound_gw = external_ai.ai_hosting_upper_bound_gw
        if use_source_cluster_ai_interface:
            expected_s4_od_provinces = len(external_ai.fixed_ai_load_mw)
            if (
                ai_operational_scenario in {"S0", "S1", "S4"}
                and len(Province) != expected_s4_od_provinces
            ):
                raise ValueError(
                    f"{ai_operational_scenario}-OD source-cluster interface requires all provinces. "
                    "Do not use test_pro_num/province subsetting unless a matching "
                    f"subset source-cluster interface is generated. Got {len(Province)} "
                        f"model provinces, expected {expected_s4_od_provinces}."
                )
            if ai_operational_scenario == "S0":
                # Main manuscript S0 uses the same source-cluster workload basis
                # as S1 and S4-OD whenever that interface is available.
                s0_use_cluster_reconstruction = True
            od_network_score, od_network_tier, od_share_cap = _load_od_network_policy(
                ai_od_network_policy_file
            )
            # Build zone adjacency from transmission topology (always, for audit)
            zone_neighbors_full, adj_source, isolated_zones, adj_pair_count = (
                _build_zone_adjacency(trans_data, province_zone)
            )
            # Apply mode switch: "none" degrades to same-zone only (Run-NoMig)
            if ai_zone_neighbors_mode == "none":
                zone_neighbors = None
            else:
                zone_neighbors = zone_neighbors_full
            destination_network_tier = {
                str(pro): _default_destination_network_tier(str(pro))
                for pro in Province
            }
            destination_hosting_class = dict(
                external_ai.metadata.get("destination_hosting_class_from_interface", {})
                or {}
            )
            ai_s4_od_params = _build_s4_od_parameters(
                source_cluster_arrival_mw=ai_source_cluster_arrival_mw,
                source_cluster_profile=ai_source_cluster_profile,
                source_cluster_mean_mw=ai_source_cluster_mean_mw,
                source_cluster_zone=ai_source_cluster_zone,
                source_cluster_type=ai_source_cluster_type,
                origin_reconstruction_weight=ai_origin_reconstruction_weight,
                destination_host_power_cap_gw=ai_destination_host_power_cap_gw,
                destination_host_energy_cap_gwh_year=(
                    ai_destination_host_energy_cap_gwh_year
                ),
                provinces=list(Province),
                hours=HOURS,
                flexible_pool_gw=external_ai.flexible_pool_gw,
                province_zone=province_zone,
                use_network_tier_caps=use_network_tier_caps,
                od_network_score=od_network_score,
                od_network_tier=od_network_tier,
                od_share_cap=od_share_cap,
                destination_network_tier=destination_network_tier,
                destination_hosting_class={
                    str(pro): destination_hosting_class.get(
                        str(pro), _default_ai_hosting_class(str(pro))
                    )
                    for pro in Province
                },
                destination_host_share_cap={
                    str(pro): AI_SCENARIO_DEST_CAPS[ai_host_cap_scenario].get(
                        destination_hosting_class.get(
                            str(pro), _default_ai_hosting_class(str(pro))
                        ),
                        0.0,
                    )
                    for pro in Province
                },
                od_class_share_cap={
                    str(pro): AI_SCENARIO_OD_CAPS[ai_host_cap_scenario].get(
                        destination_hosting_class.get(
                            str(pro), _default_ai_hosting_class(str(pro))
                        ),
                        0.0,
                    )
                    for pro in Province
                },
                use_destination_tier_fallback=ai_use_destination_tier_fallback,
                default_missing_tier=ai_od_default_missing_tier,
                host_cap_scenario=ai_host_cap_scenario,
                source_cluster_runtime_mean_mw=ai_source_cluster_runtime_mean_mw,
                source_cluster_runtime_profile=ai_source_cluster_runtime_profile,
                runtime_pool_gw=(
                    ai_interface_runtime_pool_gw
                    if ai_interface_dual_pool
                    else None
                ),
                zone_neighbors=zone_neighbors,
            )
            s0_cluster_fixed_flexible_load_mw = build_s0_cluster_fixed_load_mw(
                ai_cluster_params=ai_s4_od_params,
                provinces=list(Province),
                hours=HOURS,
            )
            s0_reconstruction_audit = audit_s0_cluster_reconstruction(
                flexible_ai_load=flexible_ai_load,
                s0_cluster_fixed_flexible_load_mw=s0_cluster_fixed_flexible_load_mw,
                provinces=list(Province),
                hours=HOURS,
            )
            s0_reconstruction_audit["applied_to_load_demand"] = bool(
                s0_use_cluster_reconstruction
            )
            if strict_cluster_ai_interface:
                rel_gap = abs(
                    float(s0_reconstruction_audit["total_energy_relative_gap"])
                )
                hourly_gap_gw = float(
                    s0_reconstruction_audit["national_hourly_max_abs_diff_gw"]
                )
                if rel_gap > s0_reconstruction_energy_tol:
                    raise ValueError(
                        "S0 source-cluster reconstruction energy mismatch. "
                        f"relative_gap={rel_gap}, tol={s0_reconstruction_energy_tol}. "
                        "Check source-cluster interface generation."
                    )
                if hourly_gap_gw > s0_reconstruction_hourly_tol_gw:
                    print(
                        "WARNING: S0 source-cluster reconstruction has non-negligible "
                        "hourly mismatch: "
                        f"national_hourly_max_abs_diff_gw={hourly_gap_gw}, "
                        f"tol={s0_reconstruction_hourly_tol_gw}."
                    )
        ai_interface_metadata = dict(external_ai.metadata or {})
        ai_interface_metadata["dual_pool_interface"] = {
            "dual_pool": bool(ai_interface_dual_pool),
            "siting_pool_gw": float(ai_interface_siting_pool_gw),
            "runtime_pool_gw": float(ai_interface_runtime_pool_gw),
            "note": (
                "siting_pool_gw == flexible_pool_gw (legacy). runtime_pool_gw is "
                "the time-shiftable subset. When dual_pool is False the data "
                "layer did not provide runtime columns and runtime == siting."
            ),
        }
        ai_interface_metadata["ai_penalty_policy"] = {
            "include_ai_penalty_in_objective": bool(include_ai_penalty_in_objective),
            "compute_ai_penalty_diagnostic": bool(compute_ai_penalty_diagnostic),
            "interpretation": (
                "Main experiments exclude AI hosting/migration preference penalties "
                "from the power-system objective by default. The penalty is retained "
                "as a diagnostic or optional sensitivity term."
            ),
            "recommended_main_result_cost": "reported_power_system_cost",
        }
        ai_interface_metadata["use_external_ai_load"] = True
        ai_interface_metadata["use_source_cluster_ai_interface"] = (
            use_source_cluster_ai_interface
        )
        ai_interface_metadata["strict_cluster_ai_interface"] = bool(
            strict_cluster_ai_interface
        )
        if strict_cluster_ai_interface and ai_operational_scenario in {"S0", "S1", "S4"}:
            if not use_source_cluster_ai_interface or not ai_s4_od_params:
                raise ValueError(
                    f"{ai_operational_scenario} requires a source-cluster AI interface "
                    "under STRICT_CLUSTER_AI_INTERFACE=true. Use an AI_LOAD_FILE "
                    "directory generated by the source-cluster interface, not a "
                    "province-level single-file AI load."
                )
        ai_interface_metadata["s0_use_cluster_reconstruction"] = bool(
            s0_use_cluster_reconstruction
        )
        if use_source_cluster_ai_interface:
            ai_interface_metadata["source_cluster_count"] = len(
                ai_source_cluster_arrival_mw
            )
            ai_interface_metadata["source_cluster_ids"] = sorted(
                ai_source_cluster_arrival_mw
            )
            ai_interface_metadata["source_cluster_zone"] = ai_source_cluster_zone
            ai_interface_metadata["source_cluster_type"] = ai_source_cluster_type
            ai_interface_metadata["source_cluster_mean_mw"] = ai_source_cluster_mean_mw
            ai_interface_metadata["origin_reconstruction_weight_count"] = len(
                ai_origin_reconstruction_weight
            )
            ai_interface_metadata["destination_host_power_cap_gw"] = (
                ai_destination_host_power_cap_gw
            )
            ai_interface_metadata["destination_host_energy_cap_gwh_year"] = (
                ai_destination_host_energy_cap_gwh_year
            )
            ai_interface_metadata["s4_od_sets"] = {
                "source_clusters": ai_s4_od_params["source_clusters"],
                "destination_provinces": ai_s4_od_params[
                    "destination_provinces"
                ],
            }
            ai_interface_metadata["cluster_od_sets"] = ai_interface_metadata["s4_od_sets"]
            ai_interface_metadata["s4_od_parameter_audit"] = ai_s4_od_params["audit"]
            ai_interface_metadata["zone_adjacency"] = {
                "mode": ai_zone_neighbors_mode,
                "adjacency_source": adj_source,
                "zone_neighbors_full": {
                    z: sorted(v) for z, v in zone_neighbors_full.items()
                },
                "isolated_zones": list(isolated_zones),
                "adjacent_pair_count": int(adj_pair_count),
                "applied_zone_neighbors": (
                    None if zone_neighbors is None
                    else {z: sorted(v) for z, v in zone_neighbors.items()}
                ),
                "note": (
                    "mode=none → same-zone OD only (Run-NoMig baseline). "
                    "mode=adjacent → same+adjacent-zone OD (Run-FreeMig). "
                    "isolated_zones have <=1 neighbor; audit their OD reach."
                ),
            }
            ai_interface_metadata["cross_zone_od_audit"] = {
                "cross_zone_arc_count": int(
                    len(ai_s4_od_params.get("cross_zone_arcs", set()))
                ),
                "cross_zone_arcs": sorted(
                    [list(a) for a in ai_s4_od_params.get("cross_zone_arcs", set())]
                ),
                "zone_neighbors_mode": ai_zone_neighbors_mode,
                "note": (
                    "Non-zero only when AI_ZONE_NEIGHBORS=adjacent. "
                    "cross_zone_arcs are (source_cluster, destination) pairs "
                    "whose destination zone differs from the cluster zone."
                ),
            }
            ai_interface_metadata["cluster_od_parameter_audit"] = ai_s4_od_params["audit"]
            ai_interface_metadata["s4_od_cluster_zone"] = ai_s4_od_params[
                "cluster_zone"
            ]
            ai_interface_metadata["s4_od_cluster_type"] = ai_s4_od_params[
                "cluster_type"
            ]
            ai_interface_metadata["s4_od_readiness"] = (
                "source-cluster interface loaded; S4-OD variables and constraints "
                "are activated when AI_OD_FLOW is non-empty."
            )
            ai_interface_metadata["cluster_od_readiness"] = (
                "source-cluster interface loaded; S1/S4 OD variables and constraints "
                "are activated when AI_OD_FLOW is non-empty."
            )
            ai_interface_metadata["s0_cluster_reconstruction_audit"] = (
                s0_reconstruction_audit
            )
        ai_interface_metadata["ai_operational_scenario"] = ai_operational_scenario
        ai_s4_zone_runtime_transfer_active = (
            ai_s4_zone_runtime_transfer
            and ai_operational_scenario == "S4"
            and use_source_cluster_ai_interface
        )
        ai_interface_metadata["scenario_definitions"] = {
            "S0": (
                "Fixed AI: source-cluster flexible AI workloads are reconstructed "
                "back to their origin provinces and run according to the original "
                "hourly cluster profiles, with no spatial relocation or operational delay."
            ),
            "S1": (
                "Planning-only AI siting: source-cluster flexible AI workloads are "
                "allocated to destination provinces through OD planning variables, "
                "but run immediately according to their exogenous hourly cluster profiles "
                "without deadline-based operational shifting."
            ),
            "S4": (
                "Source-destination AI workload relocation: source-cluster workloads are "
                "allocated to destination provinces and executed through hourly batch-run "
                "variables under destination capacity, no-advance and deadline constraints."
            ),
            "S4_ZONE_RUNTIME_TRANSFER": (
                "S4 extension: long-term siting remains province-level, while the "
                "runtime subset may be re-routed between execution zones and then "
                "executed by provincial AI_BATCH_RUN within the deadline window."
            ),
        }
        ai_interface_metadata["province_zone"] = province_zone
        ai_interface_metadata["ai_batch_delay_hours"] = ai_batch_delay_hours
        ai_interface_metadata["s4_zone_runtime_transfer_config"] = {
            "enabled": bool(ai_s4_zone_runtime_transfer),
            "active": bool(ai_s4_zone_runtime_transfer_active),
            "route_mode": ai_rt_zone_route_mode,
            "env_enable": "AI_S4_ZONE_RUNTIME_TRANSFER",
            "env_route_mode": "AI_RT_ZONE_ROUTE_MODE",
        }
        ai_interface_metadata["legacy_ignored_ai_batch_power_cap_multiplier"] = (
            ai_batch_power_cap_multiplier
        )
        ai_interface_metadata["legacy_ignored_ai_batch_power_cap_multiplier_note"] = (
            "AI_BATCH_POWER_CAP_MULTIPLIER is retained only for CLI/script compatibility; "
            "S4-OD execution bounds use destination_power_cap_gw."
        )
        ai_interface_metadata["h2_scenario"] = h2_scenario
        ai_interface_metadata["h2_aboveground_allowed"] = h2_aboveground_allowed
        ai_interface_metadata["h2_underground_allowed"] = h2_underground_allowed
        if ai_operational_scenario == "S0":
            effective_ai_scenario = (
                "S0_CLUSTER_FIXED"
                if s0_use_cluster_reconstruction and use_source_cluster_ai_interface
                else "S0_PROVINCE_FIXED"
            )
        elif ai_operational_scenario == "S1":
            effective_ai_scenario = "S1_OD_PROFILED"
        elif ai_operational_scenario == "S4":
            effective_ai_scenario = (
                "S4_ZONE_RUNTIME_TRANSFER"
                if ai_s4_zone_runtime_transfer_active
                else "S4_OD"
            )
        else:
            raise RuntimeError(f"Unexpected AI scenario: {ai_operational_scenario}")
        ai_interface_metadata["effective_ai_scenario"] = effective_ai_scenario

        province_to_load_idx = {
            str(Params.Province[i]): load_index[i]
            for i in range(len(Params.Province))
        }
        for pro in Province:
            pro = str(pro)
            idx = province_to_load_idx.get(pro, -1)
            if idx != -1:
                base_load = np.asarray(load_demand[:, idx], dtype=float) * load_reshapping
                base_load = base_load[:HOURS]
            else:
                base_load = np.zeros(HOURS, dtype=float)

            fixed_series = np.asarray(
                fixed_ai_load.get(pro, np.zeros(HOURS)), dtype=float
            )[:HOURS]
            flex_series = np.asarray(
                flexible_ai_load.get(pro, np.zeros(HOURS)), dtype=float
            )[:HOURS]

            non_ai_base_load_mw[pro] = base_load
            non_ai_base_energy_mwh += float(np.sum(base_load))
            fixed_ai_energy_mwh += float(np.sum(fixed_series))
            flexible_ai_energy_mwh += float(np.sum(flex_series))

            # External AI is additional demand. Flexible AI is added here only for S0;
            # S1/S4 inject flexible AI through OD profiles or AI_BATCH_RUN.
            LOAD_DEMAND[pro] = base_load + fixed_series
            if ai_operational_scenario == "S0":
                if (
                    s0_use_cluster_reconstruction
                    and use_source_cluster_ai_interface
                    and s0_cluster_fixed_flexible_load_mw
                ):
                    s0_flex_series = np.asarray(
                        s0_cluster_fixed_flexible_load_mw.get(
                            pro, np.zeros(HOURS)
                        ),
                        dtype=float,
                    )[:HOURS]
                    LOAD_DEMAND[pro] = LOAD_DEMAND[pro] + s0_flex_series
                else:
                    LOAD_DEMAND[pro] = LOAD_DEMAND[pro] + flex_series
        ai_total_energy_mwh = fixed_ai_energy_mwh + flexible_ai_energy_mwh
        all_AI_load = ai_total_energy_mwh
        hourly_AI_load = 0.0 if ai_operational_scenario == "S0" else external_ai.flexible_pool_gw
        total_with_ai_energy_mwh = non_ai_base_energy_mwh + ai_total_energy_mwh
        ai_to_non_ai_base_ratio = (
            ai_total_energy_mwh / non_ai_base_energy_mwh
            if non_ai_base_energy_mwh > 0
            else None
        )
        ai_share_of_total_after_addition = (
            ai_total_energy_mwh / total_with_ai_energy_mwh
            if total_with_ai_energy_mwh > 0
            else None
        )
        flexible_share_of_ai = (
            flexible_ai_energy_mwh / ai_total_energy_mwh
            if ai_total_energy_mwh > 0
            else None
        )
        energy_accounting_mwh = {
            "units": "MWh over active model horizon",
            "hours": int(HOURS),
            "non_ai_base_energy_mwh": float(non_ai_base_energy_mwh),
            "fixed_ai_energy_mwh": float(fixed_ai_energy_mwh),
            "flexible_ai_energy_mwh": float(flexible_ai_energy_mwh),
            "total_ai_energy_mwh": float(ai_total_energy_mwh),
            "total_with_ai_energy_mwh": float(total_with_ai_energy_mwh),
            "ai_to_non_ai_base_ratio": (
                float(ai_to_non_ai_base_ratio)
                if ai_to_non_ai_base_ratio is not None
                else None
            ),
            "ai_share_of_total_after_addition": (
                float(ai_share_of_total_after_addition)
                if ai_share_of_total_after_addition is not None
                else None
            ),
            "flexible_share_of_ai": (
                float(flexible_share_of_ai)
                if flexible_share_of_ai is not None
                else None
            ),
            "legacy_ai_ratio_argument": float(_get_arg(args, "ai_ratio", 0.0)),
            "legacy_ai_ratio_used": False,
        }
        ai_interface_metadata["energy_accounting_mwh"] = energy_accounting_mwh
        ai_interface_metadata["ai_energy_accounting"] = {
            "definition": (
                "External AI load is modeled as additional electricity demand on top "
                "of the non-AI baseline load."
            ),
            "ai_scale_definition": "AI_base_ratio = E_AI / E_base",
            "ai_scale_interpretation": (
                "additional AI electricity demand equivalent to "
                "AI_base_ratio of non-AI baseline national electricity demand"
            ),
            "do_not_interpret_as": (
                "AI share of final total electricity demand unless using "
                "ai_share_of_total_after_addition = E_AI / (E_base + E_AI)."
            ),
            "note": (
                "args.ai_ratio is ignored when USE_EXTERNAL_AI_LOAD=true; it only "
                "applies to the legacy USE_EXTERNAL_AI_LOAD=false branch."
            ),
            **energy_accounting_mwh,
        }
        ai_interface_metadata["recommended_experiment_naming"] = (
            "aiBaseXXXX_migYYYY_profile, where aiBase is E_AI/E_base and "
            "mig is flexible/migratable share of total AI energy."
        )

        ai_target_base_ratio = os.environ.get("AI_TARGET_BASE_RATIO")
        ai_target_final_share = os.environ.get("AI_TARGET_FINAL_SHARE")
        ai_target_flexible_share = os.environ.get("AI_TARGET_FLEXIBLE_SHARE")
        ai_target_tol = float(os.environ.get("AI_TARGET_TOL", "0.002"))
        if ai_target_base_ratio not in {None, ""}:
            target = float(ai_target_base_ratio)
            actual = ai_to_non_ai_base_ratio
            if actual is None or abs(actual - target) > ai_target_tol:
                raise ValueError(
                    f"AI base-ratio mismatch: actual={actual}, target={target}, "
                    f"tol={ai_target_tol}. Check AI_LOAD_FILE/interface scale."
                )
        if ai_target_final_share not in {None, ""}:
            target = float(ai_target_final_share)
            actual = ai_share_of_total_after_addition
            if actual is None or abs(actual - target) > ai_target_tol:
                raise ValueError(
                    f"AI final-share mismatch: actual={actual}, target={target}, "
                    f"tol={ai_target_tol}. Check AI_LOAD_FILE/interface scale."
                )
        if ai_target_flexible_share not in {None, ""}:
            target = float(ai_target_flexible_share)
            actual = flexible_share_of_ai
            if actual is None or abs(actual - target) > ai_target_tol:
                raise ValueError(
                    f"AI flexible-share mismatch: actual={actual}, target={target}, "
                    f"tol={ai_target_tol}. Check migration/flexible-share setting."
                )
        ai_interface_metadata["ai_target_check"] = {
            "target_base_ratio": (
                float(ai_target_base_ratio)
                if ai_target_base_ratio not in {None, ""}
                else None
            ),
            "target_final_share": (
                float(ai_target_final_share)
                if ai_target_final_share not in {None, ""}
                else None
            ),
            "target_flexible_share": (
                float(ai_target_flexible_share)
                if ai_target_flexible_share not in {None, ""}
                else None
            ),
            "target_tolerance": float(ai_target_tol),
            "passed": True,
        }
        local_retention_min = float(os.environ.get("AI_LOCAL_RETENTION_MIN", "0.5"))
        max_host_share = float(os.environ.get("AI_MAX_HOST_SHARE", "0.2"))
        hub_max_host_share = float(os.environ.get("AI_HUB_MAX_HOST_SHARE", "0.3"))
        use_hub_relaxation = str_to_bool(
            os.environ.get("AI_USE_HUB_RELAXATION"), default=True
        )
        ai_interface_metadata["hub_provinces"] = sorted(AI_HUB_PROVINCES)
        interface_region_max_host_share = ai_interface_metadata.get(
            "region_max_host_share_from_interface", {}
        )
        ai_region_max_host_share = {}
        for key, env_name in {
            "northwest": "AI_NORTHWEST_MAX_HOST_SHARE",
            "southwest": "AI_SOUTHWEST_MAX_HOST_SHARE",
            "coastal_demand": "AI_COASTAL_DEMAND_MAX_HOST_SHARE",
        }.items():
            value = os.environ.get(env_name)
            if value in {None, ""}:
                value = interface_region_max_host_share.get(key)
            if value not in {None, ""}:
                ai_region_max_host_share[key] = float(value)
        ai_interface_metadata["legacy_province_hosting_bound_parameters"] = {
            "local_retention_min": float(local_retention_min),
            "max_host_share": float(max_host_share),
            "hub_max_host_share": float(hub_max_host_share),
            "use_hub_relaxation": bool(use_hub_relaxation),
            "region_max_host_share": ai_region_max_host_share,
            "note": (
                "These parameters are retained for legacy province-level hosting "
                "diagnostics. Main S1/S4-OD scenarios use source-cluster OD allocation "
                "with destination energy and power capacity constraints."
            ),
        }

        for pro in Province:
            pro = str(pro)
            flex_series = np.asarray(
                flexible_ai_load.get(pro, np.zeros(HOURS)), dtype=float
            )
            fixed_series = np.asarray(
                fixed_ai_load.get(pro, np.zeros(HOURS)), dtype=float
            )
            flex_gw = float(np.mean(flex_series)) / 1000.0
            fixed_gw = float(np.max(fixed_series)) / 1000.0
            capacity_headroom_gw = max(0.0, ai_hosting_capacity_gw.get(pro, 0.0) - fixed_gw)
            share = (
                hub_max_host_share
                if use_hub_relaxation and pro in AI_HUB_PROVINCES
                else max_host_share
            )
            province_local_retention_min = ai_local_retention_min.get(
                pro, local_retention_min
            )
            share = ai_max_host_share.get(pro, share)
            lower = min(capacity_headroom_gw, province_local_retention_min * flex_gw)
            share_cap = share * hourly_AI_load
            upper = min(capacity_headroom_gw, max(lower, share_cap))
            if pro in ai_hosting_upper_bound_gw:
                upper = min(upper, max(lower, ai_hosting_upper_bound_gw[pro]))
            ai_flexible_load_gw[pro] = flex_gw
            ai_hosting_lb_gw[pro] = lower
            ai_hosting_ub_gw[pro] = upper
            if ai_operational_scenario in {"S0", "S1"}:
                ai_batch_power_cap_gw[pro] = 0.0
            elif ai_operational_scenario == "S4":
                # Diagnostic only; S4-OD execution upper bounds use destination_power_cap_gw.
                ai_batch_power_cap_gw[pro] = capacity_headroom_gw
            else:
                raise RuntimeError(f"Unexpected AI scenario: {ai_operational_scenario}")

        lower_sum = sum(ai_hosting_lb_gw.values())
        upper_sum = sum(ai_hosting_ub_gw.values())
        ai_interface_metadata["hosting_lower_bound_sum_gw"] = lower_sum
        ai_interface_metadata["hosting_upper_bound_sum_gw"] = upper_sum
        ai_interface_metadata["batch_power_cap_sum_gw"] = sum(ai_batch_power_cap_gw.values())
        is_cluster_od_interface = (
            use_source_cluster_ai_interface
            and ai_operational_scenario in {"S1", "S4"}
            and bool(ai_s4_od_params)
        )
        is_s4_od_interface = is_cluster_od_interface and ai_operational_scenario == "S4"
        is_s1_od_interface = is_cluster_od_interface and ai_operational_scenario == "S1"
        if use_external_ai_load and ai_operational_scenario in {"S1", "S4"} and not is_cluster_od_interface:
            raise ValueError(
                f"{ai_operational_scenario} requires source-cluster AI interface. "
                "Province-level single-file AI load is not allowed for main S1/S4 experiments."
            )
        if is_s4_od_interface:
            ai_interface_metadata["batch_power_cap_convention"] = (
                "S4-OD AI_BATCH_RUN variable upper bounds use destination_power_cap_gw; "
                "batch_power_cap_sum_gw is retained as a legacy headroom diagnostic "
                "and is not binding."
            )
        elif is_s1_od_interface:
            ai_interface_metadata["batch_power_cap_convention"] = (
                "S1-OD does not use AI_BATCH_RUN; flexible workload enters power "
                "balance as a fixed destination hourly profile reconstructed from "
                "source-cluster OD allocation."
            )
        else:
            ai_interface_metadata["batch_power_cap_convention"] = (
                "No AI_BATCH_RUN variables are active for this scenario."
            )
        if is_cluster_od_interface:
            clusters_by_destination = ai_s4_od_params["clusters_by_destination"]
            active_destinations = sorted(
                str(pro)
                for pro in Province
                if clusters_by_destination.get(str(pro))
            )
            inactive_destinations = sorted(
                str(pro)
                for pro in Province
                if not clusters_by_destination.get(str(pro))
            )
            ai_interface_metadata["cluster_od_active_destination_audit"] = {
                "active_destination_count": int(len(active_destinations)),
                "inactive_destination_count": int(len(inactive_destinations)),
                "active_destinations": active_destinations,
                "inactive_destinations": inactive_destinations,
                "definition": (
                    "Active destinations have at least one allowed incoming source "
                    "cluster and are counted in OD capacity feasibility audits."
                ),
            }
            zone_cluster_pool_gw = {}
            for gid in ai_s4_od_params["source_clusters"]:
                zone = ai_s4_od_params["cluster_zone"][gid]
                zone_cluster_pool_gw[zone] = zone_cluster_pool_gw.get(zone, 0.0) + float(
                    ai_s4_od_params["cluster_mean_gw"][gid]
                )
            zone_destination_energy_cap_gw = {}
            for pro in active_destinations:
                zone = province_zone[pro]
                cap_gw = (
                    float(ai_s4_od_params["destination_energy_cap_gwh_year"][pro])
                    / 8760.0
                )
                zone_destination_energy_cap_gw[zone] = (
                    zone_destination_energy_cap_gw.get(zone, 0.0) + cap_gw
                )
            zone_energy_feasibility = {}
            for zone, pool_gw in sorted(zone_cluster_pool_gw.items()):
                cap_gw = zone_destination_energy_cap_gw.get(zone, 0.0)
                zone_energy_feasibility[zone] = {
                    "source_cluster_pool_gw": float(pool_gw),
                    "destination_energy_cap_avg_gw": float(cap_gw),
                    "margin_gw": float(cap_gw - pool_gw),
                    "active_destination_only": True,
                }
                if cap_gw + 1e-6 < pool_gw:
                    raise ValueError(
                        f"{ai_operational_scenario}-OD zone {zone} destination energy caps cannot absorb "
                        f"source pool: {cap_gw} < {pool_gw}"
                    )
            od_energy_avg_cap_sum = sum(
                float(ai_s4_od_params["destination_energy_cap_gwh_year"][str(pro)])
                / 8760.0
                for pro in Province
            )
            od_power_cap_sum = sum(
                float(ai_s4_od_params["destination_power_cap_gw"][str(pro)])
                for pro in Province
            )
            od_active_energy_avg_cap_sum = sum(
                float(ai_s4_od_params["destination_energy_cap_gwh_year"][pro])
                / 8760.0
                for pro in active_destinations
            )
            od_active_power_cap_sum = sum(
                float(ai_s4_od_params["destination_power_cap_gw"][pro])
                for pro in active_destinations
            )
            ai_interface_metadata["cluster_od_all_destination_energy_avg_cap_sum_gw"] = (
                od_energy_avg_cap_sum
            )
            ai_interface_metadata["cluster_od_all_destination_power_cap_sum_gw"] = (
                od_power_cap_sum
            )
            ai_interface_metadata["cluster_od_active_destination_energy_avg_cap_sum_gw"] = (
                od_active_energy_avg_cap_sum
            )
            ai_interface_metadata["cluster_od_active_destination_power_cap_sum_gw"] = (
                od_active_power_cap_sum
            )
            ai_interface_metadata["cluster_od_destination_energy_avg_cap_sum_gw"] = (
                od_active_energy_avg_cap_sum
            )
            ai_interface_metadata["cluster_od_destination_power_cap_sum_gw"] = (
                od_active_power_cap_sum
            )
            ai_interface_metadata["cluster_od_zone_energy_feasibility"] = (
                zone_energy_feasibility
            )
            destination_host_class_share_cap = AI_SCENARIO_DEST_CAPS[
                ai_host_cap_scenario
            ]
            if ai_host_cap_scenario == "legacy_tier":
                legacy_class_share_cap = {
                    "A": ai_dest_tier3_share_cap,
                    "B": ai_dest_tier3_share_cap,
                    "C": ai_dest_tier2_share_cap,
                    "D": ai_dest_tier2_share_cap,
                    "E": ai_dest_tier1_share_cap,
                }
                destination_host_class_share_cap = legacy_class_share_cap
            zone_destination_hosting_share_cap_gw = {}
            for pro in active_destinations:
                zone = province_zone[pro]
                host_class = str(
                    ai_s4_od_params["destination_hosting_class_by_province"].get(
                        pro, _default_ai_hosting_class(pro)
                    )
                )
                share_cap = float(
                    destination_host_class_share_cap.get(host_class, 0.0)
                )
                cap_gw = share_cap * float(hourly_AI_load)
                zone_destination_hosting_share_cap_gw[zone] = (
                    zone_destination_hosting_share_cap_gw.get(zone, 0.0) + cap_gw
                )
            zone_hosting_share_feasibility = {}
            for zone, pool_gw in sorted(zone_cluster_pool_gw.items()):
                cap_gw = zone_destination_hosting_share_cap_gw.get(zone, 0.0)
                zone_hosting_share_feasibility[zone] = {
                    "source_cluster_pool_gw": float(pool_gw),
                    "destination_hosting_share_cap_gw": float(cap_gw),
                    "margin_gw": float(cap_gw - pool_gw),
                    "active_destination_only": True,
                }
                if (
                    use_network_tier_caps
                    and ai_host_cap_scenario != "legacy_tier"
                    and cap_gw + 1e-6 < pool_gw
                ):
                    raise ValueError(
                        f"{ai_operational_scenario}-OD zone {zone} destination hosting "
                        f"share caps cannot absorb source pool: cap={cap_gw}, "
                        f"pool={pool_gw}. Relax AI_HOST_CAP_SCENARIO or "
                        "destination hosting-class assignment."
                    )
            zone_network_share_feasibility = zone_hosting_share_feasibility
            ai_interface_metadata["cluster_od_zone_hosting_share_feasibility"] = (
                zone_hosting_share_feasibility
            )
            ai_interface_metadata["cluster_od_zone_network_share_feasibility"] = (
                zone_network_share_feasibility
            )
            if is_s4_od_interface:
                ai_interface_metadata["s4_od_zone_hosting_share_feasibility"] = (
                    zone_hosting_share_feasibility
                )
                ai_interface_metadata["s4_od_zone_network_share_feasibility"] = (
                    zone_network_share_feasibility
                )
            elif is_s1_od_interface:
                ai_interface_metadata["s1_od_zone_hosting_share_feasibility"] = (
                    zone_hosting_share_feasibility
                )
                ai_interface_metadata["s1_od_zone_network_share_feasibility"] = (
                    zone_network_share_feasibility
                )
            destination_combined_cap_audit = {}
            zone_combined_effective_cap_gw = {}
            for pro in active_destinations:
                zone = province_zone[pro]
                energy_cap_gw = (
                    float(
                        ai_s4_od_params["destination_energy_cap_gwh_year"].get(
                            pro, 0.0
                        )
                    )
                    / 8760.0
                )
                tier = int(
                    ai_s4_od_params["destination_network_tier"].get(pro, 1)
                )
                host_class = str(
                    ai_s4_od_params["destination_hosting_class_by_province"].get(
                        pro, _default_ai_hosting_class(pro)
                    )
                )
                share_cap = float(
                    destination_host_class_share_cap.get(host_class, 0.0)
                )
                hosting_share_cap_gw = share_cap * float(hourly_AI_load)
                effective_cap_gw = (
                    min(energy_cap_gw, hosting_share_cap_gw)
                    if use_network_tier_caps
                    else energy_cap_gw
                )
                destination_combined_cap_audit[pro] = {
                    "zone": zone,
                    "network_tier": int(tier),
                    "hosting_class": host_class,
                    "energy_cap_gw": float(energy_cap_gw),
                    "destination_hosting_share_cap": float(share_cap),
                    "hosting_share_cap_gw": float(hosting_share_cap_gw),
                    "destination_share_cap": float(share_cap),
                    "network_share_cap_gw": float(hosting_share_cap_gw),
                    "effective_cap_gw": float(effective_cap_gw),
                    "active_destination": True,
                    "binding_cap": (
                        "energy_cap"
                        if (not use_network_tier_caps)
                        or energy_cap_gw <= hosting_share_cap_gw
                        else "hosting_share_cap"
                    ),
                }
                zone_combined_effective_cap_gw[zone] = (
                    zone_combined_effective_cap_gw.get(zone, 0.0)
                    + effective_cap_gw
                )
            zone_combined_cap_feasibility = {}
            for zone, pool_gw in sorted(zone_cluster_pool_gw.items()):
                cap_gw = zone_combined_effective_cap_gw.get(zone, 0.0)
                zone_combined_cap_feasibility[zone] = {
                    "source_cluster_pool_gw": float(pool_gw),
                    "combined_effective_destination_cap_gw": float(cap_gw),
                    "margin_gw": float(cap_gw - pool_gw),
                    "active_destination_only": True,
                    "definition": (
                        "sum_d min(destination_energy_cap_gwh_year[d] / 8760, "
                        "destination_share_cap[d] * total_flexible_ai_gw)"
                    ),
                }
                if use_network_tier_caps and cap_gw + 1e-6 < pool_gw:
                    raise ValueError(
                        f"{ai_operational_scenario}-OD zone {zone} combined destination "
                        f"caps cannot absorb source pool: cap={cap_gw}, pool={pool_gw}. "
                        "Relax destination energy caps, AI_DEST_TIER*_SHARE_CAP, "
                        "or destination tier assignment."
                    )
            ai_interface_metadata["cluster_od_destination_combined_cap_audit"] = (
                destination_combined_cap_audit
            )
            ai_interface_metadata["cluster_od_zone_combined_cap_feasibility"] = (
                zone_combined_cap_feasibility
            )
            ai_interface_metadata["cluster_od_zone_active_combined_cap_feasibility"] = (
                zone_combined_cap_feasibility
            )
            if is_s4_od_interface:
                ai_interface_metadata["s4_od_destination_combined_cap_audit"] = (
                    destination_combined_cap_audit
                )
                ai_interface_metadata["s4_od_zone_combined_cap_feasibility"] = (
                    zone_combined_cap_feasibility
                )
                ai_interface_metadata["s4_od_zone_active_combined_cap_feasibility"] = (
                    zone_combined_cap_feasibility
                )
            elif is_s1_od_interface:
                ai_interface_metadata["s1_od_destination_combined_cap_audit"] = (
                    destination_combined_cap_audit
                )
                ai_interface_metadata["s1_od_zone_combined_cap_feasibility"] = (
                    zone_combined_cap_feasibility
                )
                ai_interface_metadata["s1_od_zone_active_combined_cap_feasibility"] = (
                    zone_combined_cap_feasibility
                )
            if od_active_energy_avg_cap_sum + 1e-6 < hourly_AI_load:
                raise ValueError(
                    f"{ai_operational_scenario}-OD active destination energy caps cannot absorb flexible pool: "
                    f"{od_active_energy_avg_cap_sum} < {hourly_AI_load}"
                )
            if od_active_power_cap_sum <= 0:
                raise ValueError(
                    f"{ai_operational_scenario}-OD needs positive active destination power capacity."
                )
            fixed_peak_power_cap_audit = {}
            for pro in Province:
                pro = str(pro)
                fixed_peak_gw = float(
                    np.max(fixed_ai_load.get(pro, np.zeros(HOURS)))
                ) / 1000.0
                cap_gw = float(ai_s4_od_params["destination_power_cap_gw"][pro])
                fixed_peak_power_cap_audit[pro] = {
                    "fixed_peak_gw": fixed_peak_gw,
                    "destination_power_cap_gw": cap_gw,
                    "margin_gw": cap_gw - fixed_peak_gw,
                }
                if fixed_peak_gw > cap_gw + 1e-6:
                    raise ValueError(
                        f"S4-OD destination cap infeasible for {pro}: "
                        f"fixed_peak={fixed_peak_gw}, cap={cap_gw}"
                    )
            ai_interface_metadata["cluster_od_fixed_peak_power_cap_audit"] = (
                fixed_peak_power_cap_audit
            )
            if is_s4_od_interface:
                ai_interface_metadata["s4_od_all_destination_energy_avg_cap_sum_gw"] = (
                    od_energy_avg_cap_sum
                )
                ai_interface_metadata["s4_od_all_destination_power_cap_sum_gw"] = (
                    od_power_cap_sum
                )
                ai_interface_metadata["s4_od_active_destination_energy_avg_cap_sum_gw"] = (
                    od_active_energy_avg_cap_sum
                )
                ai_interface_metadata["s4_od_active_destination_power_cap_sum_gw"] = (
                    od_active_power_cap_sum
                )
                ai_interface_metadata["s4_od_destination_energy_avg_cap_sum_gw"] = (
                    od_active_energy_avg_cap_sum
                )
                ai_interface_metadata["s4_od_destination_power_cap_sum_gw"] = (
                    od_active_power_cap_sum
                )
                ai_interface_metadata["s4_od_zone_energy_feasibility"] = (
                    zone_energy_feasibility
                )
                ai_interface_metadata["s4_od_fixed_peak_power_cap_audit"] = (
                    fixed_peak_power_cap_audit
                )
            elif is_s1_od_interface:
                ai_interface_metadata["s1_od_all_destination_energy_avg_cap_sum_gw"] = (
                    od_energy_avg_cap_sum
                )
                ai_interface_metadata["s1_od_all_destination_power_cap_sum_gw"] = (
                    od_power_cap_sum
                )
                ai_interface_metadata["s1_od_active_destination_energy_avg_cap_sum_gw"] = (
                    od_active_energy_avg_cap_sum
                )
                ai_interface_metadata["s1_od_active_destination_power_cap_sum_gw"] = (
                    od_active_power_cap_sum
                )
                ai_interface_metadata["s1_od_destination_energy_avg_cap_sum_gw"] = (
                    od_active_energy_avg_cap_sum
                )
                ai_interface_metadata["s1_od_destination_power_cap_sum_gw"] = (
                    od_active_power_cap_sum
                )
                ai_interface_metadata["s1_od_zone_energy_feasibility"] = (
                    zone_energy_feasibility
                )
                ai_interface_metadata["s1_od_fixed_peak_power_cap_audit"] = (
                    fixed_peak_power_cap_audit
                )
        print("Using external AI load interface:", json.dumps(ai_interface_metadata, ensure_ascii=False))
    else:
        for i in range(len(Params.Province)):
            if load_index[i] != -1:
                # LOAD_DEMAND[f'{Params.Province[i]}'] = load_demand[:,load_index[i]]*load_reshapping
                traditional_load = (
                    load_demand[:, load_index[i]] * load_reshapping * (1 - args.ai_ratio)
                )
                ai_baseload_val = (
                    np.mean(traditional_load) / (1 - args.ai_ratio) * args.ai_ratio
                )
                ai_load = np.full_like(traditional_load, ai_baseload_val)
                all_AI_load += np.sum(ai_load)

                # LOAD_DEMAND[f"{Params.Province[i]}"] = traditional_load + ai_load
                LOAD_DEMAND[f"{Params.Province[i]}"] = (
                    traditional_load + (1 - args.AI_load_shift) * ai_load
                )

            else:
                LOAD_DEMAND[f"{Params.Province[i]}"] = np.zeros_like(load_demand[:, 0])
        hourly_AI_load = all_AI_load * args.AI_load_shift / 8760/1000
    underground_pro = get_pro_underground(Params)
    print(underground_pro)

    # load_demnad_单位为MW
    total_load_demand = sum(np.sum(values) for values in LOAD_DEMAND.values())
    if use_external_ai_load:
        print(f"Non-AI baseline electricity: {non_ai_base_energy_mwh / 1e9:.4f} PWh")
        print(f"Fixed AI electricity: {fixed_ai_energy_mwh / 1e9:.4f} PWh")
        print(f"Flexible AI electricity: {flexible_ai_energy_mwh / 1e9:.4f} PWh")
        print(f"Total AI electricity: {ai_total_energy_mwh / 1e9:.4f} PWh")
        print(f"Total electricity after adding AI: {total_with_ai_energy_mwh / 1e9:.4f} PWh")
        print(f"AI / non-AI baseline ratio: {ai_to_non_ai_base_ratio:.4%}")
        print(f"AI share after addition: {ai_share_of_total_after_addition:.4%}")
        print(f"Flexible share of AI: {flexible_share_of_ai:.4%}")
        print(f"Flexible AI pool: {hourly_AI_load:.4f} GW")
        print(
            "Power-balance base RHS electricity "
            f"(base + fixed AI, plus flexible AI only in S0): {np.sum(total_load_demand) / 1e9:.4f} PWh"
        )
    else:
        print(f"Legacy AI ratio argument: {args.ai_ratio}")
        print(all_AI_load / 1e9, "legacy total AI electricity (PWh)")
        print(hourly_AI_load, "legacy flexible AI pool (GW)")
        print(f"Legacy power-balance RHS electricity: {np.sum(total_load_demand) / 1e9:.4f} PWh")


    with open(os.path.join(path, "pv_lcoe.pkl"), "rb") as file:
        pv_lcoe = pickle.load(file)
    with open(os.path.join(path, "pv_cf.pkl"), "rb") as file:
        pv_cf = pickle.load(file)
    with open(os.path.join(path, "pv_cell.pkl"), "rb") as file:
        pv_cell = pickle.load(file)
    with open(os.path.join(path, "pv_cap.pkl"), "rb") as file:
        pv_cap = pickle.load(file)

    with open(os.path.join(path, "wind_cap.pkl"), "rb") as file:
        wind_cap = pickle.load(file)
    with open(os.path.join(path, "wind_cell.pkl"), "rb") as file:
        wind_cell = pickle.load(file)
    with open(os.path.join(path, "wind_cf.pkl"), "rb") as file:
        wind_cf = pickle.load(file)
    with open(os.path.join(path, "wind_lcoe.pkl"), "rb") as file:
        wind_lcoe = pickle.load(file)


    for pro in range(len(offwind_pro)):
        wind_cap[offwind_pro[pro]].append(offwind_loce[pro, 0])
        wind_lcoe[offwind_pro[pro]].append(offwind_loce[pro, 5])
        wind_cf[offwind_pro[pro]].append(offwind_cf[:, pro])
        try:
            wind_cell[offwind_pro[pro]] += 1
        except:
            wind_cell[f"{offwind_pro[pro]}"] = 1

    pv_cf = set_small_values_to_zero(pv_cf)
    wind_cf = set_small_values_to_zero(wind_cf)
    pv_gen = 0
    for pro in Province:
        _pro_gen = 0
        for c in range(int(pv_cell[pro])):
            pv_gen += sum(pv_cf[pro][c]) * pv_cap[pro][c]
            _pro_gen += pv_cap[pro][c]
    print(pv_gen / 1000 / 1000 / 1000 / 1000)
    print("总的光伏发电量：", pv_gen / 1000 / 1000 / 1000 / 1000, "Pwh")
    wind_gen = 0
    for pro in Province:
        for c in range(int(wind_cell[pro])):
            wind_gen += sum(wind_cf[pro][c]) * wind_cap[pro][c]
    print("总的风电发电量：", wind_gen / 1000 / 1000 / 1000 / 1000, "Pwh")
    # KW -> GW
    wind_cap = {
        key: [x / 1000 / 1000 for x in value_list]
        for key, value_list in wind_cap.items()
    }
    print(
        f"可以安装的风电功率：{sum( [ sum(v) for v in wind_cap.values()])}GW  ---->{sum( [ sum(v)/1000 for v in wind_cap.values()])}TW"
    )
    pv_cap = {
        key: [x / 1000 / 1000 for x in value_list] for key, value_list in pv_cap.items()
    }
    print(
        f"可以安装的光伏功率：{sum( [ sum(v) for v in pv_cap.values()])}GW  ---->{sum( [ sum(v)/1000 for v in pv_cap.values()])}TW"
    )
    installed_cap_data = {}
    install_cap = pd.read_csv(os.path.join(path, "installed_cap.csv"))
    numeric_cols = install_cap.select_dtypes(include=["number"]).columns
    install_cap[numeric_cols] = install_cap[numeric_cols]
    installl_Hydro = install_cap["Hydro"] = (
        install_cap["Hydro"] * hydro_new_install_cap_conf
    )
    installl_Coal = install_cap["Coal"] = install_cap["Coal"]
    installl_Wind = install_cap["Wind"] = install_cap["Wind"]
    installl_Solar = install_cap["Solar"] = install_cap["Solar"]
    installl_GAS = install_cap["GAS"] = install_cap["GAS"] * other_new_installed_cap
    installl_Nuclear = install_cap["Nuclear"] = (
        install_cap["Nuclear"] * nuclear_new_install_cap_conf
    )
    installl_BECCS = install_cap["BECCS"] = (
        install_cap["BECCS"] * other_new_installed_cap
    )

    print(
        f"已安装的Coal装机容量：{sum(installl_Coal)}GW ----> {sum(installl_Coal)/1000}TW"
    )
    print(
        f"已安装的Hydro装机容量：{sum(installl_Hydro)}GW ----> {sum(installl_Hydro)/1000}TW"
    )
    print(
        f"已安装的Wind 装机容量：{sum(installl_Wind)}GW ----> {sum(installl_Wind)/1000}TW"
    )
    print(
        f"已安装的Solar装机容量：{sum(installl_Solar)}GW ----> {sum(installl_Solar)/1000}TW"
    )
    print(
        f"已安装的GAS装机容量：{sum(installl_GAS)}GW ----> {sum(installl_GAS)/1000}TW"
    )
    print(
        f"已安装的Nuclear装机容量：{sum(installl_Nuclear)}GW ----> {sum(installl_Nuclear)/1000}TW"
    )
    print(
        f"已安装的BECCS装机容量：{sum(installl_BECCS)}GW ----> {sum(installl_BECCS)/1000}TW"
    )

    install_cap_hydro = install_cap.set_index("Province")["Hydro"].to_dict()
    install_cap_coal = install_cap.set_index("Province")["Coal"].to_dict()
    install_cap_Wind = install_cap.set_index("Province")["Wind"].to_dict()
    install_cap_Solar = install_cap.set_index("Province")["Solar"].to_dict()
    install_cap_Other = install_cap.set_index("Province")["GAS"].to_dict()
    install_cap_Nuclear = install_cap.set_index("Province")["Nuclear"].to_dict()
    install_cap_Bios = install_cap.set_index("Province")["BECCS"].to_dict()

    default_hydro_cf_max = float(os.environ.get("HYDRO_CF_MAX", "0.5"))
    hydro_cf_max_file = os.environ.get("HYDRO_CF_MAX_FILE", "")
    hydro_cf_max_by_province = {str(pro): default_hydro_cf_max for pro in Province}
    if hydro_cf_max_file:
        hydro_cf_df = pd.read_csv(hydro_cf_max_file)
        required_cols = {"Province", "Hydro_CF_Max"}
        missing_cols = required_cols - set(hydro_cf_df.columns)
        if missing_cols:
            raise ValueError(
                f"HYDRO_CF_MAX_FILE missing columns: {sorted(missing_cols)}"
            )
        for _, row in hydro_cf_df.iterrows():
            pro = str(row["Province"])
            if pro in hydro_cf_max_by_province:
                hydro_cf_max_by_province[pro] = float(row["Hydro_CF_Max"])
    for pro, cf in hydro_cf_max_by_province.items():
        if cf < 0 or cf > 1:
            raise ValueError(
                f"Hydro CF max for province {pro} must be in [0, 1], got {cf}"
            )
    hydro_cf_policy = {
        "enabled": True,
        "default_hydro_cf_max": float(default_hydro_cf_max),
        "hydro_cf_max_file": hydro_cf_max_file or None,
        "hydro_cf_max_by_province": {
            str(pro): float(hydro_cf_max_by_province[str(pro)]) for pro in Province
        },
        "constraint": (
            "sum_h HydroGeneration[p,h] <= HydroCapacity[p] * 8760 * "
            "HORIZON_SCALE * Hydro_CF_Max[p]"
        ),
        "hydro_generation_definition": (
            "load_conv[Hydro] + trans_out[Hydro] + charge_phs[Hydro] "
            "+ charge_bat[Hydro] + charge_h2[Hydro]"
        ),
        "unit": "GWh over active model horizon",
    }
    ai_interface_metadata["hydro_cf_policy"] = hydro_cf_policy
    print("Hydro annual CF max policy:", json.dumps(hydro_cf_policy, ensure_ascii=False))

    installed_cap_data["Coal"] = install_cap_coal
    installed_cap_data["Hydro"] = install_cap_hydro
    installed_cap_data["GAS"] = install_cap_Other
    installed_cap_data["Nuclear"] = install_cap_Nuclear
    installed_cap_data["BECCS"] = install_cap_Bios



    ru_c = Params.ru_c
    rd_c = Params.rd_c
    resv_p = Params.resv_p
    ru_conf = Params.ru_conf
    rd_conf = Params.rd_conf

    print("start model")
    model = gp.Model("lp_energy_system")

    # 获取环境变量中的 GUROBI_THREADS，如果未设置则默认为 24
    gurobi_threads = int(os.environ.get("GUROBI_THREADS", 24))
    print(f"Setting Gurobi Threads to: {gurobi_threads}")
    model.setParam("Threads", gurobi_threads)
    # model.setParam("Method", 2)  # 强制使用 Barrier
    # model.setParam("ScaleFlag", 2)  # 强制使用 Barrier
    # # model.setParam("Crossover", 0)  # 如果不需要精确顶点解，关闭 Crossover 可大幅提速
    # model.setParam("NumericFocus", 2)
    # model.setParam('BarHomogeneous', 1)
    # model.setParam("Presolve", 2)

    # model.setParam("BarConvTol", 1e-4)

    # # LP feasibility / optimality tolerance
    # model.setParam("FeasibilityTol", 1e-5)
    # model.setParam("OptimalityTol", 1e-5)


    model.setParam("Method", 2)
    model.setParam("Crossover", 0)
    model.setParam("BarHomogeneous", int(os.environ.get("GUROBI_BAR_HOMOGENEOUS", "0")))
    model.setParam("NumericFocus", int(os.environ.get("GUROBI_NUMERIC_FOCUS", "1")))
    model.setParam("ScaleFlag", int(os.environ.get("GUROBI_SCALE_FLAG", "2")))
    model.setParam("Presolve", int(os.environ.get("GUROBI_PRESOLVE", "2")))
    model.setParam("PreSparsify", int(os.environ.get("GUROBI_PRESPARSIFY", "1")))
    model.setParam("Aggregate", int(os.environ.get("GUROBI_AGGREGATE", "2")))
    model.setParam("BarConvTol", float(os.environ.get("GUROBI_BARCONVTOL", "1e-5")))
    model.setParam("FeasibilityTol", float(os.environ.get("GUROBI_FEASTOL", "1e-6")))
    model.setParam("OptimalityTol", float(os.environ.get("GUROBI_OPTTOL", "1e-6")))
    if os.environ.get("GUROBI_TIME_LIMIT"):
        model.setParam("TimeLimit", float(os.environ["GUROBI_TIME_LIMIT"]))

    print("LP / Barrier parameter summary:")
    print(f"  Method         = {model.Params.Method}")
    print(f"  Crossover      = {model.Params.Crossover}")
    print(f"  BarHomogeneous = {model.Params.BarHomogeneous}")
    print(f"  NumericFocus   = {model.Params.NumericFocus}")
    print(f"  ScaleFlag      = {model.Params.ScaleFlag}")
    print(f"  Presolve       = {model.Params.Presolve}")
    print(f"  PreSparsify    = {model.Params.PreSparsify}")
    print(f"  Aggregate      = {model.Params.Aggregate}")
    print(f"  BarConvTol     = {model.Params.BarConvTol}")
    print(f"  FeasibilityTol = {model.Params.FeasibilityTol}")
    print(f"  OptimalityTol  = {model.Params.OptimalityTol}")






    ru = {"Coal": {}, "Hydro": {}, "Nuclear": {}, "BECCS": {}, "GAS": {}}
    rd = {"Coal": {}, "Hydro": {}, "Nuclear": {}, "BECCS": {}, "GAS": {}}
    trans_out = {
        "Coal": {},
        "Hydro": {},
        "Nuclear": {},
        "Wind": {},
        "Solar": {},
        "PHS": {},
        "BAT": {},
        "H2": {},
    }
    load_conv = {"Coal": {}, "Hydro": {}, "Nuclear": {}, "BECCS": {}, "GAS": {}}

    trans_pair_in_DC_installed = []
    trans_pair_in_AC_installed = []
    trans_pair_in_DC = []
    trans_pair_in_AC = []

    trans_pair_out_DC_installed = []
    trans_pair_out_AC_installed = []
    trans_pair_out_DC = []
    trans_pair_out_AC = []

    load_trans_DC_installed = {}
    load_trans_AC_installed = {}
    load_trans_DC = {}
    load_trans_AC = {}

    trans_cap_DC = {}
    trans_cap_AC = {}
    trans_cap_DC_installed_expansion = {}
    trans_cap_AC_installed_expansion = {}
    allow_installed_trans_expansion = bool(
        getattr(args, "allow_installed_trans_expansion", False)
    )
    installed_trans_cap_ub = float(getattr(args, "installed_trans_cap_ub", 0.0))
    installed_trans_final_cap_ub = float(
        getattr(args, "installed_trans_final_cap_ub", 50.0)
    )
    transmission_utilization_limit = float(
        getattr(args, "transmission_utilization_limit", 1.0)
    )
    if not (0 < transmission_utilization_limit <= 1.0):
        raise ValueError(
            "transmission_utilization_limit must be in (0, 1], got "
            f"{transmission_utilization_limit}"
        )

    def installed_expansion_ub(family, pair):
        if not allow_installed_trans_expansion:
            return 0
        base_cap = float(trans_data[family]["cap"][pair])
        return max(0, min(installed_trans_cap_ub, installed_trans_final_cap_ub - base_cap))

    def installed_flow_ub(family, pair):
        base_cap = float(trans_data[family]["cap"].get(pair, 0.0))
        if allow_installed_trans_expansion:
            return max(base_cap, installed_trans_final_cap_ub) * transmission_utilization_limit
        return base_cap * transmission_utilization_limit

    for pair in trans_data["all_pair"]:
        pair2 = (pair[1], pair[0])
        if pair in trans_data["DC_installed"]["pair"]:
            trans_pair_in_DC_installed.append(pair[1])
            trans_pair_out_DC_installed.append(pair[0])
            trans_cap_DC_installed_expansion[pair] = model.addVar(
                lb=0,
                ub=installed_expansion_ub("DC_installed", pair),
                vtype=GRB.CONTINUOUS,
            )
            if args.TC:
                load_trans_DC_installed[pair] = model.addVars(
                    HOURS,
                    lb=0,
                    ub=installed_flow_ub("DC_installed", pair),
                    vtype=GRB.CONTINUOUS,
                )
            else:
                load_trans_DC_installed[pair] = model.addVars(
                    HOURS,
                    lb=0,
                    ub=installed_flow_ub("DC_installed", pair),
                    vtype=GRB.CONTINUOUS,
                )

        else:

            trans_cap_DC_installed_expansion[pair] = model.addVar(
                lb=0, ub=0, vtype=GRB.CONTINUOUS
            )
            load_trans_DC_installed[pair] = model.addVars(
                HOURS, lb=0, ub=0, vtype=GRB.CONTINUOUS
            )
            if (
                pair2 in trans_data["all_pair_AC"]
                and pair2 not in trans_data["all_pair"]
            ):
                load_trans_DC_installed[pair2] = model.addVars(
                    HOURS, lb=0, ub=0, vtype=GRB.CONTINUOUS
                )
        if pair in trans_data["AC_installed"]["pair"]:
            trans_pair_in_AC_installed.append(pair[1])
            trans_pair_in_AC_installed.append(pair[0])
            trans_pair_out_AC_installed.append(pair[0])
            trans_pair_out_AC_installed.append(pair[1])
            trans_cap_AC_installed_expansion[pair] = model.addVar(
                lb=0,
                ub=installed_expansion_ub("AC_installed", pair),
                vtype=GRB.CONTINUOUS,
            )

            if args.TC:

                load_trans_AC_installed[pair] = model.addVars(
                    HOURS,
                    lb=0,
                    ub=installed_flow_ub("AC_installed", pair),
                    vtype=GRB.CONTINUOUS,
                )
                load_trans_AC_installed[pair2] = model.addVars(
                    HOURS,
                    lb=0,
                    ub=installed_flow_ub("AC_installed", pair),
                    vtype=GRB.CONTINUOUS,
                )

            else:
                load_trans_AC_installed[pair] = model.addVars(
                    HOURS,
                    lb=0,
                    ub=installed_flow_ub("AC_installed", pair),
                    vtype=GRB.CONTINUOUS,
                )
                load_trans_AC_installed[pair2] = model.addVars(
                    HOURS,
                    lb=0,
                    ub=installed_flow_ub("AC_installed", pair),
                    vtype=GRB.CONTINUOUS,
                )

        else:
            trans_cap_AC_installed_expansion[pair] = model.addVar(
                lb=0, ub=0, vtype=GRB.CONTINUOUS
            )
            load_trans_AC_installed[pair] = model.addVars(
                HOURS, lb=0, ub=0, vtype=GRB.CONTINUOUS
            )
            load_trans_AC_installed[pair2] = model.addVars(
                HOURS, lb=0, ub=0, vtype=GRB.CONTINUOUS
            )

        if pair in trans_data["DC"]["pair"]:
            trans_pair_in_DC.append(pair[1])
            trans_pair_out_DC.append(pair[0])
            if args.TC:
                load_trans_DC[pair] = model.addVars(
                    HOURS, lb=0, ub=0, vtype=GRB.CONTINUOUS
                )
            else:
                load_trans_DC[pair] = model.addVars(
                    HOURS, lb=0, ub=0, vtype=GRB.CONTINUOUS
                )
            trans_cap_DC[pair] = model.addVar(lb=0, ub=0, vtype=GRB.CONTINUOUS)
        else:
            load_trans_DC[pair] = model.addVars(HOURS, lb=0, ub=0, vtype=GRB.CONTINUOUS)
            trans_cap_DC[pair] = model.addVar(lb=0, ub=0, vtype=GRB.CONTINUOUS)
            if (
                pair2 in trans_data["all_pair_AC"]
                and pair2 not in trans_data["all_pair"]
            ):
                load_trans_DC[pair2] = model.addVars(
                    HOURS, lb=0, ub=0, vtype=GRB.CONTINUOUS
                )

        if pair in trans_data["AC"]["pair"]:
            trans_pair_in_AC.append(pair[1])
            trans_pair_in_AC.append(pair[0])

            trans_pair_out_AC.append(pair[0])
            trans_pair_out_AC.append(pair[1])

            if args.TC:
                load_trans_AC[pair] = model.addVars(
                    HOURS, lb=0, ub=0, vtype=GRB.CONTINUOUS
                )
                load_trans_AC[pair2] = model.addVars(
                    HOURS, lb=0, ub=0, vtype=GRB.CONTINUOUS
                )

            else:
                load_trans_AC[pair] = model.addVars(
                    HOURS, lb=0, ub=0, vtype=GRB.CONTINUOUS
                )
                load_trans_AC[pair2] = model.addVars(
                    HOURS, lb=0, ub=0, vtype=GRB.CONTINUOUS
                )

            trans_cap_AC[pair] = model.addVar(lb=0, ub=0, vtype=GRB.CONTINUOUS)
            # print('1 AC')
        else:
            # print('2 AC')
            load_trans_AC[pair] = model.addVars(HOURS, lb=0, ub=0, vtype=GRB.CONTINUOUS)
            load_trans_AC[pair2] = model.addVars(
                HOURS, lb=0, ub=0, vtype=GRB.CONTINUOUS
            )
            trans_cap_AC[pair] = model.addVar(lb=0, ub=0, vtype=GRB.CONTINUOUS)


    inter_wind = {}
    inter_solar = {}
    curtail_wind = {}
    curtail_solar = {}
    x_wind = {}
    x_solar = {}

    cap_phs = {}
    cap_bat = {}
    cap_ch_h2 = {}
    cap_dis_h2 = {}

    charge_phs = {"Wind": {}, "Solar": {}, "Coal": {}, "Hydro": {}, "Nuclear": {}}
    charge_bat = {"Wind": {}, "Solar": {}, "Coal": {}, "Hydro": {}, "Nuclear": {}}
    charge_h2 = {"Wind": {}, "Solar": {}, "Coal": {}, "Hydro": {}, "Nuclear": {}}
    dischar_phs = {}
    dischar_bat = {}
    dischar_h2 = {}

    tot_energy_phs = {}
    tot_energy_bat = {}
    tot_energy_h2 = {}
    energy_bat = {}
    energy_phs = {}
    energy_h2 = {}
    load_shedding = {}
    new_install_coal = {}
    # slack1 =  {}
    # slack2 =  {}
    # TODO： new pv
    x_wind2 = {}
    x_solar2 = {}
    sum_wind_cell = {}
    sum_solar_cell = {}
    sum_wind_hours = {}
    sum_solar_hours = {}
    var_wind_cap = {}
    var_solar_cap = {}
    PRO_AI_LOAD = {}
    AI_BATCH_RUN = {}
    AI_BATCH_CUM = {}
    AI_BATCH_CUM_SCALE = float(os.environ.get("AI_BATCH_CUM_SCALE", "1000.0"))
    if AI_BATCH_CUM_SCALE <= 0:
        raise ValueError("AI_BATCH_CUM_SCALE must be positive.")
    print(f"AI_BATCH_CUM_SCALE = {AI_BATCH_CUM_SCALE}")
    ai_interface_metadata["AI_BATCH_CUM_SCALE"] = AI_BATCH_CUM_SCALE
    ai_interface_metadata["AI_BATCH_CUM_internal_units"] = (
        "original cumulative GW-hours divided by AI_BATCH_CUM_SCALE"
    )
    AI_OD_FLOW = {}
    AI_RT_OD_FLOW = {}
    AI_RT_ZONE_ROUTE = {}
    AI_UNMET_SITING = {}  # GW: siting workload that cannot be placed (capacity-bound)
    AI_UNMET_RUNTIME = {}  # GW: runtime workload that cannot be placed
    PRO_under_h2_CAP = {}

    # add Vars  ru/rd/load_conv/
    if use_source_cluster_ai_interface and ai_operational_scenario in {"S1", "S4"}:
        allowed_destinations_by_cluster = ai_s4_od_params[
            "allowed_destinations_by_cluster"
        ]
        province_zones = sorted(set(str(z) for z in province_zone.values()))
        provinces_by_zone = {
            z: [str(p) for p in Province if str(province_zone[str(p)]) == z]
            for z in province_zones
        }
        zone_neighbors_full = ai_s4_od_params.get("zone_neighbors") or {
            z: {z} for z in province_zones
        }
        if ai_rt_zone_route_mode == "none":
            allowed_exec_zones_by_site_zone = {z: [z] for z in province_zones}
        elif ai_rt_zone_route_mode == "all":
            allowed_exec_zones_by_site_zone = {
                z: list(province_zones) for z in province_zones
            }
        else:
            allowed_exec_zones_by_site_zone = {
                z: sorted(set(zone_neighbors_full.get(z, {z})) & set(province_zones))
                for z in province_zones
            }
        runtime_fraction_by_cluster = ai_s4_od_params.get(
            "runtime_fraction_by_cluster", {}
        )
        for gid in ai_s4_od_params["source_clusters"]:
            AI_OD_FLOW[gid] = {}
            AI_RT_OD_FLOW[gid] = {}
            if ai_s4_zone_runtime_transfer_active:
                AI_RT_ZONE_ROUTE[gid] = {}
            cluster_mean_ub = float(ai_s4_od_params["cluster_mean_gw"][gid])
            rt_frac = float(runtime_fraction_by_cluster.get(gid, 1.0))
            # B-scheme: unmet slack for workload that cannot be placed
            AI_UNMET_SITING[gid] = model.addVar(
                lb=0, ub=cluster_mean_ub, vtype=GRB.CONTINUOUS,
                name=f"AI_UNMET_SITING_{gid}",
            )
            AI_UNMET_RUNTIME[gid] = model.addVar(
                lb=0, ub=max(0.0, cluster_mean_ub * rt_frac), vtype=GRB.CONTINUOUS,
                name=f"AI_UNMET_RUNTIME_{gid}",
            )
            for dest in allowed_destinations_by_cluster[gid]:
                od_share_cap = float(
                    ai_s4_od_params.get("od_share_cap", {}).get((gid, dest), 1.0)
                )
                destination_energy_avg_ub = (
                    float(
                        ai_s4_od_params["destination_energy_cap_gwh_year"].get(
                            dest, 0.0
                        )
                    )
                    / 8760.0
                )
                # NOTE: od_ub uses flexible_pool_gw * od_share_cap as the
                # primary per-arc bound (destination concentration limit).
                # This replaces the old cluster_mean * share_cap which was
                # too tight for clusters with few same-zone destinations.
                arc_concentration_ub = (
                    ai_s4_od_params.get("siting_pool_gw", hourly_AI_load)
                    * od_share_cap
                )
                od_ub = min(arc_concentration_ub, destination_energy_avg_ub)
                AI_OD_FLOW[gid][dest] = model.addVar(
                    lb=0,
                    ub=od_ub,
                    vtype=GRB.CONTINUOUS,
                    name=f"AI_OD_FLOW_{gid}_to_{dest}",
                )
                # Phase 2b: runtime OD flow is a per-arc subset of the siting
                # OD flow. Upper bound is the siting arc UB scaled by the
                # cluster runtime fraction; the per-arc subset constraint
                # AI_RT_OD_FLOW <= AI_OD_FLOW is added with the OD planning
                # constraints below.
                AI_RT_OD_FLOW[gid][dest] = model.addVar(
                    lb=0,
                    ub=max(0.0, od_ub * rt_frac),
                    vtype=GRB.CONTINUOUS,
                    name=f"AI_RT_OD_FLOW_{gid}_to_{dest}",
                )
            if ai_s4_zone_runtime_transfer_active:
                for zs in province_zones:
                    if zs not in AI_RT_ZONE_ROUTE[gid]:
                        AI_RT_ZONE_ROUTE[gid][zs] = {}
                    if zs not in allowed_exec_zones_by_site_zone:
                        continue
                    for ze in allowed_exec_zones_by_site_zone[zs]:
                        AI_RT_ZONE_ROUTE[gid][zs][ze] = model.addVar(
                            lb=0,
                            vtype=GRB.CONTINUOUS,
                            name=f"AI_RT_ZONE_ROUTE_{gid}_{zs}_to_{ze}",
                        )
        od_arc_capacity_audit = {
            "enabled": True,
            "by_cluster": {},
            "min_cluster_capacity_ratio": None,
            "infeasible_clusters": [],
        }
        for gid in ai_s4_od_params["source_clusters"]:
            cluster_mean = float(ai_s4_od_params["cluster_mean_gw"][gid])
            total_pool = float(ai_s4_od_params.get("siting_pool_gw", getattr(external_ai, "flexible_pool_gw", 0.0)))
            total_arc_ub = 0.0
            for dest in allowed_destinations_by_cluster[gid]:
                od_share = float(
                    ai_s4_od_params.get("od_share_cap", {}).get((gid, dest), 1.0)
                )
                dest_energy_avg_ub = (
                    float(
                        ai_s4_od_params["destination_energy_cap_gwh_year"].get(
                            dest, 0.0
                        )
                    )
                    / 8760.0
                )
                total_arc_ub += max(
                    0.0,
                    min(
                        total_pool * od_share,
                        dest_energy_avg_ub,
                    ),
                )
            capacity_ratio = (
                total_arc_ub / cluster_mean if cluster_mean > 1e-12 else 1.0
            )
            od_arc_capacity_audit["by_cluster"][gid] = {
                "cluster_mean_gw": float(cluster_mean),
                "sum_arc_upper_bound_gw": float(total_arc_ub),
                "capacity_ratio": float(capacity_ratio),
                "allowed_destination_count": int(
                    len(allowed_destinations_by_cluster[gid])
                ),
            }
            if capacity_ratio + 1e-9 < 1.0:
                od_arc_capacity_audit["infeasible_clusters"].append(gid)
        ratios = [
            v["capacity_ratio"]
            for v in od_arc_capacity_audit["by_cluster"].values()
        ]
        od_arc_capacity_audit["min_cluster_capacity_ratio"] = float(
            min(ratios, default=1.0)
        )
        if od_arc_capacity_audit["infeasible_clusters"]:
            print(
                f"WARNING: {len(od_arc_capacity_audit['infeasible_clusters'])} clusters"
                f" have tight OD arc share caps (od_ub uses destination energy cap,"
                f" so this is a policy audit, not a hard constraint)."
            )
        ai_interface_metadata["od_arc_capacity_audit"] = od_arc_capacity_audit
        ai_interface_metadata["od_arcs_unconstrained_diagnostic"] = str_to_bool(
            os.environ.get("AI_OD_UNCONSTRAINED_ARCS", "false"), default=False
        )
        cluster_od_variable_status = {
            "AI_OD_FLOW": {
                "created": True,
                "unit": "GW average flexible workload allocation",
                "index": "source_cluster_g_to_destination_province_d",
                "count": int(
                    sum(len(v) for v in AI_OD_FLOW.values())
                ),
                "arc_policy": (
                    "zone_limited_and_network_tier_limited"
                    if use_network_tier_caps
                    else "zone_limited"
                ),
            },
            "AI_BATCH_RUN": {
                "physically_created": True,
                "active": ai_operational_scenario == "S4",
                "unit": "GW hourly batch execution",
                "index": "destination_province_d_by_hour_h",
                "physical_count": int(len(Province) * HOURS),
                "active_count": (
                    int(len(Province) * HOURS)
                    if ai_operational_scenario == "S4"
                    else 0
                ),
                "upper_bound_convention": (
                    "destination_power_cap_gw for S4-OD; zero upper bound for S0/S1"
                ),
                "note": (
                    "AI_BATCH_RUN variables are physically created for output compatibility. "
                    "They are active only in S4-OD. S1 injects fixed OD-profiled destination "
                    "load directly into the power balance. S0 includes flexible AI directly "
                    "in LOAD_DEMAND."
                ),
            },
            "AI_BATCH_CUM": {
                "physically_created": ai_operational_scenario == "S4",
                "active": ai_operational_scenario == "S4",
                "unit": "GWh cumulative batch execution",
                "index": "destination_province_d_by_hour_h",
                "physical_count": (
                    int(len(Province) * HOURS)
                    if ai_operational_scenario == "S4"
                    else 0
                ),
                "active_count": (
                    int(len(Province) * HOURS)
                    if ai_operational_scenario == "S4"
                    else 0
                ),
                "note": (
                    "AI_BATCH_CUM variables are only created for S4-OD. "
                    "S0 and S1 save zero arrays for downstream compatibility."
                ),
            },
        }
        ai_interface_metadata["cluster_od_variable_status"] = cluster_od_variable_status
        if ai_operational_scenario == "S4":
            ai_interface_metadata["s4_od_variable_status"] = cluster_od_variable_status
        elif ai_operational_scenario == "S1":
            ai_interface_metadata["s1_od_variable_status"] = cluster_od_variable_status
        ai_interface_metadata["network_tier_policy"] = {
            "enabled": bool(use_network_tier_caps),
            "policy_file": ai_od_network_policy_file or None,
            "default_missing_tier": int(ai_od_default_missing_tier),
            "use_destination_tier_fallback": bool(
                ai_use_destination_tier_fallback
            ),
            "tier_meaning": {
                "0": "infeasible",
                "1": "weak network / low hosting feasibility",
                "2": "medium network / normal hosting feasibility",
                "3": "strong network or AI hub",
            },
            "od_tier_share_cap": {
                "1": float(ai_od_tier1_share_cap),
                "2": float(ai_od_tier2_share_cap),
                "3": float(ai_od_tier3_share_cap),
            },
            "destination_tier_share_cap": {
                "1": float(ai_dest_tier1_share_cap),
                "2": float(ai_dest_tier2_share_cap),
                "3": float(ai_dest_tier3_share_cap),
            },
            "od_share_cap_interpretation": (
                "OD share cap limits the share of one source cluster that can "
                "be assigned to one destination province."
            ),
            "destination_share_cap_interpretation": (
                "Destination share cap limits the total national flexible AI "
                "workload hosted by one destination province according to its "
                "network/hosting tier."
            ),
        }
        ai_interface_metadata["host_cap_policy"] = {
            "host_cap_scenario": ai_host_cap_scenario,
            "legacy_tier_policy_used_for_main_results": (
                ai_host_cap_scenario == "legacy_tier"
            ),
            "destination_hosting_class": {
                str(pro): _default_ai_hosting_class(str(pro)) for pro in Province
            },
            "destination_host_class_share_cap": {
                "A": float(AI_SCENARIO_DEST_CAPS[ai_host_cap_scenario].get("A", 0.0)),
                "B": float(AI_SCENARIO_DEST_CAPS[ai_host_cap_scenario].get("B", 0.0)),
                "C": float(AI_SCENARIO_DEST_CAPS[ai_host_cap_scenario].get("C", 0.0)),
                "D": float(AI_SCENARIO_DEST_CAPS[ai_host_cap_scenario].get("D", 0.0)),
                "E": float(AI_SCENARIO_DEST_CAPS[ai_host_cap_scenario].get("E", 0.0)),
            },
            "od_class_share_cap": {
                "A": float(AI_SCENARIO_OD_CAPS[ai_host_cap_scenario].get("A", 0.0)),
                "B": float(AI_SCENARIO_OD_CAPS[ai_host_cap_scenario].get("B", 0.0)),
                "C": float(AI_SCENARIO_OD_CAPS[ai_host_cap_scenario].get("C", 0.0)),
                "D": float(AI_SCENARIO_OD_CAPS[ai_host_cap_scenario].get("D", 0.0)),
                "E": float(AI_SCENARIO_OD_CAPS[ai_host_cap_scenario].get("E", 0.0)),
            },
        }
        ai_interface_metadata["legacy_tier_cap_parameters_used_for_main_results"] = (
            ai_host_cap_scenario == "legacy_tier"
        )

    for pro in Province:
        underground_h2_ub = underground_pro[pro] if h2_underground_allowed else 0
        PRO_under_h2_CAP[pro] = model.addVar(
            lb=0,
            ub=underground_h2_ub,
            vtype=GRB.CONTINUOUS,
            name="PRO_under_h2_CAP" + str(pro),
        )

        if AI_OD_FLOW:
            pro_ai_load_lb = 0.0
            pro_ai_load_ub = (
                float(ai_s4_od_params["destination_energy_cap_gwh_year"][pro])
                / 8760.0
            )
        elif use_external_ai_load and ai_operational_scenario not in {"S1", "S4"}:
            pro_ai_load_lb = 0.0
            pro_ai_load_ub = 0.0
        else:
            pro_ai_load_ub = (
                ai_hosting_ub_gw.get(pro, GRB.INFINITY)
                if use_external_ai_load
                else GRB.INFINITY
            )
            pro_ai_load_lb = ai_hosting_lb_gw.get(pro, 0.0) if use_external_ai_load else 0.0
        PRO_AI_LOAD[pro] = model.addVar(
            lb=pro_ai_load_lb,
            ub=pro_ai_load_ub,
            vtype=GRB.CONTINUOUS,
            name="PRO_AI_LOAD" + str(pro),
        )
        if AI_OD_FLOW:
            batch_run_ub = float(
                ai_s4_od_params["destination_power_cap_gw"].get(str(pro), 0.0)
            ) if ai_operational_scenario == "S4" else 0.0
        else:
            batch_run_ub = 0.0
        AI_BATCH_RUN[pro] = model.addVars(
            HOURS,
            lb=0,
            ub=batch_run_ub,
            vtype=GRB.CONTINUOUS,
            name="AI_BATCH_RUN" + str(pro),
        )
        if use_external_ai_load and ai_operational_scenario == "S4":
            batch_cum_ub = (
                batch_run_ub * HOURS / AI_BATCH_CUM_SCALE
                if batch_run_ub > 0
                else 0.0
            )
            AI_BATCH_CUM[pro] = model.addVars(
                HOURS,
                lb=0,
                ub=batch_cum_ub,
                vtype=GRB.CONTINUOUS,
                name="AI_BATCH_CUM_SCALED" + str(pro),
            )
            model.addConstr(
                AI_BATCH_CUM[pro][0] == AI_BATCH_RUN[pro][0] / AI_BATCH_CUM_SCALE,
                name=f"AI_BATCH_CUM_init_scaled_{pro}",
            )
            model.addConstrs(
                (
                    AI_BATCH_CUM[pro][h]
                    == AI_BATCH_CUM[pro][h - 1]
                    + AI_BATCH_RUN[pro][h] / AI_BATCH_CUM_SCALE
                    for h in range(1, HOURS)
                ),
                name=f"AI_BATCH_CUM_recur_scaled_{pro}",
            )
        new_install_coal[pro] = model.addVar(
            lb=0, vtype=GRB.CONTINUOUS, name="new_install_Coal" + str(pro)
        )
        ru["Coal"][pro] = model.addVars(
            HOURS, lb=0, vtype=GRB.CONTINUOUS, name="ru_Coal" + str(pro)
        )
        rd["Coal"][pro] = model.addVars(
            HOURS, lb=0, vtype=GRB.CONTINUOUS, name="rd_Coal" + str(pro)
        )
        load_conv["Coal"][pro] = model.addVars(
            HOURS, lb=0, vtype=GRB.CONTINUOUS, name="load_conv_Coal" + str(pro)
        )

        ru["Hydro"][pro] = model.addVars(
            HOURS, lb=0, vtype=GRB.CONTINUOUS, name="ru_Hydro" + str(pro)
        )
        rd["Hydro"][pro] = model.addVars(
            HOURS, lb=0, vtype=GRB.CONTINUOUS, name="rd_Hydro" + str(pro)
        )
        load_conv["Hydro"][pro] = model.addVars(
            HOURS, lb=0, vtype=GRB.CONTINUOUS, name="load_conv_Hydro" + str(pro)
        )

        ru["Nuclear"][pro] = model.addVars(
            HOURS, lb=0, vtype=GRB.CONTINUOUS, name="ru_Nuclear" + str(pro)
        )
        rd["Nuclear"][pro] = model.addVars(
            HOURS, lb=0, vtype=GRB.CONTINUOUS, name="rd_Nuclear" + str(pro)
        )
        load_conv["Nuclear"][pro] = model.addVars(
            HOURS, lb=0, vtype=GRB.CONTINUOUS, name="load_conv_Nuclear" + str(pro)
        )

        ru["BECCS"][pro] = model.addVars(
            HOURS, lb=0, vtype=GRB.CONTINUOUS, name="ru_BECCS" + str(pro)
        )
        rd["BECCS"][pro] = model.addVars(
            HOURS, lb=0, vtype=GRB.CONTINUOUS, name="rd_BECCS" + str(pro)
        )
        load_conv["BECCS"][pro] = model.addVars(
            HOURS, lb=0, vtype=GRB.CONTINUOUS, name="load_conv_BECCS" + str(pro)
        )

        ru["GAS"][pro] = model.addVars(
            HOURS, lb=0, vtype=GRB.CONTINUOUS, name="ru_GAS" + str(pro)
        )
        rd["GAS"][pro] = model.addVars(
            HOURS, lb=0, vtype=GRB.CONTINUOUS, name="rd_GAS" + str(pro)
        )
        load_conv["GAS"][pro] = model.addVars(
            HOURS, lb=0, vtype=GRB.CONTINUOUS, name="load_conv_GAS" + str(pro)
        )

        if install_cap_coal[pro] == 0:
            trans_out["Coal"][pro] = model.addVars(
                HOURS,
                lb=0,
                ub=0,
                vtype=GRB.CONTINUOUS,
                name="trans_out_Coal" + str(pro),
            )
        else:
            trans_out["Coal"][pro] = model.addVars(
                HOURS, lb=0, vtype=GRB.CONTINUOUS, name="trans_out_Coal" + str(pro)
            )

        if install_cap_hydro[pro] == 0:
            trans_out["Hydro"][pro] = model.addVars(
                HOURS,
                lb=0,
                ub=0,
                vtype=GRB.CONTINUOUS,
                name="trans_out_Hydro" + str(pro),
            )
        else:
            trans_out["Hydro"][pro] = model.addVars(
                HOURS, lb=0, vtype=GRB.CONTINUOUS, name="trans_out_Hydro" + str(pro)
            )

        if install_cap_Nuclear[pro] == 0:
            trans_out["Nuclear"][pro] = model.addVars(
                HOURS,
                lb=0,
                ub=0,
                vtype=GRB.CONTINUOUS,
                name="trans_out_Nuclear" + str(pro),
            )
        else:
            trans_out["Nuclear"][pro] = model.addVars(
                HOURS, lb=0, vtype=GRB.CONTINUOUS, name="trans_out_Nuclear" + str(pro)
            )

        has_wind_resource = np.sum(wind_cap[pro]) > 1e-9
        has_solar_resource = np.sum(pv_cap[pro]) > 1e-9

        for l in ["Wind", "Solar"]:
            resource_ub = GRB.INFINITY
            if (l == "Wind" and not has_wind_resource) or (
                l == "Solar" and not has_solar_resource
            ):
                resource_ub = 0
            trans_out[l][pro] = model.addVars(
                HOURS,
                lb=0,
                ub=resource_ub,
                vtype=GRB.CONTINUOUS,
                name=f"trans_out_{l}" + str(pro),
            )

        trans_out["PHS"][pro] = model.addVars(
            HOURS, lb=0, vtype=GRB.CONTINUOUS, name="trans_out_PHS" + str(pro)
        )
        trans_out["BAT"][pro] = model.addVars(
            HOURS, lb=0, vtype=GRB.CONTINUOUS, name="trans_out_BAT" + str(pro)
        )
        trans_out["H2"][pro] = model.addVars(
            HOURS, lb=0, vtype=GRB.CONTINUOUS, name="trans_out_H2" + str(pro)
        )

        var_wind_cap[pro] = model.addVars(
            int(wind_cell[pro]), lb=0, vtype=GRB.CONTINUOUS, name="wind_cap" + str(pro)
        )
        var_solar_cap[pro] = model.addVars(
            int(pv_cell[pro]), lb=0, vtype=GRB.CONTINUOUS, name="pv_cap" + str(pro)
        )

        wind_resource_ub = GRB.INFINITY if has_wind_resource else 0
        sum_wind_cell[pro] = model.addVars(
            int(wind_cell[pro]),
            lb=0,
            ub=wind_resource_ub,
            vtype=GRB.CONTINUOUS,
            name="sum_wind_cell" + str(pro),
        )
        sum_wind_hours[pro] = model.addVars(
            HOURS,
            lb=0,
            ub=wind_resource_ub,
            vtype=GRB.CONTINUOUS,
            name="sum_wind_hours" + str(pro),
        )
        inter_wind[pro] = model.addVars(
            HOURS,
            lb=0,
            ub=wind_resource_ub,
            vtype=GRB.CONTINUOUS,
            name="inter_wind" + str(pro),
        )

        solar_resource_ub = GRB.INFINITY if has_solar_resource else 0
        sum_solar_cell[pro] = model.addVars(
            int(pv_cell[pro]),
            lb=0,
            ub=solar_resource_ub,
            vtype=GRB.CONTINUOUS,
            name="sum_solar_cell" + str(pro),
        )

        sum_solar_hours[pro] = model.addVars(
            HOURS,
            lb=0,
            ub=solar_resource_ub,
            vtype=GRB.CONTINUOUS,
            name="sum_solar_hours" + str(pro),
        )

        wind_cf_arr = np.asarray(wind_cf[pro], dtype=float)
        pv_cf_arr = np.asarray(pv_cf[pro], dtype=float)

        model.addConstrs(
            sum_wind_cell[pro][c]
            == var_wind_cap[pro][c] * float(wind_cf_arr[c, :HOURS].sum())
            for c in range(int(wind_cell[pro]))
        )

        model.addConstrs(
            sum_solar_cell[pro][c]
            == var_solar_cap[pro][c] * float(pv_cf_arr[c, :HOURS].sum())
            for c in range(int(pv_cell[pro]))
        )

        model.addConstrs(
            sum_wind_hours[pro][h]
            == gp.quicksum(
                var_wind_cap[pro][c] * float(wind_cf_arr[c, h])
                for c in range(int(wind_cell[pro]))
            )
            for h in range(HOURS)
        )

        model.addConstrs(
            sum_solar_hours[pro][h]
            == gp.quicksum(
                var_solar_cap[pro][c] * float(pv_cf_arr[c, h])
                for c in range(int(pv_cell[pro]))
            )
            for h in range(HOURS)
        )

        inter_solar[pro] = model.addVars(
            HOURS,
            lb=0,
            ub=solar_resource_ub,
            vtype=GRB.CONTINUOUS,
            name="inter_solar" + str(pro),
        )
        curtail_wind[pro] = model.addVars(
            HOURS,
            lb=0,
            ub=wind_resource_ub,
            vtype=GRB.CONTINUOUS,
            name="curtail_wind" + str(pro),
        )
        curtail_solar[pro] = model.addVars(
            HOURS,
            lb=0,
            ub=solar_resource_ub,
            vtype=GRB.CONTINUOUS,
            name="curtail_solar" + str(pro),
        )
        load_shedding[pro] = model.addVars(
            HOURS, lb=0, vtype=GRB.CONTINUOUS, name="load_shedding" + str(pro)
        )

        ## TODO: h2 energy FC EL
        # cap
        cap_phs[pro] = model.addVar(
            lb=Params.PHS_installed[pro] / 1000,
            ub=(Params.PHS_under_installed[pro] + Params.PHS_long_term_cap[pro]) / 1000,
            vtype=GRB.CONTINUOUS,
            name="cap_phs" + str(pro),
        )
        cap_bat[pro] = model.addVar(
            lb=0, vtype=GRB.CONTINUOUS, name="cap_bat" + str(pro)
        )
        h2_aboveground_ub = GRB.INFINITY if h2_aboveground_allowed else 0
        cap_ch_h2[pro] = model.addVar(
            lb=0, ub=h2_aboveground_ub, vtype=GRB.CONTINUOUS, name="cap_h2" + str(pro)
        )
        cap_dis_h2[pro] = model.addVar(
            lb=0, ub=h2_aboveground_ub, vtype=GRB.CONTINUOUS, name="cap_h2_dis" + str(pro)
        )

        # energy
        energy_bat[pro] = model.addVar(
            lb=0, vtype=GRB.CONTINUOUS, name="energy_bat" + str(pro)
        )
        energy_phs[pro] = model.addVar(
            lb=0, vtype=GRB.CONTINUOUS, name="energy_phs" + str(pro)
        )
        energy_h2[pro] = model.addVar(
            lb=0, ub=h2_aboveground_ub, vtype=GRB.CONTINUOUS, name="energy_h2" + str(pro)
        )

        for et in ["Wind", "Solar", "Coal", "Hydro", "Nuclear"]:
            charge_ub = GRB.INFINITY
            if et == "Wind" and not has_wind_resource:
                charge_ub = 0
            elif et == "Solar" and not has_solar_resource:
                charge_ub = 0
            charge_phs[et][pro] = model.addVars(
                HOURS,
                lb=0,
                ub=charge_ub,
                vtype=GRB.CONTINUOUS,
                name=f"charge_phs_{et}" + str(pro),
            )
            charge_bat[et][pro] = model.addVars(
                HOURS,
                lb=0,
                ub=charge_ub,
                vtype=GRB.CONTINUOUS,
                name=f"charge_bat_{et}" + str(pro),
            )
            charge_h2[et][pro] = model.addVars(
                HOURS,
                lb=0,
                ub=charge_ub,
                vtype=GRB.CONTINUOUS,
                name=f"charge_h2_{et}" + str(pro),
            )

        dischar_phs[pro] = model.addVars(
            HOURS, lb=0, vtype=GRB.CONTINUOUS, name="dischar_phs" + str(pro)
        )
        dischar_bat[pro] = model.addVars(
            HOURS, lb=0, vtype=GRB.CONTINUOUS, name="dischar_bat" + str(pro)
        )
        dischar_h2[pro] = model.addVars(
            HOURS, lb=0, vtype=GRB.CONTINUOUS, name="dischar_h2" + str(pro)
        )

        tot_energy_h2[pro] = model.addVars(
            HOURS, lb=0, vtype=GRB.CONTINUOUS, name="tot_energy_h2" + str(pro)
        )
        tot_energy_phs[pro] = model.addVars(
            HOURS, lb=0, vtype=GRB.CONTINUOUS, name="tot_energy_phs" + str(pro)
        )
        tot_energy_bat[pro] = model.addVars(
            HOURS, lb=0, vtype=GRB.CONTINUOUS, name="tot_energy_bat" + str(pro)
        )
        # TODO: add objective

    wind_gen_cost = []
    for pro in Province:
        if np.array(wind_cf[pro]).shape[0] == 0:
            continue

        wind_gen_cost += [
            wind_lcoe[pro][c]
            * sum_wind_cell[pro][c]
            * min(lcoe[args.Mode]["LandbasedWind"](args.test_years), 1)
            for c in range(int(wind_cell[pro]))
        ]

    solar_gen_cost = []
    for pro in Province:
        if np.array(pv_cf[pro]).shape[0] == 0:
            continue

        solar_gen_cost += [
            pv_lcoe[pro][c]
            * sum_solar_cell[pro][c]
            * min(lcoe[args.Mode]["CommPV"](args.test_years), 1)
            for c in range(int(pv_cell[pro]))
        ]

    ramp_up_cost = [
        ru_c[l] * ru[l][pro][h]
        for l in ["Coal", "Hydro", "Nuclear", "BECCS", "GAS"]
        for pro in Province
        for h in range(HOURS)
    ]
    ramp_dn_cost = [
        rd_c[l] * rd[l][pro][h]
        for l in ["Coal", "Hydro", "Nuclear", "BECCS", "GAS"]
        for pro in Province
        for h in range(HOURS)
    ]

    coal_ccs_cost = [
        Params.Coal_price[pro] * COAL_PRICE_YEAR_MULTIPLIER.get(args.test_years, 1.0)
        * (
            load_conv["Coal"][pro][h]
            + trans_out["Coal"][pro][h]
            + charge_bat["Coal"][pro][h]
            + charge_phs["Coal"][pro][h]
            + charge_h2["Coal"][pro][h]
        )
        for pro in Province
        for h in range(HOURS)
    ]

    coal_ccs_cost_fixed = [
        min(lcoe[args.Mode]["Coal_FE"](args.test_years), 1)
        * Params.CAPEX.gen.Coal
        * CRF(0.05, 40)
        * (install_cap_coal[pro] + new_install_coal[pro])
        * HORIZON_SCALE
        for pro in Province
    ]
    coal_ccs_OM_fixed = [
        Params.OM.Fixed.Coal
        * (install_cap_coal[pro] + new_install_coal[pro])
        * HORIZON_SCALE
        for pro in Province
    ]

    gas_ccs_cost = [
        Params.Gas_price[pro] * (load_conv["GAS"][pro][h])
        for pro in Province
        for h in range(HOURS)
    ]
    gas_ccs_cost_fixed = [
        min(lcoe[args.Mode]["Coal_FE"](args.test_years), 1)
        * Params.CAPEX.gen.Gas
        * CRF(0.05, 40)
        * install_cap_Other[pro]
        * HORIZON_SCALE
        for pro in Province
    ]
    gas_ccs_OM_fixed = [
        Params.OM.Fixed.Gas * install_cap_Other[pro] * HORIZON_SCALE
        for pro in Province
    ]

    hydro_cost_fixed = [
        min(lcoe[args.Mode]["Hydropower"](args.test_years), 1)
        * Params.CAPEX.gen.Hydro
        * CRF(0.05, 50)
        * install_cap_hydro[pro]
        * HORIZON_SCALE
        for pro in Province
    ]
    hydro_OM_fixed = [
        Params.OM.Fixed.Hydro * install_cap_hydro[pro] * HORIZON_SCALE
        for pro in Province
    ]
    hydro_cost_fuel = [
        0.1
        * (
            load_conv["Hydro"][pro][h]
            + trans_out["Hydro"][pro][h]
            + charge_bat["Hydro"][pro][h]
            + charge_phs["Hydro"][pro][h]
            + charge_h2["Hydro"][pro][h]
        )
        for pro in Province
        for h in range(HOURS)
    ]

    nuclear_cost_fixed = [
        min(lcoe[args.Mode]["Nuclear"](args.test_years), 1)
        * Params.CAPEX.gen.Nuclear
        * CRF(0.05, 50)
        * install_cap_Nuclear[pro]
        * HORIZON_SCALE
        for pro in Province
    ]

    nuclear_cost_var = [
        Params.OM.fuel.Nuclear
        * (
            load_conv["Nuclear"][pro][h]
            + trans_out["Nuclear"][pro][h]
            + charge_bat["Nuclear"][pro][h]
            + charge_phs["Nuclear"][pro][h]
            + charge_h2["Nuclear"][pro][h]
        )
        for pro in Province
        for h in range(HOURS)
    ]

    nuclear_OM_fixed = [
        Params.OM.Fixed.Nuclear * install_cap_Nuclear[pro] * HORIZON_SCALE
        for pro in Province
    ]

    beccs_cost_fixed = [
        min(lcoe[args.Mode]["Biopower"](args.test_years), 1)
        * Params.CAPEX.gen.BECCS
        * CRF(0.05, 35)
        * install_cap_Bios[pro]
        * HORIZON_SCALE
        for pro in Province
    ]

    beccs_OM_fixed = [
        Params.OM.Fixed.BECCS * install_cap_Bios[pro] * HORIZON_SCALE
        for pro in Province
    ]

    beccs_fuel = [
        0.2 * (load_conv["BECCS"][pro][h]) for pro in Province for h in range(HOURS)
    ]

    load_shedding_cost = [
        load_shedding[pro][h] * Params.load_shadding[f"{pro}"]
        for pro in Province
        for h in range(HOURS)
    ]
    fixed_phs_cost = [
        (
            Params.CAPEX.storage.Hydro.power * cap_phs[pro]
            + Params.CAPEX.storage.Hydro.CAP * energy_phs[pro]
        )
        * CRF(discount_rate=0.05, lifetime=40)
        * HORIZON_SCALE
        for pro in Province
    ]
    fixed_bat_cost = [
        (
            min(lcoe[args.Mode]["Utility-Scale Battery Storage"](args.test_years), 1)
            * Params.CAPEX.storage.Batter.power
            * cap_bat[pro]
            + min(lcoe[args.Mode]["Utility-Scale Battery Storage"](args.test_years), 1)
            * Params.CAPEX.storage.Batter.CAP
            * energy_bat[pro]
        )
        * CRF(discount_rate=0.05, lifetime=20)
        * HORIZON_SCALE
        for pro in Province
    ]
    fixed_h2_cost = [
        (
            lcoe["h2"][args.test_years]["el"][args.Mode][args.EL]
            * CRF(discount_rate=0.05, lifetime=15)
            * cap_ch_h2[pro]
            + lcoe["h2"][args.test_years]["fc"][args.Mode]
            * CRF(discount_rate=0.05, lifetime=15)
            * cap_dis_h2[pro]
            + lcoe["h2"][args.test_years]["ht"][args.Mode]
            * CRF(discount_rate=0.05, lifetime=40)
            * energy_h2[pro]
            + PRO_under_h2_CAP[pro]
            * lcoe["h2"]["underground"]
            * CRF(discount_rate=0.05, lifetime=40)
        )
        * HORIZON_SCALE
        for pro in Province
    ]
    storage_charge_sources = ["Wind", "Solar", "Coal", "Hydro", "Nuclear"]
    var_phs_cost = [
        Params.OM.VOM.he
        * (
            gp.quicksum(charge_phs[et][pro][h] for et in storage_charge_sources)
            + dischar_phs[pro][h]
            + trans_out["PHS"][pro][h]
        )
        for pro in Province
        for h in range(HOURS)
    ]
    var_bat_cost = [
        Params.OM.VOM.bat
        * (
            gp.quicksum(charge_bat[et][pro][h] for et in storage_charge_sources)
            + dischar_bat[pro][h]
            + trans_out["BAT"][pro][h]
        )
        for pro in Province
        for h in range(HOURS)
    ]
    var_h2_cost = [
        Params.OM.VOM.H2
        * (
            gp.quicksum(charge_h2[et][pro][h] for et in storage_charge_sources)
            + dischar_h2[pro][h]
            + trans_out["H2"][pro][h]
        )
        for pro in Province
        for h in range(HOURS)
    ]

    wheeling_cost = float(getattr(args, "wheeling_cost", 0.0))
    # Objective costs are in 1e6 yuan. Flow variables are GW for each hour
    # (GWh), so RMB/MWh * GWh = 1000 RMB = 1e-3 million RMB.
    wheeling_cost_obj = wheeling_cost / 1000

    trans_fixed_cost_dc_installed = [
        trans_data["DC_installed"]["lcoe"][pair]
        * trans_data["DC_installed"]["cap"][pair]
        * args.trans_cost
        * HORIZON_SCALE
        for pair in trans_data["DC_installed"]["pair"]
    ]
    trans_fixed_cost_ac_installed = [
        trans_data["AC_installed"]["lcoe"][pair]
        * trans_data["AC_installed"]["cap"][pair]
        * args.trans_cost
        * HORIZON_SCALE
        for pair in trans_data["AC_installed"]["pair"]
    ]
    trans_expansion_cost_dc_installed = [
        trans_data["DC_installed"]["lcoe"][pair]
        * trans_cap_DC_installed_expansion[pair]
        * args.trans_cost
        * HORIZON_SCALE
        for pair in trans_data["DC_installed"]["pair"]
    ]
    trans_expansion_cost_ac_installed = [
        trans_data["AC_installed"]["lcoe"][pair]
        * trans_cap_AC_installed_expansion[pair]
        * args.trans_cost
        * HORIZON_SCALE
        for pair in trans_data["AC_installed"]["pair"]
    ]
    trans_fixed_cost_dc = [
        trans_data["DC"]["lcoe"][pair]
        * trans_cap_DC[pair]
        * args.trans_cost
        * HORIZON_SCALE
        for pair in trans_data["DC"]["pair"]
    ]
    trans_fixed_cost_ac = [
        trans_data["AC"]["lcoe"][pair]
        * trans_cap_AC[pair]
        * args.trans_cost
        * HORIZON_SCALE
        for pair in trans_data["AC"]["pair"]
    ]
    trans_flow_cost_dc_installed = [
        wheeling_cost_obj * load_trans_DC_installed[pair][h]
        for pair in trans_data["DC_installed"]["pair"]
        for h in range(HOURS)
    ]
    trans_flow_cost_ac_installed = [
        wheeling_cost_obj
        * (
            load_trans_AC_installed[pair][h]
            + load_trans_AC_installed[pair[1], pair[0]][h]
        )
        for pair in trans_data["AC_installed"]["pair"]
        for h in range(HOURS)
    ]
    trans_flow_cost_dc = [
        wheeling_cost_obj * load_trans_DC[pair][h]
        for pair in trans_data["DC"]["pair"]
        for h in range(HOURS)
    ]
    trans_flow_cost_ac = [
        wheeling_cost_obj
        * (load_trans_AC[pair][h] + load_trans_AC[pair[1], pair[0]][h])
        for pair in trans_data["AC"]["pair"]
        for h in range(HOURS)
    ]
    ai_hosting_preference_penalty = []
    ai_batch_hosting_preference_penalty = []
    ai_hosting_penalty_cost = []
    ai_batch_hosting_penalty_cost = []
    if compute_ai_penalty_diagnostic and (
        (not use_external_ai_load) or ai_operational_scenario in {"S1", "S4"}
    ):
        # Objective costs are in 1e6 yuan; PRO_AI_LOAD is GW average.
        # RMB/MWh * (GW * hour) = RMB/MWh * GWh = 1e-3 million yuan.
        ai_hosting_preference_penalty = [
            ai_hosting_penalty_rmb_per_mwh.get(pro, 0.0)
            * PRO_AI_LOAD[pro]
            * HOURS
            / 1000.0
            for pro in Province
        ]
    ai_hosting_penalty_cost = ai_hosting_preference_penalty
    ai_batch_hosting_penalty_cost = ai_batch_hosting_preference_penalty

    # ---- Phase C: migration cost term (cross-zone siting OD arcs) ----
    # Applied in stage 2 when cross-zone arcs are identified.
    # c_mig_per_gw_year [yuan/GW-yr] / 1e6 → million yuan / GW-yr.
    migration_cost = []
    # Phase C: cross-zone migration penalty on siting OD flow.
    # Objective unit = 1e6 yuan; AI_OD_FLOW is GW (average over horizon),
    # c_mig_per_gw_year is yuan/(GW·yr). HORIZON_SCALE annualizes.
    # million yuan = GW * (yuan/GW-yr / 1e6) * HORIZON_SCALE
    if (
        AI_OD_FLOW
        and ai_s4_od_params.get("cross_zone_arcs")
        and c_mig_per_gw_year > 0.0
    ):
        c_mig_obj = c_mig_per_gw_year / 1e6  # → million yuan / GW-yr
        for (gid, dest) in ai_s4_od_params["cross_zone_arcs"]:
            if gid in AI_OD_FLOW and dest in AI_OD_FLOW[gid]:
                migration_cost.append(
                    AI_OD_FLOW[gid][dest] * c_mig_obj * HORIZON_SCALE
                )

    # B-scheme: unmet penalty. High price per GW of unplaceable workload.
    # Default 1e6 million-yuan/GW ensures unmet only when capacity truly insufficient.
    ai_unmet_penalty_per_gw = float(
        os.environ.get("AI_UNMET_PENALTY_PER_GW", "1e6")
    )
    unmet_penalty = []
    if AI_UNMET_SITING:
        _p = ai_unmet_penalty_per_gw * HORIZON_SCALE
        for gid in AI_UNMET_SITING:
            unmet_penalty.append(AI_UNMET_SITING[gid] * _p)

    power_system_objective = (
        gp.quicksum(wind_gen_cost)
        + gp.quicksum(solar_gen_cost)
        + gp.quicksum(ramp_up_cost)
        + gp.quicksum(ramp_dn_cost)
        + gp.quicksum(coal_ccs_cost)
        + gp.quicksum(coal_ccs_OM_fixed)
        + gp.quicksum(coal_ccs_cost_fixed)
        + gp.quicksum(gas_ccs_cost)
        + gp.quicksum(gas_ccs_OM_fixed)
        + gp.quicksum(gas_ccs_cost_fixed)
        + gp.quicksum(nuclear_cost_var)
        + gp.quicksum(nuclear_cost_fixed)
        + gp.quicksum(nuclear_OM_fixed)
        + gp.quicksum(hydro_cost_fixed)
        + gp.quicksum(hydro_OM_fixed)
        + gp.quicksum(beccs_cost_fixed)
        + gp.quicksum(beccs_OM_fixed)
        + gp.quicksum(load_shedding_cost)
        + gp.quicksum(trans_fixed_cost_dc_installed)
        + gp.quicksum(trans_fixed_cost_ac_installed)
        + gp.quicksum(trans_expansion_cost_dc_installed)
        + gp.quicksum(trans_expansion_cost_ac_installed)
        + gp.quicksum(trans_fixed_cost_dc)
        + gp.quicksum(trans_fixed_cost_ac)
        + gp.quicksum(trans_flow_cost_dc_installed)
        + gp.quicksum(trans_flow_cost_ac_installed)
        + gp.quicksum(trans_flow_cost_dc)
        + gp.quicksum(trans_flow_cost_ac)
        + gp.quicksum(fixed_phs_cost)
        + gp.quicksum(fixed_bat_cost)
        + gp.quicksum(fixed_h2_cost)
        + gp.quicksum(var_phs_cost)
        + gp.quicksum(var_bat_cost)
        + gp.quicksum(var_h2_cost)
        + gp.quicksum(hydro_cost_fuel)
        + gp.quicksum(beccs_fuel)
        + gp.quicksum(migration_cost)
        + gp.quicksum(unmet_penalty)
    )
    ai_preference_penalty_objective = (
        gp.quicksum(ai_hosting_preference_penalty)
        + gp.quicksum(ai_batch_hosting_preference_penalty)
    )
    model.setObjective(
        power_system_objective
        + (
            ai_preference_penalty_objective
            if include_ai_penalty_in_objective
            else 0
        ),
        GRB.MINIMIZE,
    )

    wind_cell_installed_cap_cell2 = {}
    pv_cell_installed_cap_cell2 = {}
    for pro in Province:
        if np.sum(wind_cap[pro]) == 0:
            wind_cell_installed_cap_cell2[pro] = np.zeros(int(wind_cell[pro]))
            continue
        if (install_cap_Wind[pro] / np.sum(wind_cap[pro])) >= 1:
            wind_cell_installed_cap_cell2[pro] = np.array(wind_cap[pro])
        else:
            wind_cell_installed_cap_cell2[pro] = (
                install_cap_Wind[pro]
                / int(wind_cell[pro])
                * np.ones(int(wind_cell[pro]))
            )

        wind_cell_installed_cap_cell2[pro] = np.minimum(
            wind_cell_installed_cap_cell2[pro], np.array(wind_cap[pro])
        )
    model.addConstrs(
        (var_wind_cap[pro][c] >= wind_cell_installed_cap_cell2[pro][c])
        for pro in Province
        for c in range(int(wind_cell[pro]))
    )

    model.addConstrs(
        (var_wind_cap[pro][c] <= wind_cap[pro][c])
        for pro in Province
        for c in range(int(wind_cell[pro]))
    )

    # Redundant sanity check under source-cluster OD planning:
    # source balance + destination plan link already imply this equality.
    # Keep it to catch interface-scale inconsistencies.
    if (not use_external_ai_load) or ai_operational_scenario in {"S0", "S1", "S4"}:
        unmet_siting_total = (
            gp.quicksum(AI_UNMET_SITING[gid] for gid in AI_UNMET_SITING)
            if AI_UNMET_SITING else 0.0
        )
        model.addConstr(
            gp.quicksum([PRO_AI_LOAD[pro] for pro in Province])
            + unmet_siting_total
            == hourly_AI_load,
            name="AI_global_flexible_pool_sanity_balance",
        )
    # Phase 2b: global runtime-pool balance. The runtime OD flow must sum to the
    # runtime pool (a subset of the siting pool). Only meaningful for S1/S4 with
    # an active OD interface; under single-pool degradation runtime == siting so
    # this reduces to the same total as the siting balance.
    if (
        use_external_ai_load
        and ai_operational_scenario in {"S1", "S4"}
        and AI_RT_OD_FLOW
    ):
        runtime_pool_gw_target = float(
            ai_s4_od_params.get("runtime_pool_gw", hourly_AI_load)
        )
        unmet_runtime_total = (
            gp.quicksum(AI_UNMET_RUNTIME[gid] for gid in AI_UNMET_RUNTIME)
            if AI_UNMET_RUNTIME else 0.0
        )
        model.addConstr(
            gp.quicksum(
                AI_RT_OD_FLOW[gid][dest]
                for gid in AI_RT_OD_FLOW
                for dest in AI_RT_OD_FLOW[gid]
            )
            + unmet_runtime_total
            == runtime_pool_gw_target,
            name="AI_global_runtime_pool_balance",
        )
    if use_external_ai_load:
        region_constraints = [
            ("northwest", AI_NORTHWEST_PROVINCES),
            ("southwest", AI_SOUTHWEST_PROVINCES),
            ("coastal_demand", AI_COASTAL_DEMAND_PROVINCES),
        ]
        if ai_operational_scenario in {"S1", "S4"} and not AI_OD_FLOW:
            for region_name, region_provinces in region_constraints:
                if region_name not in ai_region_max_host_share:
                    continue
                region_share = float(ai_region_max_host_share[region_name])
                active = [pro for pro in Province if pro in region_provinces]
                if not active:
                    continue
                model.addConstr(
                    gp.quicksum(PRO_AI_LOAD[pro] for pro in active)
                    <= region_share * hourly_AI_load,
                        name=f"AI_region_host_cap_{region_name}",
                    )

        # Load cluster_profile early so both S1 and S4 can access it
        if AI_OD_FLOW:
            cluster_profile = ai_s4_od_params["cluster_profile"]
            cluster_profile_cum = ai_s4_od_params["cluster_profile_cum"]
            runtime_cluster_profile_cum = ai_s4_od_params.get(
                "runtime_cluster_profile_cum", cluster_profile_cum
            )

        if ai_operational_scenario == "S1" and AI_OD_FLOW:
            destination_provinces = ai_s4_od_params["destination_provinces"]
            allowed_destinations_by_cluster = ai_s4_od_params[
                "allowed_destinations_by_cluster"
            ]
            clusters_by_destination = ai_s4_od_params["clusters_by_destination"]
            destination_power_cap_gw = ai_s4_od_params["destination_power_cap_gw"]
            s1_od_planning_status = add_cluster_od_planning_constraints(
                model=model,
                AI_OD_FLOW=AI_OD_FLOW,
                PRO_AI_LOAD=PRO_AI_LOAD,
                ai_cluster_params=ai_s4_od_params,
                prefix="AI_S1_OD",
                unmet_siting=AI_UNMET_SITING,
            )
            s1_destination_network_share_status = (
                add_destination_hosting_share_constraints(
                    model=model,
                    PRO_AI_LOAD=PRO_AI_LOAD,
                    destination_provinces=destination_provinces,
                    hourly_AI_load=hourly_AI_load,
                    destination_hosting_class=ai_s4_od_params[
                        "destination_hosting_class_by_province"
                    ],
                    class_share_cap=AI_SCENARIO_DEST_CAPS[ai_host_cap_scenario],
                    prefix="AI_S1_OD",
                )
                if ai_host_cap_scenario != "legacy_tier"
                else {"enabled": False, "constraints": 0}
            )

            for dest in destination_provinces:
                incoming_clusters = clusters_by_destination[dest]
                for h in range(HOURS):
                    fixed_gw = float(fixed_ai_load.get(dest, np.zeros(HOURS))[h]) / 1000.0
                    profiled_flexible_gw = gp.quicksum(
                        AI_OD_FLOW[gid][dest] * float(cluster_profile[gid][h])
                        for gid in incoming_clusters
                    )
                    model.addConstr(
                        fixed_gw + profiled_flexible_gw
                        <= float(destination_power_cap_gw[dest]),
                        name=f"AI_S1_OD_destination_power_cap_{dest}_{h}",
                    )
            ai_interface_metadata["s1_od_constraint_status"] = {
                **s1_od_planning_status,
                "destination_hosting_share_constraints": (
                    s1_destination_network_share_status
                ),
                "destination_network_share_constraints": (
                    s1_destination_network_share_status
                ),
                "destination_power_cap_constraints": int(len(destination_provinces) * HOURS),
                "allowed_od_arc_count": int(
                    sum(len(v) for v in allowed_destinations_by_cluster.values())
                ),
                "temporal_coupling": (
                    "fixed destination hourly load profile: "
                    "sum_g AI_OD_FLOW[g,d] * cluster_profile[g,h]"
                ),
                "temporal_execution_constraints": 0,
                "temporal_execution_note": (
                    "S1-OD has no AI_BATCH_RUN/AI_BATCH_CUM temporal execution decision; "
                    "assigned source-cluster profiles run immediately."
                ),
                "batch_execution_decision": False,
            }
            ai_interface_metadata["s1_od_profile_constraint_status"] = (
                ai_interface_metadata["s1_od_constraint_status"]
            )

        if ai_operational_scenario == "S4":
            if not AI_OD_FLOW:
                raise ValueError("S4 must be S4-OD and requires source-cluster OD interface.")
            destination_provinces = ai_s4_od_params["destination_provinces"]
            allowed_destinations_by_cluster = ai_s4_od_params[
                "allowed_destinations_by_cluster"
            ]
            clusters_by_destination = ai_s4_od_params[
                "clusters_by_destination"
            ]
            destination_power_cap_gw = ai_s4_od_params[
                "destination_power_cap_gw"
            ]
            s4_od_planning_status = add_cluster_od_planning_constraints(
                model=model,
                AI_OD_FLOW=AI_OD_FLOW,
                PRO_AI_LOAD=PRO_AI_LOAD,
                ai_cluster_params=ai_s4_od_params,
                prefix="AI_S4_OD",
                unmet_siting=AI_UNMET_SITING,
            )
            s4_destination_network_share_status = (
                add_destination_hosting_share_constraints(
                    model=model,
                    PRO_AI_LOAD=PRO_AI_LOAD,
                    destination_provinces=destination_provinces,
                    hourly_AI_load=hourly_AI_load,
                    destination_hosting_class=ai_s4_od_params[
                        "destination_hosting_class_by_province"
                    ],
                    class_share_cap=AI_SCENARIO_DEST_CAPS[ai_host_cap_scenario],
                    prefix="AI_S4_OD",
                )
                if ai_host_cap_scenario != "legacy_tier"
                else {"enabled": False, "constraints": 0}
            )

            # Phase 2b: per-arc subset (runtime <= siting) and runtime source
            # balance (sum_d AI_RT_OD_FLOW[g,d] == runtime_cluster_mean_gw[g]).
            runtime_cluster_mean_gw = ai_s4_od_params.get(
                "runtime_cluster_mean_gw", {}
            )
            s4_subset_constraints = 0
            s4_runtime_source_balance = 0
            for gid in ai_s4_od_params["source_clusters"]:
                for dest in allowed_destinations_by_cluster[gid]:
                    model.addConstr(
                        AI_RT_OD_FLOW[gid][dest] <= AI_OD_FLOW[gid][dest],
                        name=f"AI_S4_OD_runtime_subset_{gid}_{dest}",
                    )
                    s4_subset_constraints += 1
                model.addConstr(
                    gp.quicksum(
                        AI_RT_OD_FLOW[gid][dest]
                        for dest in allowed_destinations_by_cluster[gid]
                    )
                    + AI_UNMET_RUNTIME.get(gid, 0.0)
                    == float(runtime_cluster_mean_gw.get(gid, 0.0)),
                    name=f"AI_S4_OD_runtime_source_balance_{gid}",
                )
                model.addConstr(
                    AI_UNMET_RUNTIME.get(gid, 0.0) <= AI_UNMET_SITING.get(gid, 0.0),
                    name=f"AI_UNMET_runtime_le_siting_{gid}",
                )
                s4_runtime_source_balance += 1

            if ai_s4_zone_runtime_transfer_active:
                s4z_origin_balance_constraints = 0
                s4z_total_execution_constraints = 0
                s4z_no_advance_constraints = 0
                s4z_deadline_constraints = 0
                s4z_allowed_route_count = 0
                for gid in ai_s4_od_params["source_clusters"]:
                    for zs in province_zones:
                        site_zone_runtime = gp.quicksum(
                            AI_RT_OD_FLOW[gid][p]
                            for p in provinces_by_zone[zs]
                            if p in AI_RT_OD_FLOW.get(gid, {})
                        )
                        route_out = gp.quicksum(
                            AI_RT_ZONE_ROUTE[gid][zs][ze]
                            for ze in AI_RT_ZONE_ROUTE.get(gid, {}).get(zs, {})
                        )
                        model.addConstr(
                            route_out == site_zone_runtime,
                            name=f"AI_S4Z_zone_runtime_origin_balance_{gid}_{zs}",
                        )
                        s4z_origin_balance_constraints += 1
                        s4z_allowed_route_count += len(
                            AI_RT_ZONE_ROUTE.get(gid, {}).get(zs, {})
                        )

                for ze in province_zones:
                    zone_runtime_exec = gp.quicksum(
                        AI_RT_ZONE_ROUTE[gid][zs][ze]
                        for gid in AI_RT_ZONE_ROUTE
                        for zs in AI_RT_ZONE_ROUTE[gid]
                        if ze in AI_RT_ZONE_ROUTE[gid][zs]
                    )
                    model.addConstr(
                        gp.quicksum(
                            AI_BATCH_CUM[p][HOURS - 1]
                            for p in provinces_by_zone[ze]
                        )
                        == zone_runtime_exec * HOURS / AI_BATCH_CUM_SCALE,
                        name=f"AI_S4Z_zone_total_execution_scaled_{ze}",
                    )
                    s4z_total_execution_constraints += 1
                    for h in range(HOURS):
                        zone_run_cum = gp.quicksum(
                            AI_BATCH_CUM[p][h] for p in provinces_by_zone[ze]
                        )
                        zone_arrival_cum = gp.quicksum(
                            AI_RT_ZONE_ROUTE[gid][zs][ze]
                            * (
                                float(runtime_cluster_profile_cum[gid][h])
                                / AI_BATCH_CUM_SCALE
                            )
                            for gid in AI_RT_ZONE_ROUTE
                            for zs in AI_RT_ZONE_ROUTE[gid]
                            if ze in AI_RT_ZONE_ROUTE[gid][zs]
                        )
                        model.addConstr(
                            zone_run_cum <= zone_arrival_cum,
                            name=f"AI_S4Z_no_advance_{ze}_{h}",
                        )
                        s4z_no_advance_constraints += 1
                        due_idx = h - ai_batch_delay_hours
                        if due_idx >= 0:
                            zone_deadline_cum = gp.quicksum(
                                AI_RT_ZONE_ROUTE[gid][zs][ze]
                                * (
                                    float(runtime_cluster_profile_cum[gid][due_idx])
                                    / AI_BATCH_CUM_SCALE
                                )
                                for gid in AI_RT_ZONE_ROUTE
                                for zs in AI_RT_ZONE_ROUTE[gid]
                                if ze in AI_RT_ZONE_ROUTE[gid][zs]
                            )
                            model.addConstr(
                                zone_run_cum >= zone_deadline_cum,
                                name=f"AI_S4Z_deadline_{ze}_{h}",
                            )
                            s4z_deadline_constraints += 1

            else:
                s4z_origin_balance_constraints = 0
                s4z_total_execution_constraints = 0
                s4z_no_advance_constraints = 0
                s4z_deadline_constraints = 0
                s4z_allowed_route_count = 0
                for dest in destination_provinces:
                    incoming_clusters = clusters_by_destination[dest]
                    # Siting-only (non-shiftable) destination inflow runs immediately
                    # via the shared cluster profile; only the runtime subset is
                    # deadline-shifted through AI_BATCH_RUN.
                    runtime_planned_dest = gp.quicksum(
                        AI_RT_OD_FLOW[gid][dest] for gid in incoming_clusters
                    )
                    # Total batch execution must equal the runtime plan (not the
                    # full siting plan).
                    model.addConstr(
                        AI_BATCH_CUM[dest][HOURS - 1]
                        == runtime_planned_dest * HOURS / AI_BATCH_CUM_SCALE,
                        name=f"AI_S4_OD_destination_total_execution_scaled_{dest}",
                    )
                    for h in range(HOURS):
                        run_cum = AI_BATCH_CUM[dest][h]
                        # No-advance / deadline use the RUNTIME arrival cumulative
                        # profile (runtime flow scaled by the shared cum profile).
                        arrival_cum = gp.quicksum(
                            AI_RT_OD_FLOW[gid][dest]
                            * (
                                float(runtime_cluster_profile_cum[gid][h])
                                / AI_BATCH_CUM_SCALE
                            )
                            for gid in incoming_clusters
                        )
                        model.addConstr(
                            run_cum <= arrival_cum,
                            name=f"AI_S4_OD_no_advance_{dest}_{h}",
                        )
                        due_idx = h - ai_batch_delay_hours
                        if due_idx >= 0:
                            deadline_cum = gp.quicksum(
                                AI_RT_OD_FLOW[gid][dest]
                                * (
                                    float(runtime_cluster_profile_cum[gid][due_idx])
                                    / AI_BATCH_CUM_SCALE
                                )
                                for gid in incoming_clusters
                            )
                            model.addConstr(
                                run_cum >= deadline_cum,
                                name=f"AI_S4_OD_deadline_{dest}_{h}",
                            )

            for dest in destination_provinces:
                incoming_clusters = clusters_by_destination[dest]
                for h in range(HOURS):
                    fixed_gw = (
                        float(fixed_ai_load.get(dest, np.zeros(HOURS))[h])
                        / 1000.0
                    )
                    # Siting-only immediate injection at hour h.
                    siting_only_gw = gp.quicksum(
                        (AI_OD_FLOW[gid][dest] - AI_RT_OD_FLOW[gid][dest])
                        * float(cluster_profile[gid][h])
                        for gid in incoming_clusters
                    )
                    model.addConstr(
                        fixed_gw + siting_only_gw + AI_BATCH_RUN[dest][h]
                        <= float(destination_power_cap_gw[dest]),
                        name=f"AI_S4_OD_destination_power_cap_{dest}_{h}",
                    )
            ai_interface_metadata["s4_od_constraint_status"] = {
                **s4_od_planning_status,
                "destination_hosting_share_constraints": (
                    s4_destination_network_share_status
                ),
                "destination_network_share_constraints": (
                    s4_destination_network_share_status
                ),
                "destination_total_execution_constraints": (
                    0 if ai_s4_zone_runtime_transfer_active else len(destination_provinces)
                ),
                "zone_runtime_transfer_enabled": bool(ai_s4_zone_runtime_transfer),
                "zone_runtime_transfer_active": bool(
                    ai_s4_zone_runtime_transfer_active
                ),
                "zone_runtime_origin_balance_constraints": int(
                    s4z_origin_balance_constraints
                ),
                "zone_total_execution_constraints": int(
                    s4z_total_execution_constraints
                ),
                "destination_power_cap_constraints": int(
                    len(destination_provinces) * HOURS
                ),
                "no_advance_constraints": int(
                    s4z_no_advance_constraints
                    if ai_s4_zone_runtime_transfer_active
                    else len(destination_provinces) * HOURS
                ),
                "deadline_constraints": int(
                    s4z_deadline_constraints
                    if ai_s4_zone_runtime_transfer_active
                    else len(destination_provinces)
                    * max(0, HOURS - ai_batch_delay_hours)
                ),
                "local_execution_retention_constraints": 0,
                "local_execution_retention_note": (
                    "Skipped for S4-OD because destination total execution "
                    "already equals planned_dest * HOURS; any meaningful "
                    "local-retention rule should be expressed at the OD-flow layer."
                ),
                "total_execution_energy_constraints": 0,
                "allowed_od_arc_count": int(
                    sum(len(v) for v in allowed_destinations_by_cluster.values())
                ),
                "allowed_runtime_zone_route_count": int(s4z_allowed_route_count),
                "runtime_zone_route_mode": ai_rt_zone_route_mode,
                "temporal_coupling": (
                    "zone-level runtime route with provincial execution"
                    if ai_s4_zone_runtime_transfer_active
                    else "destination-specific OD arrival profile"
                ),
            }

    for pro in Province:
        if np.sum(pv_cap[pro]) == 0:
            pv_cell_installed_cap_cell2[pro] = np.zeros(int(pv_cell[pro]))
            continue
        if (install_cap_Solar[pro] / np.sum(pv_cap[pro])) >= 1:
            pv_cell_installed_cap_cell2[pro] = np.array(pv_cap[pro])
        else:
            pv_cell_installed_cap_cell2[pro] = (
                install_cap_Solar[pro] / int(pv_cell[pro]) * np.ones(int(pv_cell[pro]))
            )
        pv_cell_installed_cap_cell2[pro] = np.minimum(
            pv_cell_installed_cap_cell2[pro], np.array(pv_cap[pro])
        )

    model.addConstrs(
        (var_solar_cap[pro][c] >= pv_cell_installed_cap_cell2[pro][c])
        for pro in Province
        for c in range(int(pv_cell[pro]))
    )

    model.addConstrs(
        (var_solar_cap[pro][c] <= pv_cap[pro][c])
        for pro in Province
        for c in range(int(pv_cell[pro]))
    )

    INITIAL_SOC = 0.2

    model.addConstrs(
        tot_energy_phs[pro][0] == INITIAL_SOC * energy_phs[pro]
        for pro in Province
    )
    model.addConstrs(
        tot_energy_bat[pro][0] == INITIAL_SOC * energy_bat[pro]
        for pro in Province
    )
    model.addConstrs(
        tot_energy_h2[pro][0]
        == INITIAL_SOC * (energy_h2[pro] + PRO_under_h2_CAP[pro])
        for pro in Province
    )

    model.addConstrs(
        INITIAL_SOC * energy_phs[pro]
        == tot_energy_phs[pro][HOURS - 1]
        - dischar_phs[pro][HOURS - 1]
        - trans_out["PHS"][pro][HOURS - 1]
        + gp.quicksum(
            [
                charge_phs[et][pro][HOURS - 1] * Params.efficiency.phs.charge
                for et in ["Wind", "Solar", "Coal", "Hydro", "Nuclear"]
            ]
        )
        for pro in Province
    )
    model.addConstrs(
        INITIAL_SOC * energy_bat[pro]
        == (1 - Params.self_discharge.bat) * tot_energy_bat[pro][HOURS - 1]
        - dischar_bat[pro][HOURS - 1]
        - trans_out["BAT"][pro][HOURS - 1]
        + gp.quicksum(
            [
                charge_bat[et][pro][HOURS - 1] * Params.efficiency.bat.charge
                for et in ["Wind", "Solar", "Coal", "Hydro", "Nuclear"]
            ]
        )
        for pro in Province
    )
    model.addConstrs(
        INITIAL_SOC * (energy_h2[pro] + PRO_under_h2_CAP[pro])
        == tot_energy_h2[pro][HOURS - 1]
        - dischar_h2[pro][HOURS - 1]
        - trans_out["H2"][pro][HOURS - 1]
        + gp.quicksum(
            [
                charge_h2[et][pro][HOURS - 1] * Params.efficiency.h2.charge
                for et in ["Wind", "Solar", "Coal", "Hydro", "Nuclear"]
            ]
        )
        for pro in Province
    )

    model.addConstrs(
        tot_energy_phs[pro][h]
        == tot_energy_phs[pro][h - 1]
        - dischar_phs[pro][h - 1]
        - trans_out["PHS"][pro][h - 1]
        + gp.quicksum(
            [
                charge_phs[et][pro][h - 1] * Params.efficiency.phs.charge
                for et in ["Wind", "Solar", "Coal", "Hydro", "Nuclear"]
            ]
        )
        for pro in Province
        for h in range(1, HOURS)
    )

    model.addConstrs(
        tot_energy_bat[pro][h]
        == (1 - Params.self_discharge.bat) * tot_energy_bat[pro][h - 1]
        - dischar_bat[pro][h - 1]
        - trans_out["BAT"][pro][h - 1]
        + gp.quicksum(
            [
                charge_bat[et][pro][h - 1] * Params.efficiency.bat.charge
                for et in ["Wind", "Solar", "Coal", "Hydro", "Nuclear"]
            ]
        )
        for pro in Province
        for h in range(1, HOURS)
    )

    model.addConstrs(
        tot_energy_h2[pro][h]
        == tot_energy_h2[pro][h - 1]
        - dischar_h2[pro][h - 1]
        - trans_out["H2"][pro][h - 1]
        + gp.quicksum(
            [
                charge_h2[et][pro][h - 1] * Params.efficiency.h2.charge
                for et in ["Wind", "Solar", "Coal", "Hydro", "Nuclear"]
            ]
        )
        for pro in Province
        for h in range(1, HOURS)
    )

    model.addConstrs(
        dischar_phs[pro][h] + trans_out["PHS"][pro][h] <= tot_energy_phs[pro][h]
        for pro in Province
        for h in range(HOURS)
    )

    model.addConstrs(
        dischar_bat[pro][h] + trans_out["BAT"][pro][h] <= tot_energy_bat[pro][h]
        for pro in Province
        for h in range(HOURS)
    )

    model.addConstrs(
        dischar_h2[pro][h] + trans_out["H2"][pro][h] <= tot_energy_h2[pro][h]
        for pro in Province
        for h in range(HOURS)
    )

    model.addConstrs(
        tot_energy_phs[pro][h] <= energy_phs[pro]
        for pro in Province
        for h in range(HOURS)
    )

    model.addConstrs(
        tot_energy_bat[pro][h] <= energy_bat[pro]
        for pro in Province
        for h in range(HOURS)
    )

    model.addConstrs(
        tot_energy_h2[pro][h] <= energy_h2[pro] + PRO_under_h2_CAP[pro]
        for pro in Province
        for h in range(HOURS)
    )

    model.addConstrs(
        cap_bat[pro] * Params.duration.bat.min <= energy_bat[pro] for pro in Province
    )

    model.addConstrs(
        energy_bat[pro] <= cap_bat[pro] * Params.duration.bat.max for pro in Province
    )

    model.addConstrs(
        energy_phs[pro] <= cap_phs[pro] * Params.duration.hydro.max for pro in Province
    )


    model.addConstrs(
         cap_phs[pro] * Params.duration.hydro.min <= energy_phs[pro] for pro in Province
    )

    model.addConstrs(
        cap_dis_h2[pro] * Params.duration.h2.min
        <= energy_h2[pro] + PRO_under_h2_CAP[pro]
        for pro in Province
    )

    model.addConstrs(
        energy_h2[pro] + PRO_under_h2_CAP[pro]
        <= cap_dis_h2[pro] * Params.duration.h2.max
        for pro in Province
    )

    model.addConstrs(
        gp.quicksum(
            [
                charge_phs[et][pro][h]
                for et in ["Wind", "Solar", "Coal", "Hydro", "Nuclear"]
            ]
        )
        <= cap_phs[pro]
        for pro in Province
        for h in range(HOURS)
    )

    model.addConstrs(
        (dischar_phs[pro][h] + trans_out["PHS"][pro][h]) <= cap_phs[pro]
        for pro in Province
        for h in range(HOURS)
    )

    model.addConstrs(
        gp.quicksum(
            [
                charge_bat[et][pro][h]
                for et in ["Wind", "Solar", "Coal", "Hydro", "Nuclear"]
            ]
        )
        <= cap_bat[pro]
        for pro in Province
        for h in range(HOURS)
    )

    model.addConstrs(
        (dischar_bat[pro][h] + trans_out["BAT"][pro][h]) <= cap_bat[pro]
        for pro in Province
        for h in range(HOURS)
    )
    model.addConstrs(
        gp.quicksum(
            [
                charge_h2[et][pro][h]
                for et in ["Wind", "Solar", "Coal", "Hydro", "Nuclear"]
            ]
        )
        <= cap_ch_h2[pro]
        for pro in Province
        for h in range(HOURS)
    )

    model.addConstrs(
        (dischar_h2[pro][h] + trans_out["H2"][pro][h]) <= cap_dis_h2[pro]
        for pro in Province
        for h in range(HOURS)
    )

    def add_zero_vre_constraints(pro, tech):
        if tech == "Wind":
            model.addConstrs(inter_wind[pro][h] == 0 for h in range(HOURS))
            model.addConstrs(charge_phs["Wind"][pro][h] == 0 for h in range(HOURS))
            model.addConstrs(charge_bat["Wind"][pro][h] == 0 for h in range(HOURS))
            model.addConstrs(charge_h2["Wind"][pro][h] == 0 for h in range(HOURS))
            model.addConstrs(trans_out["Wind"][pro][h] == 0 for h in range(HOURS))
            model.addConstrs(curtail_wind[pro][h] == 0 for h in range(HOURS))
            model.addConstrs(sum_wind_hours[pro][h] == 0 for h in range(HOURS))
            model.addConstrs(
                sum_wind_cell[pro][c] == 0 for c in range(int(wind_cell[pro]))
            )
        elif tech == "Solar":
            model.addConstrs(inter_solar[pro][h] == 0 for h in range(HOURS))
            model.addConstrs(charge_phs["Solar"][pro][h] == 0 for h in range(HOURS))
            model.addConstrs(charge_bat["Solar"][pro][h] == 0 for h in range(HOURS))
            model.addConstrs(charge_h2["Solar"][pro][h] == 0 for h in range(HOURS))
            model.addConstrs(trans_out["Solar"][pro][h] == 0 for h in range(HOURS))
            model.addConstrs(curtail_solar[pro][h] == 0 for h in range(HOURS))
            model.addConstrs(sum_solar_hours[pro][h] == 0 for h in range(HOURS))
            model.addConstrs(
                sum_solar_cell[pro][c] == 0 for c in range(int(pv_cell[pro]))
            )
        else:
            raise ValueError(f"Unsupported zero VRE technology: {tech}")

    for pro in Province:
        if pro in trans_data["in_pro"]:
            if np.sum(wind_cap[pro]) > 0:

                model.addConstrs(
                    inter_wind[pro][h]
                    + charge_phs["Wind"][pro][h]
                    + charge_bat["Wind"][pro][h]
                    + trans_out["Wind"][pro][h]
                    + charge_h2["Wind"][pro][h]
                    + curtail_wind[pro][h]
                    == sum_wind_hours[pro][h]
                    for h in range(HOURS)
                )
                model.addConstrs(
                    inter_wind[pro][h]
                    + charge_phs["Wind"][pro][h]
                    + charge_bat["Wind"][pro][h]
                    + trans_out["Wind"][pro][h]
                    + charge_h2["Wind"][pro][h]
                    >= sum_wind_hours[pro][h] * vre_min_utilization
                    for h in range(HOURS)
                )
            else:
                add_zero_vre_constraints(pro, "Wind")
            if np.sum(pv_cap[pro]) > 0:

                model.addConstrs(
                    inter_solar[pro][h]
                    + charge_phs["Solar"][pro][h]
                    + charge_bat["Solar"][pro][h]
                    + trans_out["Solar"][pro][h]
                    + charge_h2["Solar"][pro][h]
                    + curtail_solar[pro][h]
                    == sum_solar_hours[pro][h]
                    for h in range(HOURS)
                )
                model.addConstrs(
                    inter_solar[pro][h]
                    + charge_phs["Solar"][pro][h]
                    + charge_bat["Solar"][pro][h]
                    + trans_out["Solar"][pro][h]
                    + charge_h2["Solar"][pro][h]
                    >= sum_solar_hours[pro][h] * vre_min_utilization
                    for h in range(HOURS)
                )
            else:
                add_zero_vre_constraints(pro, "Solar")
            ac_out_pro = get_ac_out_neighbors(trans_data, pro)
            dc_out_pro = get_dc_out_neighbors(trans_data, pro)
            model.addConstrs(
                gp.quicksum(
                    load_trans_AC[pro, dst][h]
                    + load_trans_AC_installed[pro, dst][h]
                    for dst in ac_out_pro
                )
                + gp.quicksum(
                    load_trans_DC[pro, dst][h]
                    + load_trans_DC_installed[pro, dst][h]
                    for dst in dc_out_pro
                )
                == trans_out["Wind"][pro][h]
                + trans_out["Solar"][pro][h]
                + trans_out["Coal"][pro][h]
                + trans_out["Hydro"][pro][h]
                + trans_out["Nuclear"][pro][h]
                + trans_out["PHS"][pro][h] * Params.efficiency.phs.discharge
                + trans_out["BAT"][pro][h] * Params.efficiency.bat.discharge
                + trans_out["H2"][pro][h] * Params.efficiency.h2.discharge
                for h in range(HOURS)
            )
        else:
            if np.sum(wind_cap[pro]) > 0:
                model.addConstrs(
                    inter_wind[pro][h]
                    + charge_phs["Wind"][pro][h]
                    + charge_bat["Wind"][pro][h]
                    + charge_h2["Wind"][pro][h]
                    >= sum_wind_hours[pro][h] * vre_min_utilization
                    for h in range(HOURS)
                )
                model.addConstrs(
                    inter_wind[pro][h]
                    + charge_phs["Wind"][pro][h]
                    + charge_bat["Wind"][pro][h]
                    + charge_h2["Wind"][pro][h]
                    + curtail_wind[pro][h]
                    == sum_wind_hours[pro][h]
                    for h in range(HOURS)
                )
            else:
                add_zero_vre_constraints(pro, "Wind")
            if np.sum(pv_cap[pro]) > 0:
                model.addConstrs(
                    inter_solar[pro][h]
                    + charge_phs["Solar"][pro][h]
                    + charge_bat["Solar"][pro][h]
                    + charge_h2["Solar"][pro][h]
                    >= sum_solar_hours[pro][h] * vre_min_utilization
                    for h in range(HOURS)
                )
                model.addConstrs(
                    inter_solar[pro][h]
                    + charge_phs["Solar"][pro][h]
                    + charge_bat["Solar"][pro][h]
                    + charge_h2["Solar"][pro][h]
                    + curtail_solar[pro][h]
                    == sum_solar_hours[pro][h]
                    for h in range(HOURS)
                )
            else:
                add_zero_vre_constraints(pro, "Solar")
            model.addConstrs(trans_out["Wind"][pro][h] == 0 for h in range(HOURS))

            model.addConstrs(trans_out["Solar"][pro][h] == 0 for h in range(HOURS))

            model.addConstrs(trans_out["Coal"][pro][h] == 0 for h in range(HOURS))

            model.addConstrs(trans_out["Hydro"][pro][h] == 0 for h in range(HOURS))

            model.addConstrs(trans_out["Nuclear"][pro][h] == 0 for h in range(HOURS))

            model.addConstrs(trans_out["PHS"][pro][h] == 0 for h in range(HOURS))

            model.addConstrs(trans_out["BAT"][pro][h] == 0 for h in range(HOURS))

            model.addConstrs(trans_out["H2"][pro][h] == 0 for h in range(HOURS))

    def ai_flexible_power_rhs(pro, h):
        """Flexible AI load entering the provincial power balance in GW."""
        if use_external_ai_load:
            if ai_operational_scenario == "S1":
                if not AI_OD_FLOW:
                    raise RuntimeError(
                        "S1 requires AI_OD_FLOW. Refusing to fall back to constant S1."
                    )
                incoming_clusters = ai_s4_od_params["clusters_by_destination"][pro]
                return gp.quicksum(
                    AI_OD_FLOW[gid][pro]
                    * float(ai_s4_od_params["cluster_profile"][gid][h])
                    for gid in incoming_clusters
                )
            if ai_operational_scenario == "S4":
                if not AI_OD_FLOW:
                    raise RuntimeError("S4 must be S4-OD and requires AI_OD_FLOW.")
                # Phase 2b: siting-only (non-shiftable) inflow is injected
                # immediately via the shared cluster profile; only the runtime
                # subset is executed through AI_BATCH_RUN. siting_only = siting
                # OD flow minus runtime OD flow.
                incoming_clusters = ai_s4_od_params["clusters_by_destination"][pro]
                siting_only_gw = gp.quicksum(
                    (AI_OD_FLOW[gid][pro] - AI_RT_OD_FLOW[gid][pro])
                    * float(ai_s4_od_params["cluster_profile"][gid][h])
                    for gid in incoming_clusters
                )
                return siting_only_gw + AI_BATCH_RUN[pro][h]
            return 0.0
        return PRO_AI_LOAD[pro]

    dem = {}
    for pro in Province:
        if pro in trans_data["out_pro"]:
            ac_in_pro = get_ac_in_neighbors(trans_data, pro)
            dc_in_pro = get_dc_in_neighbors(trans_data, pro)
            dem[pro] = model.addConstrs(
                load_conv["Coal"][pro][h]
                + load_conv["Hydro"][pro][h]
                + load_conv["Nuclear"][pro][h]
                + load_conv["BECCS"][pro][h]
                + load_conv["GAS"][pro][h]
                + gp.quicksum(
                    (1 - trans_data["AC"]["loss"].get((src, pro), 1))
                    * load_trans_AC[src, pro][h]
                    + (1 - trans_data["AC_installed"]["loss"].get((src, pro), 1))
                    * load_trans_AC_installed[src, pro][h]
                    for src in ac_in_pro
                )
                + gp.quicksum(
                    (1 - trans_data["DC"]["loss"].get((src, pro), 1))
                    * load_trans_DC[src, pro][h]
                    + (1 - trans_data["DC_installed"]["loss"].get((src, pro), 1))
                    * load_trans_DC_installed[src, pro][h]
                    for src in dc_in_pro
                )
                + inter_wind[pro][h]
                + inter_solar[pro][h]
                + load_shedding[pro][h]
                + Params.efficiency.bat.discharge * dischar_bat[pro][h]
                + Params.efficiency.phs.discharge * dischar_phs[pro][h]
                + Params.efficiency.h2.discharge * dischar_h2[pro][h]
                == (LOAD_DEMAND[pro][h]) / 1000 + ai_flexible_power_rhs(pro, h)
                for h in range(HOURS)
            )

        else:
            dem[pro] = model.addConstrs(
                load_conv["Coal"][pro][h]
                + load_conv["Hydro"][pro][h]
                + load_conv["Nuclear"][pro][h]
                + load_conv["BECCS"][pro][h]
                + load_conv["GAS"][pro][h]
                + inter_wind[pro][h]
                + inter_solar[pro][h]
                + load_shedding[pro][h]
                + Params.efficiency.bat.discharge * dischar_bat[pro][h]
                + Params.efficiency.phs.discharge * dischar_phs[pro][h]
                + Params.efficiency.h2.discharge * dischar_h2[pro][h]
                == (LOAD_DEMAND[pro][h]) / 1000 + ai_flexible_power_rhs(pro, h)
                for h in range(HOURS)
            )

    for pro_pair in trans_data["all_pair"]:
        model.addConstrs(
            load_trans_AC[(pro_pair[0], pro_pair[1])][h]
            + load_trans_AC[(pro_pair[1], pro_pair[0])][h]
            <= trans_cap_AC[pro_pair] * transmission_utilization_limit
            for h in range(HOURS)
        )

        model.addConstrs(
            load_trans_DC[(pro_pair[0], pro_pair[1])][h]
            <= trans_cap_DC[pro_pair] * transmission_utilization_limit
            for h in range(HOURS)
        )

        model.addConstrs(
            load_trans_DC_installed[(pro_pair[0], pro_pair[1])][h]
            <= (
                trans_data["DC_installed"]["cap"][(pro_pair[0], pro_pair[1])]
                + trans_cap_DC_installed_expansion[pro_pair]
            )
            * transmission_utilization_limit
            for h in range(HOURS)
        )

        model.addConstrs(
            load_trans_AC_installed[(pro_pair[0], pro_pair[1])][h]
            + load_trans_AC_installed[(pro_pair[1], pro_pair[0])][h]
            <= (
                trans_data["AC_installed"]["cap"][(pro_pair[0], pro_pair[1])]
                + trans_cap_AC_installed_expansion[pro_pair]
            )
            * transmission_utilization_limit
            for h in range(HOURS)
        )

    model.addConstrs(
        load_conv["Coal"][pro][h]
        + trans_out["Coal"][pro][h]
        + charge_phs["Coal"][pro][h]
        + charge_bat["Coal"][pro][h]
        + charge_h2["Coal"][pro][h]
        <= (install_cap_coal[pro] + new_install_coal[pro])
        for pro in Province
        for h in range(HOURS)
    )

    model.addConstrs(
        load_conv["Nuclear"][pro][h]
        + trans_out["Nuclear"][pro][h]
        + charge_phs["Nuclear"][pro][h]
        + charge_bat["Nuclear"][pro][h]
        + charge_h2["Nuclear"][pro][h]
        <= install_cap_Nuclear[pro]
        for pro in Province
        for h in range(HOURS)
    )

    model.addConstrs(
        load_conv["Nuclear"][pro][h]
        + trans_out["Nuclear"][pro][h]
        + charge_phs["Nuclear"][pro][h]
        + charge_bat["Nuclear"][pro][h]
        + charge_h2["Nuclear"][pro][h]
        >= install_cap_Nuclear[pro] * 0.7
        for pro in Province
        for h in range(HOURS)
    )

    model.addConstrs(
        load_conv["Hydro"][pro][h]
        + trans_out["Hydro"][pro][h]
        + charge_phs["Hydro"][pro][h]
        + charge_bat["Hydro"][pro][h]
        + charge_h2["Hydro"][pro][h]
        <= install_cap_hydro[pro]
        for pro in Province
        for h in range(HOURS)
    )

    model.addConstrs(
        (
            gp.quicksum(
                load_conv["Hydro"][pro][h]
                + trans_out["Hydro"][pro][h]
                + charge_phs["Hydro"][pro][h]
                + charge_bat["Hydro"][pro][h]
                + charge_h2["Hydro"][pro][h]
                for h in range(HOURS)
            )
            <= install_cap_hydro[pro]
            * 8760.0
            * HORIZON_SCALE
            * float(hydro_cf_max_by_province[str(pro)])
            for pro in Province
        ),
        name="Hydro_annual_CF_max",
    )

    model.addConstrs(
        load_conv["BECCS"][pro][h] <= install_cap_Bios[pro]
        for pro in Province
        for h in range(HOURS)
    )

    model.addConstrs(
        load_conv["GAS"][pro][h] <= install_cap_Other[pro]
        for pro in Province
        for h in range(HOURS)
    )

    for pro in Province:
        for h in range(1, HOURS):
            model.addConstr(
                ru["Coal"][pro][h]
                <= ru_conf["Coal"] * (install_cap_coal[pro] + new_install_coal[pro])
            )

            model.addConstr(
                ru["Hydro"][pro][h] <= ru_conf["Hydro"] * install_cap_hydro[pro]
            )
            model.addConstr(
                ru["Nuclear"][pro][h] <= ru_conf["Nuclear"] * install_cap_Nuclear[pro]
            )
            model.addConstr(
                ru["BECCS"][pro][h] <= ru_conf["BECCS"] * install_cap_Bios[pro]
            )
            model.addConstr(
                ru["GAS"][pro][h] <= ru_conf["GAS"] * install_cap_Other[pro]
            )

            model.addConstr(
                rd["Coal"][pro][h]
                <= rd_conf["Coal"] * (install_cap_coal[pro] + new_install_coal[pro])
            )
            model.addConstr(
                rd["Hydro"][pro][h] <= rd_conf["Hydro"] * install_cap_hydro[pro]
            )
            model.addConstr(
                rd["Nuclear"][pro][h] <= rd_conf["Nuclear"] * install_cap_Nuclear[pro]
            )
            model.addConstr(
                rd["BECCS"][pro][h] <= rd_conf["BECCS"] * install_cap_Bios[pro]
            )
            model.addConstr(
                rd["GAS"][pro][h] <= rd_conf["GAS"] * install_cap_Other[pro]
            )
    model.addConstrs(
        (
            ru["Coal"][pro][h]
            >= load_conv["Coal"][pro][h]
            + charge_phs["Coal"][pro][h]
            + charge_bat["Coal"][pro][h]
            + charge_h2["Coal"][pro][h]
            + trans_out["Coal"][pro][h]
            - load_conv["Coal"][pro][h - 1]
            - charge_phs["Coal"][pro][h - 1]
            - charge_bat["Coal"][pro][h - 1]
            - charge_h2["Coal"][pro][h - 1]
            - trans_out["Coal"][pro][h - 1]
            for pro in Province
            for h in range(1, HOURS)
        ),
        name="Coal_ru",
    )

    model.addConstrs(
        (
            rd["Coal"][pro][h]
            >= load_conv["Coal"][pro][h - 1]
            + charge_phs["Coal"][pro][h - 1]
            + charge_bat["Coal"][pro][h - 1]
            + charge_h2["Coal"][pro][h - 1]
            + trans_out["Coal"][pro][h - 1]
            - load_conv["Coal"][pro][h]
            - charge_phs["Coal"][pro][h]
            - charge_bat["Coal"][pro][h]
            - charge_h2["Coal"][pro][h]
            - trans_out["Coal"][pro][h]
            for pro in Province
            for h in range(1, HOURS)
        ),
        name="Coal_rd",
    )

    model.addConstrs(
        (
            ru["Hydro"][pro][h]
            >= load_conv["Hydro"][pro][h]
            + charge_phs["Hydro"][pro][h]
            + charge_bat["Hydro"][pro][h]
            + charge_h2["Hydro"][pro][h]
            + trans_out["Hydro"][pro][h]
            - load_conv["Hydro"][pro][h - 1]
            - charge_phs["Hydro"][pro][h - 1]
            - charge_bat["Hydro"][pro][h - 1]
            - charge_h2["Hydro"][pro][h - 1]
            - trans_out["Hydro"][pro][h - 1]
            for pro in Province
            for h in range(1, HOURS)
        ),
        name="Hydro_ru",
    )

    model.addConstrs(
        (
            rd["Hydro"][pro][h]
            >= load_conv["Hydro"][pro][h - 1]
            + charge_phs["Hydro"][pro][h - 1]
            + charge_bat["Hydro"][pro][h - 1]
            + charge_h2["Hydro"][pro][h - 1]
            + trans_out["Hydro"][pro][h - 1]
            - load_conv["Hydro"][pro][h]
            - (
                charge_phs["Hydro"][pro][h]
                + charge_bat["Hydro"][pro][h]
                + charge_h2["Hydro"][pro][h]
            )
            - trans_out["Hydro"][pro][h]
            for pro in Province
            for h in range(1, HOURS)
        ),
        name="Hydro_rd",
    )

    model.addConstrs(
        (
            ru["Nuclear"][pro][h]
            >= load_conv["Nuclear"][pro][h]
            + charge_phs["Nuclear"][pro][h]
            + charge_bat["Nuclear"][pro][h]
            + charge_h2["Nuclear"][pro][h]
            + trans_out["Nuclear"][pro][h]
            - (
                charge_phs["Nuclear"][pro][h - 1]
                + charge_bat["Nuclear"][pro][h - 1]
                + charge_h2["Nuclear"][pro][h - 1]
            )
            - load_conv["Nuclear"][pro][h - 1]
            - trans_out["Nuclear"][pro][h - 1]
            for pro in Province
            for h in range(1, HOURS)
        ),
        name="Nuclear_ru",
    )

    model.addConstrs(
        (
            rd["Nuclear"][pro][h]
            >= load_conv["Nuclear"][pro][h - 1]
            + charge_phs["Nuclear"][pro][h - 1]
            + charge_bat["Nuclear"][pro][h - 1]
            + charge_h2["Nuclear"][pro][h - 1]
            + trans_out["Nuclear"][pro][h - 1]
            - (
                charge_phs["Nuclear"][pro][h]
                + charge_bat["Nuclear"][pro][h]
                + charge_h2["Nuclear"][pro][h]
            )
            - load_conv["Nuclear"][pro][h]
            - trans_out["Nuclear"][pro][h]
            for pro in Province
            for h in range(1, HOURS)
        ),
        name="Nuclear_rd",
    )

    model.addConstrs(
        (
            ru["BECCS"][pro][h]
            >= load_conv["BECCS"][pro][h] - load_conv["BECCS"][pro][h - 1]
            for pro in Province
            for h in range(1, HOURS)
        ),
        name="BE_ru",
    )

    model.addConstrs(
        (
            rd["BECCS"][pro][h]
            >= load_conv["BECCS"][pro][h - 1] - load_conv["BECCS"][pro][h]
            for pro in Province
            for h in range(1, HOURS)
        ),
        name="BE_rd",
    )

    model.addConstrs(
        (
            ru["GAS"][pro][h] >= load_conv["GAS"][pro][h] - load_conv["GAS"][pro][h - 1]
            for pro in Province
            for h in range(1, HOURS)
        ),
        name="GAS_du",
    )
    model.addConstrs(
        (
            rd["GAS"][pro][h] >= load_conv["GAS"][pro][h - 1] - load_conv["GAS"][pro][h]
            for pro in Province
            for h in range(1, HOURS)
        ),
        name="GAS_rd",
    )

    for pro in Province:

        # 保留原逐时削减约束（按需可恢复）
        # model.addConstrs(
        #     load_shedding[pro][h] <= Params.shedding_conf[pro] * LOAD_DEMAND[pro][h]
        #     for h in range(HOURS)
        # )

        # 分省年度总削减上限：<= 平均每小时需求 * 2.4小时
        base_fixed_avg_gw = np.sum(LOAD_DEMAND[pro]) / HOURS / 1000.0
        if use_external_ai_load:
            if ai_operational_scenario == "S0":
                flexible_avg_gw_expr = 0.0
            elif ai_operational_scenario == "S1":
                flexible_avg_gw_expr = PRO_AI_LOAD[pro]
            elif ai_operational_scenario == "S4":
                flexible_avg_gw_expr = (
                    gp.quicksum(AI_BATCH_RUN[pro][h] for h in range(HOURS)) / HOURS
                )
            else:
                flexible_avg_gw_expr = 0.0
        else:
            flexible_avg_gw_expr = PRO_AI_LOAD[pro]
        model.addConstr(
            gp.quicksum(load_shedding[pro][h] for h in range(HOURS))
            <= 2.4 * (base_fixed_avg_gw + flexible_avg_gw_expr),
            name=f"shedding_total_cap_{pro}",
        )

    from datetime import timedelta
    from datetime import datetime
    import time

    def format_duration(seconds):
        """将秒数转换为 天_小时_分钟 格式"""
        td = timedelta(seconds=seconds)
        days = td.days
        seconds_remaining = td.seconds
        hours = seconds_remaining // 3600
        minutes = (seconds_remaining % 3600) // 60
        return f"模型求解时间：{days}天_{hours}小时_{minutes}分钟"

    start_time = time.time()

    now = datetime.now()
    formatted_time = now.strftime("%Y_%m_%d_%H_%M")

    # lark_callback.send_msg(
    #     content=f"模型{Params.Hours}_raw_coal_开始求解，当前时间： {formatted_time}",  # 通知内容
    # )

    print(f"当前设置的MIPGap参数值为: {model.Params.MIPGap}")

    if use_external_ai_load:
        print("AI scenario summary:")
        print(f"  scenario: {ai_operational_scenario}")
        print(
            "  effective_ai_scenario: "
            f"{ai_interface_metadata.get('effective_ai_scenario')}"
        )
        print(f"  use_source_cluster_ai_interface: {use_source_cluster_ai_interface}")
        print(f"  strict_cluster_ai_interface: {strict_cluster_ai_interface}")
        print(f"  s0_use_cluster_reconstruction: {s0_use_cluster_reconstruction}")
        print(f"  AI_OD_FLOW active: {bool(AI_OD_FLOW)}")
        print(f"  AI_BATCH_CUM active: {bool(AI_BATCH_CUM)}")
        print(f"  flexible pool GW: {hourly_AI_load}")

    if use_external_ai_load and strict_cluster_ai_interface:
        if ai_operational_scenario not in {"S0", "S1", "S4"}:
            raise RuntimeError(f"Unexpected scenario: {ai_operational_scenario}")
        if not use_source_cluster_ai_interface:
            raise RuntimeError(
                "STRICT_CLUSTER_AI_INTERFACE=true but source-cluster interface is not loaded."
            )
        if ai_operational_scenario == "S0" and not s0_use_cluster_reconstruction:
            raise RuntimeError(
                "S0 must use source-cluster reconstruction under strict mode."
            )
        if ai_operational_scenario in {"S1", "S4"} and not AI_OD_FLOW:
            raise RuntimeError(f"{ai_operational_scenario} requires non-empty AI_OD_FLOW.")
        if ai_operational_scenario == "S4" and not AI_BATCH_CUM:
            raise RuntimeError("S4-OD requires AI_BATCH_CUM variables.")

    model.update()

    print("Original model statistics:")
    model.printStats()

    if os.environ.get("PRINT_PRESOLVED_STATS", "0") == "1":
        print("Presolved model statistics:")
        presolved_model = model.presolve()
        presolved_model.printStats()

    model.optimize()

    if model.SolCount > 0:
        print("Solution quality:")
        model.printQuality()
    else:
        print("Solution quality: no solution available to print.")

    # Always surface the terminal solver status (no more silent exits).
    _status_name_map = {
        GRB.OPTIMAL: "OPTIMAL", GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE", GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED", GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED", GRB.NUMERIC: "NUMERIC",
    }
    print(
        f"[SOLVER] terminal status={int(model.status)} "
        f"({_status_name_map.get(model.status, 'OTHER')}), "
        f"SolCount={model.SolCount}"
    )

    # Check for infeasibility and compute IIS
    if model.status == gp.GRB.INFEASIBLE:
        print("Model is infeasible. Computing IIS to locate conflicting constraints...")
        model.computeIIS()
        model.write("model_iis.ilp")
        print("Irreducible Inconsistent Subsystem (IIS) written to model_iis.ilp")
        print("The following constraints are part of the IIS:")
        for c in model.getConstrs():
            if c.IISConstr:
                print(f" - {c.constrName}")
        return {}
    elif model.status == gp.GRB.INF_OR_UNBD:
        print("Model is Infeasible or Unbounded. Setting DualReductions=0 to confirm...")
        model.setParam("DualReductions", 0)
        model.optimize()
        if model.status == gp.GRB.INFEASIBLE:
            print("Model is infeasible. Computing IIS...")
            model.computeIIS()
            model.write("model_iis.ilp")
            print("IIS written to model_iis.ilp")
            for c in model.getConstrs():
                if c.IISConstr:
                    print(f" - {c.constrName}")
        return {}

    end_time = time.time()
    duration = end_time - start_time
    formatted_duration = format_duration(duration)
    print(f" method 3 模型开始求解，当前时间： {formatted_duration}")

    now = datetime.now()
    formatted_time = now.strftime("%Y_%m_%d_%H_%M")
    # data save

    write_lp_debug = str_to_bool(
        os.environ.get("WRITE_LP_DEBUG", "false"),
        default=False,
    )
    if write_lp_debug:
        debug_dir = os.path.join(args.output_dir, args.Mode)
        os.makedirs(debug_dir, exist_ok=True)
        model.write(os.path.join(debug_dir, "model_check_detail.lp"))

    acceptable_status = {
        GRB.OPTIMAL,
        GRB.SUBOPTIMAL,
    }
    # Phase C fix: also accept solutions that terminated early (TIME_LIMIT,
    # INTERRUPTED) but still carry a usable feasible solution.
    # NUMERIC is deliberately excluded because a feasible incumbent from a
    # numerical-failure termination is not reliable enough for default outputs.
    # These are flagged non-strict in solver_audit so downstream can decide.
    usable_with_feasible_solution = {
        GRB.TIME_LIMIT,
        GRB.INTERRUPTED,
        GRB.SUBOPTIMAL,
    }
    has_usable_solution = (
        model.status in acceptable_status
        or (model.status in usable_with_feasible_solution and model.SolCount > 0)
    )

    if has_usable_solution and model.SolCount > 0:
        if model.status != GRB.OPTIMAL:
            print(
                f"WARNING: solver status {int(model.status)} "
                f"({_status_name_map.get(model.status, 'OTHER')}) with "
                f"SolCount={model.SolCount}. Writing results from best feasible "
                "solution; treat as non-strict (see solver_audit.solver_strict)."
            )
        if model.status == GRB.SUBOPTIMAL and model.status not in usable_with_feasible_solution:
            print("WARNING: Gurobi returned SUBOPTIMAL. Check solver_audit before using results.")
        os.makedirs(args.output_dir, exist_ok=True)
        res_dir = os.path.join(args.output_dir, args.Mode)
        res_dir_pkl = os.path.join(res_dir, "pkl_data")
        os.makedirs(res_dir_pkl, exist_ok=True)

        os.makedirs(res_dir, exist_ok=True)

        def _safe_model_attr(attr_name, default=None):
            try:
                return getattr(model, attr_name)
            except (AttributeError, gp.GurobiError):
                return default

        def _safe_float(value):
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        solver_status_names = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.NUMERIC: "NUMERIC",
        }
        solver_audit = {
            "extreme_cf": dict(extreme_cf_metadata),
            "status": int(model.status),
            "status_name": solver_status_names.get(model.status, "OTHER"),
            "sol_count": int(model.SolCount),
            "objective": _safe_float(_safe_model_attr("ObjVal")),
            "objective_bound": _safe_float(_safe_model_attr("ObjBound")),
            "mip_gap": _safe_float(_safe_model_attr("MIPGap")),
            "runtime_sec": _safe_float(_safe_model_attr("Runtime")),
            "wall_runtime_sec": float(duration),
            "iter_count": _safe_float(_safe_model_attr("IterCount")),
            "node_count": _safe_float(_safe_model_attr("NodeCount")),
            "bar_iter_count": _safe_float(_safe_model_attr("BarIterCount")),
            "mip_gap_param": _safe_float(model.Params.MIPGap),
            "write_lp_debug": bool(write_lp_debug),
            "solver_strict": bool(model.status == GRB.OPTIMAL),
            "solver_terminal_status": int(model.status),
            "solver_terminal_status_name": solver_status_names.get(
                model.status, "OTHER"
            ),
            "solution_is_best_feasible_not_proven_optimal": bool(
                model.status in {
                    GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.SUBOPTIMAL
                }
            ),
        }
        with open(os.path.join(res_dir, "solver_audit.json"), "w", encoding="utf-8") as f:
            json.dump(solver_audit, f, indent=2, ensure_ascii=False)

        # laod demand
        with open(os.path.join(res_dir_pkl, "load_demand.pkl"), "wb") as f:
            pickle.dump(LOAD_DEMAND, f)
        with open(os.path.join(res_dir_pkl, "non_ai_base_load_mw.pkl"), "wb") as f:
            pickle.dump(non_ai_base_load_mw, f)
        with open(os.path.join(res_dir_pkl, "fixed_ai_load.pkl"), "wb") as f:
            pickle.dump(fixed_ai_load, f)
        with open(os.path.join(res_dir_pkl, "flexible_ai_load.pkl"), "wb") as f:
            pickle.dump(flexible_ai_load, f)
        with open(os.path.join(res_dir_pkl, "ai_hosting_capacity_gw.pkl"), "wb") as f:
            pickle.dump(ai_hosting_capacity_gw, f)
        with open(os.path.join(res_dir_pkl, "ai_hosting_lb_gw.pkl"), "wb") as f:
            pickle.dump(ai_hosting_lb_gw, f)
        with open(os.path.join(res_dir_pkl, "ai_hosting_ub_gw.pkl"), "wb") as f:
            pickle.dump(ai_hosting_ub_gw, f)
        with open(os.path.join(res_dir_pkl, "ai_flexible_load_gw.pkl"), "wb") as f:
            pickle.dump(ai_flexible_load_gw, f)
        with open(os.path.join(res_dir_pkl, "ai_batch_power_cap_gw.pkl"), "wb") as f:
            pickle.dump(ai_batch_power_cap_gw, f)
        if AI_OD_FLOW:
            with open(os.path.join(res_dir_pkl, "ai_s4_od_params.pkl"), "wb") as f:
                pickle.dump(ai_s4_od_params, f)
        if use_external_ai_load and ai_operational_scenario == "S4":
            s4_unmet_runtime_gw = float(
                sum(AI_UNMET_RUNTIME[gid].x for gid in AI_UNMET_RUNTIME)
            ) if AI_UNMET_RUNTIME else 0.0
            s4_planned_energy_gwh = (
                max(
                    0.0,
                    float(ai_s4_od_params.get("runtime_pool_gw", hourly_AI_load))
                    - s4_unmet_runtime_gw,
                )
                * HOURS
            )
            s4_executed_energy_gwh = sum(
                AI_BATCH_RUN[pro][h].x for pro in Province for h in range(HOURS)
            )
            s4_relative_gap = abs(s4_planned_energy_gwh - s4_executed_energy_gwh) / max(
                1e-9, s4_planned_energy_gwh
            )
            ai_interface_metadata["s4_audit"] = {
                "basis": "runtime batch execution only",
                "unmet_runtime_gw": float(s4_unmet_runtime_gw),
                "planned_energy_gwh": float(s4_planned_energy_gwh),
                "executed_energy_gwh": float(s4_executed_energy_gwh),
                "relative_gap": float(s4_relative_gap),
                "zone_runtime_transfer_enabled": bool(ai_s4_zone_runtime_transfer),
            }
            print("S4 audit:")
            print(" planned AI energy (GWh):", s4_planned_energy_gwh)
            print(" executed AI energy (GWh):", s4_executed_energy_gwh)
            print(" relative gap:", s4_relative_gap)
        # new_install_coal
        results_new_install_coal = {}
        results_var_wind_cap = {}
        results_var_solar_cap = {}
        results_x_wind2 = {}
        results_x_solar2 = {}
        results_sum_wind_hours = {}
        results_sum_wind_cell = {}
        results_inter_wind = {}
        results_curtail_wind = {}
        results_sum_solar_hours = {}
        results_sum_solar_cell = {}
        results_inter_solar = {}
        results_curtail_solar = {}
        results_load_shedding = {}
        results_cap_phs = {}
        results_cap_bat = {}
        results_cap_ch_h2 = {}
        results_cap_dis_h2 = {}
        results_energy_bat = {}
        results_energy_phs = {}
        results_energy_h2 = {}
        results_underground_h2 = {}
        results_dischar_phs = {}
        results_dischar_bat = {}
        results_dischar_h2 = {}
        results_tot_energy_h2 = {}
        results_tot_energy_phs = {}
        results_tot_energy_bat = {}
        results_load_trans = {}
        results_load_trans_DC = {}
        results_load_trans_AC = {}
        results_load_trans_DC_installed = {}
        results_load_trans_AC_installed = {}
        results_AI_load = {}
        results_AI_batch_run = {}
        results_AI_batch_cum = {}
        results_AI_od_flow = {}
        results_AI_rt_od_flow = {}
        results_AI_rt_zone_route = {}
        results_AI_zone_batch_run = {}
        results_AI_s1_profiled_load = {}
        results_AI_implied_province_od = {}
        results_AI_implied_province_od_records = []
        results_vre_curtail_summary = {}
        results_hydro_cf_audit = {
            "enabled": True,
            "hours": int(HOURS),
            "horizon_scale": float(HORIZON_SCALE),
            "by_province": {},
        }

        results_ru = {"Coal": {}, "Hydro": {}, "Nuclear": {}, "BECCS": {}, "GAS": {}}
        results_rd = {"Coal": {}, "Hydro": {}, "Nuclear": {}, "BECCS": {}, "GAS": {}}
        results_load_conv = {
            "Coal": {},
            "Hydro": {},
            "Nuclear": {},
            "BECCS": {},
            "GAS": {},
        }

        results_trans_out = {
            "Coal": {},
            "Hydro": {},
            "Nuclear": {},
            "Wind": {},
            "Solar": {},
            "PHS": {},
            "BAT": {},
            "H2": {},
        }

        results_charge_phs = {
            "Coal": {},
            "Hydro": {},
            "Nuclear": {},
            "Wind": {},
            "Solar": {},
        }
        results_charge_bat = {
            "Coal": {},
            "Hydro": {},
            "Nuclear": {},
            "Wind": {},
            "Solar": {},
        }

        results_charge_h2 = {
            "Coal": {},
            "Hydro": {},
            "Nuclear": {},
            "Wind": {},
            "Solar": {},
        }

        gen_category1 = ["Coal", "Hydro", "Nuclear", "BECCS", "GAS"]
        gen_category2 = ["Coal", "Hydro", "Nuclear", "Wind", "Solar"]
        gen_category3 = [
            "Coal",
            "Hydro",
            "Nuclear",
            "Wind",
            "Solar",
            "PHS",
            "BAT",
            "H2",
        ]
        for pro in Province:
            results_new_install_coal[pro] = new_install_coal[pro].x
            results_var_wind_cap[pro] = np.array(
                [var_wind_cap[pro][c].x for c in range(int(wind_cell[pro]))]
            )
            results_var_solar_cap[pro] = np.array(
                [var_solar_cap[pro][c].x for c in range(int(pv_cell[pro]))]
            )
            results_sum_wind_hours[pro] = np.array(
                [sum_wind_hours[pro][h].x for h in range(HOURS)]
            )
            results_sum_wind_cell[pro] = np.array(
                [sum_wind_cell[pro][c].x for c in range(int(wind_cell[pro]))]
            )
            results_inter_wind[pro] = np.array(
                [inter_wind[pro][h].x for h in range(HOURS)]
            )
            results_curtail_wind[pro] = np.array(
                [curtail_wind[pro][h].x for h in range(HOURS)]
            )
            results_sum_solar_hours[pro] = np.array(
                [sum_solar_hours[pro][h].x for h in range(HOURS)]
            )
            results_sum_solar_cell[pro] = np.array(
                [sum_solar_cell[pro][c].x for c in range(int(pv_cell[pro]))]
            )
            results_inter_solar[pro] = np.array(
                [inter_solar[pro][h].x for h in range(HOURS)]
            )
            results_curtail_solar[pro] = np.array(
                [curtail_solar[pro][h].x for h in range(HOURS)]
            )
            results_load_shedding[pro] = np.array(
                [load_shedding[pro][h].x for h in range(HOURS)]
            )
            results_dischar_phs[pro] = np.array(
                [dischar_phs[pro][h].x for h in range(HOURS)]
            )
            results_dischar_bat[pro] = np.array(
                [dischar_bat[pro][h].x for h in range(HOURS)]
            )
            results_dischar_h2[pro] = np.array(
                [dischar_h2[pro][h].x for h in range(HOURS)]
            )
            results_tot_energy_h2[pro] = np.array(
                [tot_energy_h2[pro][h].x for h in range(HOURS)]
            )
            results_AI_load[pro] = np.array([PRO_AI_LOAD[pro].x])
            results_AI_batch_run[pro] = np.array(
                [AI_BATCH_RUN[pro][h].x for h in range(HOURS)]
            )
            if pro in AI_BATCH_CUM:
                results_AI_batch_cum[pro] = np.array(
                    [AI_BATCH_CUM[pro][h].x for h in range(HOURS)]
                ) * AI_BATCH_CUM_SCALE
            else:
                results_AI_batch_cum[pro] = np.zeros(HOURS)

            results_tot_energy_phs[pro] = np.array(
                [tot_energy_phs[pro][h].x for h in range(HOURS)]
            )
            results_tot_energy_bat[pro] = np.array(
                [tot_energy_bat[pro][h].x for h in range(HOURS)]
            )

            results_cap_phs[pro] = np.array([cap_phs[pro].x])
            results_cap_bat[pro] = np.array([cap_bat[pro].x])
            results_cap_ch_h2[pro] = np.array([cap_ch_h2[pro].x])
            results_cap_dis_h2[pro] = np.array([cap_dis_h2[pro].x])
            results_energy_bat[pro] = np.array([energy_bat[pro].x])
            results_energy_phs[pro] = np.array([energy_phs[pro].x])
            results_energy_h2[pro] = np.array([energy_h2[pro].x])
            results_underground_h2[pro] = np.array([PRO_under_h2_CAP[pro].x])

            for k in gen_category1:
                results_ru[k][pro] = np.array([ru[k][pro][h].x for h in range(HOURS)])
                results_rd[k][pro] = np.array([rd[k][pro][h].x for h in range(HOURS)])
                results_load_conv[k][pro] = np.array(
                    [load_conv[k][pro][h].x for h in range(HOURS)]
                )
            for k in gen_category2:
                results_charge_phs[k][pro] = np.array(
                    [charge_phs[k][pro][h].x for h in range(HOURS)]
                )
                results_charge_bat[k][pro] = np.array(
                    [charge_bat[k][pro][h].x for h in range(HOURS)]
                )
                results_charge_h2[k][pro] = np.array(
                    [charge_h2[k][pro][h].x for h in range(HOURS)]
                )
            for k in gen_category3:
                if pro in trans_data["in_pro"]:
                    results_trans_out[k][pro] = np.array(
                        [trans_out[k][pro][h].x for h in range(HOURS)]
                    )

            hydro_gen_gwh = float(
                sum(
                    load_conv["Hydro"][pro][h].x
                    + trans_out["Hydro"][pro][h].x
                    + charge_phs["Hydro"][pro][h].x
                    + charge_bat["Hydro"][pro][h].x
                    + charge_h2["Hydro"][pro][h].x
                    for h in range(HOURS)
                )
            )
            hydro_full_cf_gwh = float(install_cap_hydro[pro]) * 8760.0 * HORIZON_SCALE
            hydro_energy_cap_gwh = hydro_full_cf_gwh * float(
                hydro_cf_max_by_province[str(pro)]
            )
            hydro_actual_cf = (
                hydro_gen_gwh / hydro_full_cf_gwh
                if hydro_full_cf_gwh > 1e-12
                else 0.0
            )
            results_hydro_cf_audit["by_province"][str(pro)] = {
                "hydro_generation_gwh": hydro_gen_gwh,
                "hydro_capacity_gw": float(install_cap_hydro[pro]),
                "hydro_cf_max": float(hydro_cf_max_by_province[str(pro)]),
                "hydro_actual_cf": float(hydro_actual_cf),
                "hydro_full_cf_gwh": float(hydro_full_cf_gwh),
                "hydro_energy_cap_gwh": float(hydro_energy_cap_gwh),
                "binding_ratio": (
                    float(hydro_gen_gwh / hydro_energy_cap_gwh)
                    if hydro_energy_cap_gwh > 1e-12
                    else None
                ),
            }

            num_cell_wind = int(wind_cell[pro])
            num_cell_solar = int(pv_cell[pro])
            wind_caps = np.array(
                [var_wind_cap[pro][c].x for c in range(num_cell_wind)],
                dtype=float,
            )
            solar_caps = np.array(
                [var_solar_cap[pro][c].x for c in range(num_cell_solar)],
                dtype=float,
            )
            if num_cell_wind > 0:
                wind_cf_out = (
                    np.asarray(wind_cf[pro], dtype=float)
                    .reshape(num_cell_wind, -1)[:, :HOURS]
                )
                results_x_wind2[pro] = wind_caps[:, None] * wind_cf_out
            else:
                results_x_wind2[pro] = np.zeros((0, HOURS))
            if num_cell_solar > 0:
                solar_cf_out = (
                    np.asarray(pv_cf[pro], dtype=float)
                    .reshape(num_cell_solar, -1)[:, :HOURS]
                )
                results_x_solar2[pro] = solar_caps[:, None] * solar_cf_out
            else:
                results_x_solar2[pro] = np.zeros((0, HOURS))

            if pro in trans_data["in_pro"]:
                for pro1 in get_ac_in_neighbors(trans_data, pro):
                    results_load_trans_AC[f"{pro1}_{pro}"] = np.array(
                        [load_trans_AC[pro1, pro][h].x for h in range(HOURS)]
                    )
                    results_load_trans_AC_installed[f"{pro1}_{pro}"] = np.array(
                        [load_trans_AC_installed[pro1, pro][h].x for h in range(HOURS)]
                    )
                for pro1 in get_dc_in_neighbors(trans_data, pro):
                    results_load_trans_DC[f"{pro1}_{pro}"] = np.array(
                        [load_trans_DC[pro1, pro][h].x for h in range(HOURS)]
                    )
                    results_load_trans_DC_installed[f"{pro1}_{pro}"] = np.array(
                        [load_trans_DC_installed[pro1, pro][h].x for h in range(HOURS)]
                    )
        if AI_OD_FLOW:
            results_AI_od_flow = {
                gid: {
                    dest: np.array([AI_OD_FLOW[gid][dest].x])
                    for dest in AI_OD_FLOW[gid]
                }
                for gid in AI_OD_FLOW
            }
            # Phase 2b: runtime OD flow results + post-solve subset audit.
            results_AI_rt_od_flow = {
                gid: {
                    dest: np.array([AI_RT_OD_FLOW[gid][dest].x])
                    for dest in AI_RT_OD_FLOW.get(gid, {})
                }
                for gid in AI_RT_OD_FLOW
            }
            results_AI_rt_zone_route = {
                gid: {
                    zs: {
                        ze: np.array([AI_RT_ZONE_ROUTE[gid][zs][ze].x])
                        for ze in AI_RT_ZONE_ROUTE.get(gid, {}).get(zs, {})
                    }
                    for zs in AI_RT_ZONE_ROUTE.get(gid, {})
                }
                for gid in AI_RT_ZONE_ROUTE
            }
            if results_AI_rt_zone_route:
                rt_zone_total_gw = 0.0
                rt_zone_cross_gw = 0.0
                rt_zone_export_by_site_zone = {}
                rt_zone_import_by_exec_zone = {}
                for gid, by_site_zone in results_AI_rt_zone_route.items():
                    for zs, by_exec_zone in by_site_zone.items():
                        for ze, arr in by_exec_zone.items():
                            val = float(arr[0])
                            rt_zone_total_gw += val
                            rt_zone_export_by_site_zone[zs] = (
                                rt_zone_export_by_site_zone.get(zs, 0.0) + val
                            )
                            rt_zone_import_by_exec_zone[ze] = (
                                rt_zone_import_by_exec_zone.get(ze, 0.0) + val
                            )
                            if ze != zs:
                                rt_zone_cross_gw += val
                ai_interface_metadata["s4_zone_runtime_transfer_solution_audit"] = {
                    "enabled": bool(ai_s4_zone_runtime_transfer),
                    "active": bool(ai_s4_zone_runtime_transfer_active),
                    "route_mode": ai_rt_zone_route_mode,
                    "runtime_zone_route_total_gw": float(rt_zone_total_gw),
                    "cross_zone_runtime_route_gw": float(rt_zone_cross_gw),
                    "cross_zone_runtime_route_share": float(
                        rt_zone_cross_gw / rt_zone_total_gw
                        if rt_zone_total_gw > 1e-12 else 0.0
                    ),
                    "by_site_zone_export_gw": rt_zone_export_by_site_zone,
                    "by_exec_zone_import_gw": rt_zone_import_by_exec_zone,
                }
            rt_subset_max_violation_gw = 0.0
            rt_total_gw = 0.0
            siting_total_gw = 0.0
            for gid in AI_OD_FLOW:
                for dest in AI_OD_FLOW[gid]:
                    s_val = float(AI_OD_FLOW[gid][dest].x)
                    r_val = float(
                        AI_RT_OD_FLOW.get(gid, {}).get(dest).x
                    ) if dest in AI_RT_OD_FLOW.get(gid, {}) else 0.0
                    siting_total_gw += s_val
                    rt_total_gw += r_val
                    rt_subset_max_violation_gw = max(
                        rt_subset_max_violation_gw, r_val - s_val
                    )
            ai_interface_metadata["dual_pool_solution_audit"] = {
                "enabled": True,
                "siting_total_gw": float(siting_total_gw),
                "runtime_total_gw": float(rt_total_gw),
                "runtime_over_siting_ratio": float(
                    rt_total_gw / siting_total_gw if siting_total_gw > 1e-12 else 0.0
                ),
                "runtime_le_siting_max_violation_gw": float(
                    rt_subset_max_violation_gw
                ),
                "runtime_le_siting_violations": int(
                    rt_subset_max_violation_gw > 1e-6
                ),
                "siting_pool_gw_target": float(
                    ai_s4_od_params.get("siting_pool_gw", hourly_AI_load)
                ),
                "runtime_pool_gw_target": float(
                    ai_s4_od_params.get("runtime_pool_gw", hourly_AI_load)
                ),
            }
            for gid in AI_OD_FLOW:
                for origin, weight in ai_s4_od_params[
                    "origin_weight_by_cluster"
                ][gid].items():
                    results_AI_implied_province_od.setdefault(origin, {})
                    for dest in AI_OD_FLOW[gid]:
                        flow_gw = float(weight) * AI_OD_FLOW[gid][dest].x
                        results_AI_implied_province_od[origin][dest] = (
                            results_AI_implied_province_od[origin].get(dest, 0.0)
                            + flow_gw
                        )
            for origin, dest_flows in sorted(results_AI_implied_province_od.items()):
                origin_total = sum(dest_flows.values())
                for dest, flow_gw in sorted(dest_flows.items()):
                    results_AI_implied_province_od_records.append(
                        {
                            "origin_province": origin,
                            "destination_province": dest,
                            "implied_flow_gw": float(flow_gw),
                            "implied_flow_gwh_horizon": float(flow_gw * HOURS),
                            "origin_total_flow_gw": float(origin_total),
                            "share_of_origin_flow": float(
                                flow_gw / origin_total if origin_total > 1e-12 else 0.0
	                            ),
	                        }
	                    )
            if ai_operational_scenario == "S1":
                for dest in ai_s4_od_params["destination_provinces"]:
                    incoming_clusters = ai_s4_od_params["clusters_by_destination"][dest]
                    profiled_load = np.zeros(HOURS)
                    for gid in incoming_clusters:
                        profiled_load += (
                            AI_OD_FLOW[gid][dest].x
                            * np.asarray(ai_s4_od_params["cluster_profile"][gid], dtype=float)
                        )
                    results_AI_s1_profiled_load[dest] = profiled_load
            cluster_flow_total = sum(
                AI_OD_FLOW[gid][dest].x for gid in AI_OD_FLOW for dest in AI_OD_FLOW[gid]
            )
            cross_zone_flow = 0.0
            cross_zone_arc_count = 0
            source_balance_error = {}
            destination_plan_error = {}
            destination_power_cap_violation = {}
            for gid in AI_OD_FLOW:
                source_outflow = sum(AI_OD_FLOW[gid][dest].x for dest in AI_OD_FLOW[gid])
                source_balance_error[gid] = float(
                    source_outflow - float(ai_s4_od_params["cluster_mean_gw"][gid])
                )
                source_zone = ai_s4_od_params["cluster_zone"][gid]
                for dest in AI_OD_FLOW[gid]:
                    if str(province_zone.get(dest, "")) != str(source_zone):
                        cross_zone_flow += AI_OD_FLOW[gid][dest].x
                        cross_zone_arc_count += 1
            for dest in ai_s4_od_params["destination_provinces"]:
                incoming_clusters = ai_s4_od_params["clusters_by_destination"][dest]
                incoming_flow = sum(AI_OD_FLOW[gid][dest].x for gid in incoming_clusters)
                destination_plan_error[dest] = float(PRO_AI_LOAD[dest].x - incoming_flow)
                cap_gw = float(ai_s4_od_params["destination_power_cap_gw"][dest])
                max_violation = 0.0
                for h in range(HOURS):
                    fixed_gw = (
                        float(fixed_ai_load.get(dest, np.zeros(HOURS))[h]) / 1000.0
                    )
                    if ai_operational_scenario == "S1":
                        run_gw = float(results_AI_s1_profiled_load[dest][h])
                    else:
                        siting_only_gw = sum(
                            (
                                float(AI_OD_FLOW[gid][dest].x)
                                - float(AI_RT_OD_FLOW[gid][dest].x)
                            )
                            * float(ai_s4_od_params["cluster_profile"][gid][h])
                            for gid in incoming_clusters
                        )
                        run_gw = siting_only_gw + AI_BATCH_RUN[dest][h].x
                    max_violation = max(max_violation, fixed_gw + run_gw - cap_gw)
                destination_power_cap_violation[dest] = float(
                    max(0.0, max_violation)
                )
            implied_flow_total = sum(
                sum(dest_flows.values())
                for dest_flows in results_AI_implied_province_od.values()
            )
            cluster_od_solution_audit = {
                "enabled": True,
                "cross_zone_flow_gw": float(cross_zone_flow),
                "cross_zone_arc_count": int(cross_zone_arc_count),
                "ai_migration_ctilde": float(ai_migration_ctilde),
                "migration_cost_total_myuan": float(
                    gp.quicksum(migration_cost).getValue() if migration_cost else 0.0
                ),
                "migration_cost_per_cross_zone_gw_myuan": float(
                    (gp.quicksum(migration_cost).getValue() / cross_zone_flow)
                    if migration_cost and cross_zone_flow > 1e-9 else 0.0
                ),
                "zone_limited_od_enforced": bool(cross_zone_flow <= 1e-9),
                "source_balance_max_abs_error_gw": float(
                    max(
                        (abs(v) for v in source_balance_error.values()),
                        default=0.0,
                    )
                ),
                "destination_plan_max_abs_error_gw": float(
                    max(
                        (abs(v) for v in destination_plan_error.values()),
                        default=0.0,
                    )
                ),
                "destination_power_cap_max_violation_gw": float(
                    max(destination_power_cap_violation.values(), default=0.0)
                ),
                "source_balance_error_gw": source_balance_error,
                "destination_plan_error_gw": destination_plan_error,
                "destination_power_cap_violation_by_destination_gw": (
                    destination_power_cap_violation
                ),
            }
            ai_interface_metadata["cluster_od_solution_audit"] = cluster_od_solution_audit
            if ai_operational_scenario == "S4":
                ai_interface_metadata["s4_od_solution_audit"] = cluster_od_solution_audit
            elif ai_operational_scenario == "S1":
                ai_interface_metadata["s1_od_solution_audit"] = cluster_od_solution_audit

            # B-scheme unmet audit: unplaceable flexible AI
            unmet_siting_total_gw = float(
                sum(AI_UNMET_SITING[gid].x for gid in AI_UNMET_SITING)
            ) if AI_UNMET_SITING else 0.0
            unmet_runtime_total_gw = float(
                sum(AI_UNMET_RUNTIME[gid].x for gid in AI_UNMET_RUNTIME)
            ) if AI_UNMET_RUNTIME else 0.0
            siting_pool_target = float(
                ai_s4_od_params.get("siting_pool_gw", hourly_AI_load)
            )
            ai_interface_metadata["unmet_workload_audit"] = {
                "enabled": True,
                "unit": "GW (capacity-bound unplaceable flexible AI)",
                "unmet_siting_total_gw": unmet_siting_total_gw,
                "unmet_runtime_total_gw": unmet_runtime_total_gw,
                "unmet_siting_share_of_pool": float(
                    unmet_siting_total_gw / siting_pool_target
                    if siting_pool_target > 1e-9 else 0.0
                ),
                "unmet_by_cluster_gw": {
                    gid: float(AI_UNMET_SITING[gid].x) for gid in AI_UNMET_SITING
                } if AI_UNMET_SITING else {},
                "ai_unmet_penalty_per_gw": ai_unmet_penalty_per_gw,
                "interpretation": (
                    "unmet quantifies flexible AI that cannot be hosted under "
                    "destination capacity constraints. Run-NoMig (same-zone) vs "
                    "Run-FreeMig (adjacent-zone): reduction in unmet measures "
                    "the workload that spatial flexibility additionally absorbs."
                ),
            }

            ai_interface_metadata["implied_province_od_postprocess"] = {
                "enabled": True,
                "method": (
                    "Y_hat[p,d] = origin_reconstruction_weight[p,g] * Y[g,d]; "
                    "province-level OD is reconstructed ex post and is not directly optimized"
                ),
                "cluster_flow_total_gw": float(cluster_flow_total),
                "implied_province_flow_total_gw": float(implied_flow_total),
                "total_abs_error_gw": float(
                    abs(cluster_flow_total - implied_flow_total)
                ),
                "record_count": len(results_AI_implied_province_od_records),
            }
            if ai_operational_scenario == "S4":
                no_advance_violation = {}
                deadline_violation = {}
                max_arrival_execution_gap = {}
                no_advance_max_hour_by_destination = {}
                deadline_max_hour_by_destination = {}
                if ai_s4_zone_runtime_transfer_active:
                    results_AI_zone_batch_run = {}
                    for ze in province_zones:
                        run_cum = np.cumsum(
                            [
                                sum(
                                    AI_BATCH_RUN[p][h].x
                                    for p in provinces_by_zone[ze]
                                )
                                for h in range(HOURS)
                            ]
                        )
                        results_AI_zone_batch_run[ze] = np.array(
                            [
                                sum(
                                    AI_BATCH_RUN[p][h].x
                                    for p in provinces_by_zone[ze]
                                )
                                for h in range(HOURS)
                            ]
                        )
                        arr_cum = np.zeros(HOURS)
                        for gid in AI_RT_ZONE_ROUTE:
                            for zs in AI_RT_ZONE_ROUTE[gid]:
                                if ze not in AI_RT_ZONE_ROUTE[gid][zs]:
                                    continue
                                flow_gw = AI_RT_ZONE_ROUTE[gid][zs][ze].x
                                arr_cum += flow_gw * np.asarray(
                                    ai_s4_od_params.get(
                                        "runtime_cluster_profile_cum",
                                        ai_s4_od_params["cluster_profile_cum"],
                                    )[gid],
                                    dtype=float,
                                )
                        no_advance_gap_series = run_cum - arr_cum
                        no_advance_violation[ze] = float(
                            max(0.0, np.max(no_advance_gap_series))
                        )
                        no_advance_max_hour_by_destination[ze] = int(
                            np.argmax(no_advance_gap_series)
                        )
                        if ai_batch_delay_hours < HOURS:
                            required_cum = np.zeros(HOURS)
                            required_cum[ai_batch_delay_hours:] = arr_cum[
                                : HOURS - ai_batch_delay_hours
                            ]
                            deadline_gap_series = required_cum - run_cum
                            deadline_violation[ze] = float(
                                max(0.0, np.max(deadline_gap_series))
                            )
                            deadline_max_hour_by_destination[ze] = int(
                                np.argmax(deadline_gap_series)
                            )
                        else:
                            deadline_violation[ze] = 0.0
                            deadline_max_hour_by_destination[ze] = 0
                        max_arrival_execution_gap[ze] = float(
                            max(0.0, np.max(arr_cum - run_cum))
                        )
                    ai_interface_metadata["s4_zone_runtime_transfer_deadline_audit"] = {
                        "enabled": True,
                        "granularity": "execution_zone",
                        "method": (
                            "For each execution zone ze, run_cum[ze,h] is the "
                            "sum of provincial AI_BATCH_CUM within the zone and "
                            "arrival_cum[ze,h] is built from AI_RT_ZONE_ROUTE "
                            "and source-cluster cumulative profiles."
                        ),
                        "max_no_advance_violation_gwh": float(
                            max(no_advance_violation.values(), default=0.0)
                        ),
                        "max_deadline_violation_gwh": float(
                            max(deadline_violation.values(), default=0.0)
                        ),
                        "max_arrival_execution_gap_gwh": float(
                            max(max_arrival_execution_gap.values(), default=0.0)
                        ),
                        "no_advance_violation_by_zone_gwh": no_advance_violation,
                        "deadline_violation_by_zone_gwh": deadline_violation,
                        "no_advance_max_hour_by_zone": no_advance_max_hour_by_destination,
                        "deadline_max_hour_by_zone": deadline_max_hour_by_destination,
                    }
                    ai_interface_metadata.pop("s4_od_destination_deadline_audit", None)
                else:
                    for dest in ai_s4_od_params["destination_provinces"]:
                        incoming_clusters = ai_s4_od_params["clusters_by_destination"][dest]
                        run_cum = np.cumsum(
                            [AI_BATCH_RUN[dest][h].x for h in range(HOURS)]
                        )
                        arr_cum = np.zeros(HOURS)
                        for gid in incoming_clusters:
                            flow_gw = (
                                AI_RT_OD_FLOW[gid][dest].x
                                if dest in AI_RT_OD_FLOW.get(gid, {})
                                else 0.0
                            )
                            arr_cum += flow_gw * np.asarray(
                                ai_s4_od_params.get(
                                    "runtime_cluster_profile_cum",
                                    ai_s4_od_params["cluster_profile_cum"],
                                )[gid],
                                dtype=float,
                            )
                        no_advance_gap_series = run_cum - arr_cum
                        no_advance_violation[dest] = float(max(0.0, np.max(no_advance_gap_series)))
                        no_advance_max_hour_by_destination[dest] = int(
                            np.argmax(no_advance_gap_series)
                        )
                        if ai_batch_delay_hours < HOURS:
                            required_cum = np.zeros(HOURS)
                            required_cum[ai_batch_delay_hours:] = arr_cum[
                                : HOURS - ai_batch_delay_hours
                            ]
                            deadline_gap_series = required_cum - run_cum
                            deadline_violation[dest] = float(
                                max(0.0, np.max(deadline_gap_series))
                            )
                            deadline_max_hour_by_destination[dest] = int(
                                np.argmax(deadline_gap_series)
                            )
                        else:
                            deadline_violation[dest] = 0.0
                            deadline_max_hour_by_destination[dest] = 0
                        max_arrival_execution_gap[dest] = float(
                            max(0.0, np.max(arr_cum - run_cum))
                        )
                    ai_interface_metadata["s4_od_destination_deadline_audit"] = {
                        "enabled": True,
                        "granularity": "destination_province",
                        "method": (
                            "For each destination d, arrival_cum[d,h] = "
                            "sum_g Y_rt[g,d] * cumulative_profile[g,h]. Execution "
                            "must not exceed arrival_cum, and must catch up to "
                            "arrival_cum delayed by AI_BATCH_DELAY_HOURS."
                        ),
                        "max_no_advance_violation_gwh": float(
                            max(no_advance_violation.values(), default=0.0)
                        ),
                        "max_deadline_violation_gwh": float(
                            max(deadline_violation.values(), default=0.0)
                        ),
                        "max_arrival_execution_gap_gwh": float(
                            max(max_arrival_execution_gap.values(), default=0.0)
                        ),
                        "no_advance_violation_by_destination_gwh": no_advance_violation,
                        "deadline_violation_by_destination_gwh": deadline_violation,
                        "no_advance_max_hour_by_destination": (
                            no_advance_max_hour_by_destination
                        ),
                        "deadline_max_hour_by_destination": (
                            deadline_max_hour_by_destination
                        ),
                    }
            ai_interface_metadata["ai_hosting_concentration_audit"] = (
                compute_ai_hosting_concentration_audit(
                    results_AI_load=results_AI_load,
                    province_hosting_class=ai_s4_od_params.get(
                        "destination_hosting_class_by_province", {}
                    ),
                    hourly_AI_load=hourly_AI_load,
                    class_share_cap=AI_SCENARIO_DEST_CAPS.get(
                        ai_host_cap_scenario, {}
                    ),
                )
            )
            ai_interface_metadata["ai_od_concentration_audit"] = (
                compute_od_concentration_audit(
                    AI_OD_FLOW=AI_OD_FLOW,
                    ai_cluster_params=ai_s4_od_params,
                )
            )
        total_hydro_gen_gwh = sum(
            v["hydro_generation_gwh"]
            for v in results_hydro_cf_audit["by_province"].values()
        )
        total_hydro_full_cf_gwh = sum(
            v["hydro_full_cf_gwh"]
            for v in results_hydro_cf_audit["by_province"].values()
        )
        total_hydro_energy_cap_gwh = sum(
            v["hydro_energy_cap_gwh"]
            for v in results_hydro_cf_audit["by_province"].values()
        )
        results_hydro_cf_audit["national"] = {
            "total_hydro_generation_gwh": float(total_hydro_gen_gwh),
            "total_hydro_full_cf_gwh": float(total_hydro_full_cf_gwh),
            "total_hydro_energy_cap_gwh": float(total_hydro_energy_cap_gwh),
            "national_actual_cf": (
                float(total_hydro_gen_gwh / total_hydro_full_cf_gwh)
                if total_hydro_full_cf_gwh > 1e-12
                else 0.0
            ),
            "national_binding_ratio": (
                float(total_hydro_gen_gwh / total_hydro_energy_cap_gwh)
                if total_hydro_energy_cap_gwh > 1e-12
                else None
            ),
        }
        ai_interface_metadata["hydro_cf_audit"] = results_hydro_cf_audit
        print("Hydro CF audit national:", results_hydro_cf_audit["national"])

        total_wind_available = sum(np.sum(v) for v in results_sum_wind_hours.values())
        total_solar_available = sum(np.sum(v) for v in results_sum_solar_hours.values())
        total_wind_curtail = sum(np.sum(v) for v in results_curtail_wind.values())
        total_solar_curtail = sum(np.sum(v) for v in results_curtail_solar.values())
        results_vre_curtail_summary = {
            "wind_available": total_wind_available,
            "solar_available": total_solar_available,
            "wind_curtail": total_wind_curtail,
            "solar_curtail": total_solar_curtail,
            "wind_curtail_rate": total_wind_curtail / total_wind_available
            if total_wind_available > 0
            else 0,
            "solar_curtail_rate": total_solar_curtail / total_solar_available
            if total_solar_available > 0
            else 0,
            "vre_min_utilization": vre_min_utilization,
        }
        print("VRE curtailment summary:", results_vre_curtail_summary)

        def _expr_list_value(items):
            if not items:
                return 0.0
            return float(gp.quicksum(items).getValue())

        cost_components = {
            "objective": float(model.ObjVal),
            "power_system_objective": _expr_list_value([power_system_objective]),
            "wind_gen_cost": _expr_list_value(wind_gen_cost),
            "solar_gen_cost": _expr_list_value(solar_gen_cost),
            "ramp_up_cost": _expr_list_value(ramp_up_cost),
            "ramp_down_cost": _expr_list_value(ramp_dn_cost),
            "coal_ccs_variable_cost": _expr_list_value(coal_ccs_cost),
            "coal_ccs_fixed_cost": (
                _expr_list_value(coal_ccs_OM_fixed)
                + _expr_list_value(coal_ccs_cost_fixed)
            ),
            "gas_ccs_variable_cost": _expr_list_value(gas_ccs_cost),
            "gas_ccs_fixed_cost": (
                _expr_list_value(gas_ccs_OM_fixed)
                + _expr_list_value(gas_ccs_cost_fixed)
            ),
            "nuclear_cost": (
                _expr_list_value(nuclear_cost_var)
                + _expr_list_value(nuclear_cost_fixed)
                + _expr_list_value(nuclear_OM_fixed)
            ),
            "hydro_cost": (
                _expr_list_value(hydro_cost_fixed)
                + _expr_list_value(hydro_OM_fixed)
                + _expr_list_value(hydro_cost_fuel)
            ),
            "beccs_cost": (
                _expr_list_value(beccs_cost_fixed)
                + _expr_list_value(beccs_OM_fixed)
                + _expr_list_value(beccs_fuel)
            ),
            "load_shedding_cost": _expr_list_value(load_shedding_cost),
            "transmission_fixed_cost": (
                _expr_list_value(trans_fixed_cost_dc_installed)
                + _expr_list_value(trans_fixed_cost_ac_installed)
                + _expr_list_value(trans_fixed_cost_dc)
                + _expr_list_value(trans_fixed_cost_ac)
            ),
            "transmission_expansion_cost": (
                _expr_list_value(trans_expansion_cost_dc_installed)
                + _expr_list_value(trans_expansion_cost_ac_installed)
            ),
            "transmission_flow_cost": (
                _expr_list_value(trans_flow_cost_dc_installed)
                + _expr_list_value(trans_flow_cost_ac_installed)
                + _expr_list_value(trans_flow_cost_dc)
                + _expr_list_value(trans_flow_cost_ac)
            ),
            "ai_hosting_penalty_cost": _expr_list_value(ai_hosting_penalty_cost),
            "migration_cost": _expr_list_value(migration_cost),
            "ai_migration_ctilde": float(ai_migration_ctilde),
            "ai_migration_scale_per_gw_year": float(ai_migration_scale_per_gw_year),
            "c_mig_per_gw_year": float(c_mig_per_gw_year),
            "ai_batch_hosting_penalty_cost": _expr_list_value(
                ai_batch_hosting_penalty_cost
            ),
            "ai_penalty_in_objective": bool(include_ai_penalty_in_objective),
            "storage_fixed_cost": (
                _expr_list_value(fixed_phs_cost)
                + _expr_list_value(fixed_bat_cost)
                + _expr_list_value(fixed_h2_cost)
            ),
            "storage_variable_cost": (
                _expr_list_value(var_phs_cost)
                + _expr_list_value(var_bat_cost)
                + _expr_list_value(var_h2_cost)
            ),
        }
        cost_components["total_ai_penalty_cost"] = (
            cost_components["ai_hosting_penalty_cost"]
            + cost_components["ai_batch_hosting_penalty_cost"]
        )
        cost_components["ai_hosting_preference_penalty"] = cost_components[
            "ai_hosting_penalty_cost"
        ]
        cost_components["ai_batch_hosting_preference_penalty"] = cost_components[
            "ai_batch_hosting_penalty_cost"
        ]
        cost_components["total_ai_preference_penalty"] = cost_components[
            "total_ai_penalty_cost"
        ]
        cost_components["objective_includes_ai_penalty"] = bool(
            include_ai_penalty_in_objective
        )
        cost_components["ai_penalty_interpretation"] = (
            "AI hosting/migration penalty is a non-monetized preference diagnostic. "
            "It is excluded from reported power-system cost unless "
            "INCLUDE_AI_PENALTY_IN_OBJECTIVE=true."
        )
        component_sum_keys = [
            "wind_gen_cost",
            "solar_gen_cost",
            "ramp_up_cost",
            "ramp_down_cost",
            "coal_ccs_variable_cost",
            "coal_ccs_fixed_cost",
            "gas_ccs_variable_cost",
            "gas_ccs_fixed_cost",
            "nuclear_cost",
            "hydro_cost",
            "beccs_cost",
            "load_shedding_cost",
            "transmission_fixed_cost",
            "transmission_expansion_cost",
            "transmission_flow_cost",
            "storage_fixed_cost",
            "storage_variable_cost",
        ]
        if include_ai_penalty_in_objective:
            component_sum_keys = component_sum_keys + [
                "ai_hosting_penalty_cost",
                "ai_batch_hosting_penalty_cost",
            ]
        cost_components["component_sum_keys"] = component_sum_keys
        cost_components["component_sum"] = float(
            sum(cost_components[key] for key in component_sum_keys)
        )
        cost_components["component_sum_gap"] = float(
            cost_components["objective"] - cost_components["component_sum"]
        )
        cost_components["component_sum_relative_gap"] = float(
            cost_components["component_sum_gap"] / cost_components["objective"]
            if abs(cost_components["objective"]) > 0
            else 0.0
        )
        cost_components["reported_power_system_cost"] = float(
            cost_components["objective"] - cost_components["total_ai_penalty_cost"]
            if include_ai_penalty_in_objective
            else cost_components["objective"]
        )
        cost_components["objective_excluding_ai_penalty"] = float(
            cost_components["reported_power_system_cost"]
        )
        ai_interface_metadata["cost_components"] = cost_components

        results_trans_AC_cap = {}
        results_trans_DC_cap = {}
        results_trans_AC_installed_expansion_cap = {}
        results_trans_DC_installed_expansion_cap = {}
        for pair in trans_data["all_pair"]:
            if pair in trans_data["AC"]["pair"]:
                results_trans_AC_cap[pair] = np.array([trans_cap_AC[pair].x])
            if pair in trans_data["DC"]["pair"]:
                results_trans_DC_cap[pair] = np.array([trans_cap_DC[pair].x])
            if pair in trans_data["AC_installed"]["pair"]:
                results_trans_AC_installed_expansion_cap[pair] = np.array(
                    [trans_cap_AC_installed_expansion[pair].x]
                )
            if pair in trans_data["DC_installed"]["pair"]:
                results_trans_DC_installed_expansion_cap[pair] = np.array(
                    [trans_cap_DC_installed_expansion[pair].x]
                )

        with open(os.path.join(res_dir_pkl, "results_new_install_coal.pkl"), "wb") as f:
            pickle.dump(results_new_install_coal, f)
        with open(os.path.join(res_dir_pkl, "results_var_wind_cap.pkl"), "wb") as f:
            pickle.dump(results_var_wind_cap, f)
        with open(os.path.join(res_dir_pkl, "results_var_solar_cap.pkl"), "wb") as f:
            pickle.dump(results_var_solar_cap, f)
        with open(os.path.join(res_dir_pkl, "results_x_wind2.pkl"), "wb") as f:
            pickle.dump(results_x_wind2, f)
        with open(os.path.join(res_dir_pkl, "results_x_solar2.pkl"), "wb") as f:
            pickle.dump(results_x_solar2, f)
        with open(os.path.join(res_dir_pkl, "results_sum_wind_hours.pkl"), "wb") as f:
            pickle.dump(results_sum_wind_hours, f)
        with open(os.path.join(res_dir_pkl, "results_sum_wind_cell.pkl"), "wb") as f:
            pickle.dump(results_sum_wind_cell, f)
        with open(os.path.join(res_dir_pkl, "results_inter_wind.pkl"), "wb") as f:
            pickle.dump(results_inter_wind, f)
        with open(os.path.join(res_dir_pkl, "results_curtail_wind.pkl"), "wb") as f:
            pickle.dump(results_curtail_wind, f)
        with open(os.path.join(res_dir_pkl, "results_sum_solar_hours.pkl"), "wb") as f:
            pickle.dump(results_sum_solar_hours, f)
        with open(os.path.join(res_dir_pkl, "results_sum_solar_cell.pkl"), "wb") as f:
            pickle.dump(results_sum_solar_cell, f)
        with open(os.path.join(res_dir_pkl, "results_inter_solar.pkl"), "wb") as f:
            pickle.dump(results_inter_solar, f)
        with open(os.path.join(res_dir_pkl, "results_curtail_solar.pkl"), "wb") as f:
            pickle.dump(results_curtail_solar, f)
        with open(os.path.join(res_dir_pkl, "results_load_shedding.pkl"), "wb") as f:
            pickle.dump(results_load_shedding, f)
        with open(os.path.join(res_dir_pkl, "results_vre_curtail_summary.pkl"), "wb") as f:
            pickle.dump(results_vre_curtail_summary, f)
        with open(os.path.join(res_dir_pkl, "results_hydro_cf_audit.pkl"), "wb") as f:
            pickle.dump(results_hydro_cf_audit, f)
        with open(os.path.join(res_dir_pkl, "results_cap_phs.pkl"), "wb") as f:
            pickle.dump(results_cap_phs, f)
        with open(os.path.join(res_dir_pkl, "results_cap_bat.pkl"), "wb") as f:
            pickle.dump(results_cap_bat, f)
        with open(os.path.join(res_dir_pkl, "results_cap_ch_h2.pkl"), "wb") as f:
            pickle.dump(results_cap_ch_h2, f)
        with open(os.path.join(res_dir_pkl, "results_cap_dis_h2.pkl"), "wb") as f:
            pickle.dump(results_cap_dis_h2, f)

        with open(os.path.join(res_dir_pkl, "results_energy_bat.pkl"), "wb") as f:
            pickle.dump(results_energy_bat, f)
        with open(os.path.join(res_dir_pkl, "results_energy_phs.pkl"), "wb") as f:
            pickle.dump(results_energy_phs, f)
        with open(os.path.join(res_dir_pkl, "results_energy_h2.pkl"), "wb") as f:
            pickle.dump(results_energy_h2, f)
        with open(os.path.join(res_dir_pkl, "results_AI_load.pkl"), "wb") as f:
            pickle.dump(results_AI_load, f)
        with open(os.path.join(res_dir_pkl, "results_AI_batch_run.pkl"), "wb") as f:
            pickle.dump(results_AI_batch_run, f)
        ai_interface_metadata["ai_batch_run_result"] = {
            "file": "pkl_data/results_AI_batch_run.pkl",
            "saved": True,
            "unit": "GW hourly batch execution",
            "province_count": int(len(results_AI_batch_run)),
            "hours": int(HOURS),
            "active": bool(ai_operational_scenario == "S4"),
            "zero_filled_when_inactive": True,
            "note": (
                "AI_BATCH_RUN is active only in S4-OD. "
                "S1 injects fixed OD-profiled destination load directly into the power balance. "
                "S0 includes flexible AI directly in LOAD_DEMAND. "
                "S0/S1 arrays are saved as zeros for downstream compatibility."
            ),
        }
        with open(os.path.join(res_dir_pkl, "results_AI_batch_cum.pkl"), "wb") as f:
            pickle.dump(results_AI_batch_cum, f)
        if use_source_cluster_ai_interface:
            with open(os.path.join(res_dir_pkl, "results_AI_od_flow.pkl"), "wb") as f:
                pickle.dump(results_AI_od_flow, f)
            with open(os.path.join(res_dir_pkl, "results_AI_rt_od_flow.pkl"), "wb") as f:
                pickle.dump(results_AI_rt_od_flow, f)
            if ai_s4_zone_runtime_transfer_active:
                with open(
                    os.path.join(res_dir_pkl, "results_AI_rt_zone_route.pkl"), "wb"
                ) as f:
                    pickle.dump(results_AI_rt_zone_route, f)
                with open(
                    os.path.join(res_dir_pkl, "results_AI_zone_batch_run.pkl"), "wb"
                ) as f:
                    pickle.dump(results_AI_zone_batch_run, f)
        if results_AI_s1_profiled_load:
            with open(os.path.join(res_dir_pkl, "results_AI_s1_profiled_load.pkl"), "wb") as f:
                pickle.dump(results_AI_s1_profiled_load, f)
        if results_AI_implied_province_od:
            with open(os.path.join(res_dir_pkl, "results_AI_implied_province_od.pkl"), "wb") as f:
                pickle.dump(results_AI_implied_province_od, f)
            pd.DataFrame(results_AI_implied_province_od_records).to_csv(
                os.path.join(res_dir, "results_AI_implied_province_od.csv"),
                index=False,
            )

        ai_interface_metadata["ai_batch_cum_result"] = {
            "file": "pkl_data/results_AI_batch_cum.pkl",
            "saved": True,
            "unit": "GWh cumulative batch execution",
            "province_count": int(len(results_AI_batch_cum)),
            "hours": int(HOURS),
            "created_variable_province_count": int(len(AI_BATCH_CUM)),
            "zero_filled_when_variable_absent": True,
            "note": (
                "AI_BATCH_CUM variables are only created for S4-OD. "
                "S0 and S1 save zero arrays for downstream compatibility."
            ),
        }
        flexible_arrival_by_province_gwh = {
            str(pro): float(
                np.sum(
                    np.asarray(
                        flexible_ai_load.get(pro, np.zeros(HOURS)),
                        dtype=float,
                    )[:HOURS]
                )
                / 1000.0
            )
            for pro in Province
        }
        batch_run_by_province_gwh = {
            str(pro): float(np.sum(results_AI_batch_run.get(pro, np.zeros(HOURS))))
            for pro in Province
        }
        s1_profiled_by_province_gwh = {
            str(pro): float(np.sum(results_AI_s1_profiled_load.get(pro, np.zeros(HOURS))))
            for pro in Province
        }
        s1_profiled_total_gwh = float(sum(s1_profiled_by_province_gwh.values()))
        planned_flexible_by_province_gwh = {
            str(pro): float(results_AI_load[pro][0] * HOURS)
            for pro in Province
        }
        flexible_arrival_total_gwh = float(
            sum(flexible_arrival_by_province_gwh.values())
        )
        batch_run_total_gwh = float(sum(batch_run_by_province_gwh.values()))
        planned_flexible_total_gwh = float(
            sum(planned_flexible_by_province_gwh.values())
        )
        s0_reconstructed_run_by_province_gwh = None
        s0_reconstructed_run_total_gwh = None
        if use_external_ai_load:
            if ai_operational_scenario == "S0":
                if s0_use_cluster_reconstruction and s0_cluster_fixed_flexible_load_mw:
                    s0_reconstructed_run_by_province_gwh = {
                        str(pro): float(
                            np.sum(
                                np.asarray(
                                    s0_cluster_fixed_flexible_load_mw.get(
                                        str(pro), np.zeros(HOURS)
                                    ),
                                    dtype=float,
                                )[:HOURS]
                            )
                            / 1000.0
                        )
                        for pro in Province
                    }
                    s0_reconstructed_run_total_gwh = float(
                        sum(s0_reconstructed_run_by_province_gwh.values())
                    )
                    conservation_run_total_gwh = s0_reconstructed_run_total_gwh
                    conservation_run_by_province_gwh = (
                        s0_reconstructed_run_by_province_gwh
                    )
                    conservation_basis = (
                        "S0 immediate execution uses source-cluster reconstructed "
                        "origin flexible load in LOAD_DEMAND."
                    )
                else:
                    conservation_run_total_gwh = flexible_arrival_total_gwh
                    conservation_run_by_province_gwh = flexible_arrival_by_province_gwh
                    conservation_basis = (
                        "S0 immediate execution equals input province-level flexible_ai_load."
                    )
            elif ai_operational_scenario == "S1":
                conservation_run_total_gwh = s1_profiled_total_gwh
                conservation_run_by_province_gwh = s1_profiled_by_province_gwh
                conservation_basis = (
                    "S1-OD conservation compares source-cluster flexible arrivals, "
                    "OD allocation, and fixed destination hourly profiled load."
                )
            elif ai_operational_scenario == "S4":
                conservation_run_total_gwh = batch_run_total_gwh
                conservation_run_by_province_gwh = batch_run_by_province_gwh
                conservation_basis = (
                    "S4-OD conservation compares source-cluster flexible arrivals, "
                    "OD allocation, and destination AI_BATCH_RUN execution."
                )
            else:
                conservation_run_total_gwh = 0.0
                conservation_run_by_province_gwh = {
                    str(pro): 0.0 for pro in Province
                }
                conservation_basis = "No flexible AI execution in this scenario."
            conservation_arrival_total_gwh = flexible_arrival_total_gwh
            conservation_arrival_by_province_gwh = flexible_arrival_by_province_gwh
        else:
            conservation_arrival_total_gwh = float(hourly_AI_load * HOURS)
            conservation_arrival_by_province_gwh = {
                str(pro): 0.0 for pro in Province
            }
            conservation_run_total_gwh = (
                planned_flexible_total_gwh
                if ai_operational_scenario in {"S1", "S4"}
                else conservation_arrival_total_gwh
            )
            conservation_run_by_province_gwh = (
                planned_flexible_by_province_gwh
                if ai_operational_scenario in {"S1", "S4"}
                else {str(pro): 0.0 for pro in Province}
            )
            conservation_basis = (
                "Legacy non-external AI load uses hourly_AI_load as the flexible workload pool."
            )
        if AI_OD_FLOW:
            od_source_total_gwh = float(
                sum(
                    float(ai_s4_od_params["cluster_mean_gw"][gid]) * HOURS
                    for gid in ai_s4_od_params["source_clusters"]
                )
            )
            od_allocated_total_gwh = float(
                sum(
                    float(results_AI_od_flow[gid][dest][0]) * HOURS
                    for gid in results_AI_od_flow
                    for dest in results_AI_od_flow[gid]
                )
            )
            conservation_arrival_total_gwh = od_source_total_gwh
            if ai_operational_scenario == "S1":
                conservation_run_total_gwh = s1_profiled_total_gwh
                conservation_run_by_province_gwh = s1_profiled_by_province_gwh
                conservation_basis = (
                    "S1-OD conservation compares source-cluster flexible arrivals, "
                    "OD allocation, and fixed destination hourly profiled load."
                )
            else:
                # Phase 2b fix: the flexible AI that enters the destination power
                # balance is the siting-only immediate injection
                # (AI_OD_FLOW - AI_RT_OD_FLOW) PLUS the runtime batch execution
                # (AI_BATCH_RUN). batch_run_total_gwh alone counts only the
                # runtime (time-shiftable) subset, so comparing it against the
                # full siting arrival under-counts execution by (1 - theta) and
                # produces a false FAIL_STRUCTURAL.
                s4_siting_only_by_province_gwh = {}
                for d in ai_s4_od_params["destination_provinces"]:
                    incoming = ai_s4_od_params["clusters_by_destination"][d]
                    siting_only = 0.0
                    for gid in incoming:
                        s_val = float(AI_OD_FLOW[gid][d].x)
                        r_val = (
                            float(AI_RT_OD_FLOW[gid][d].x)
                            if d in AI_RT_OD_FLOW.get(gid, {})
                            else 0.0
                        )
                        # Constraint AI_RT_OD_FLOW <= AI_OD_FLOW guarantees s>=r.
                        siting_only += max(0.0, s_val - r_val) * HOURS
                    s4_siting_only_by_province_gwh[str(d)] = siting_only
                conservation_run_by_province_gwh = {
                    str(pro): float(
                        batch_run_by_province_gwh.get(str(pro), 0.0)
                        + s4_siting_only_by_province_gwh.get(str(pro), 0.0)
                    )
                    for pro in Province
                }
                conservation_run_total_gwh = float(
                    sum(conservation_run_by_province_gwh.values())
                )
                conservation_basis = (
                    "S4-OD conservation compares source-cluster flexible arrivals "
                    "against total destination injection = siting-only immediate "
                    "profile injection (AI_OD_FLOW - AI_RT_OD_FLOW) plus runtime "
                    "AI_BATCH_RUN execution. batch_run alone covers only the "
                    "runtime (theta) subset."
                )
        else:
            od_source_total_gwh = None
            od_allocated_total_gwh = None

        cluster_od_allocation_gap_gwh = (
            float(od_allocated_total_gwh - od_source_total_gwh)
            if od_source_total_gwh is not None and od_allocated_total_gwh is not None
            else None
        )
        # B-scheme: expected arrival is net of unplaceable workload (unmet).
        unmet_energy_gwh = (
            ai_interface_metadata.get("unmet_workload_audit", {})
            .get("unmet_siting_total_gw", 0.0)
            * HOURS
        )
        ai_execution_gap_gwh = float(
            conservation_run_total_gwh
            - (conservation_arrival_total_gwh - unmet_energy_gwh)
        )
        ai_execution_abs_gap_gwh = abs(ai_execution_gap_gwh)
        ai_execution_relative_abs_gap = float(
            ai_execution_abs_gap_gwh / abs(conservation_arrival_total_gwh)
            if abs(conservation_arrival_total_gwh) > 1e-9
            else 0.0
        )
        ai_energy_audit_status = classify_energy_gap(
            abs_gap_gwh=ai_execution_abs_gap_gwh,
            rel_gap=ai_execution_relative_abs_gap,
            strict_abs_gwh=ai_energy_audit_abs_tol_gwh,
            strict_rel=ai_energy_audit_rel_tol,
            warn_abs_gwh=ai_energy_warn_abs_tol_gwh,
            warn_rel=ai_energy_warn_rel_tol,
            review_abs_gwh=ai_energy_review_abs_tol_gwh,
            review_rel=ai_energy_review_rel_tol,
            fail_abs_gwh=ai_energy_fail_abs_tol_gwh,
            fail_rel=ai_energy_fail_rel_tol,
        )
        ai_energy_audit_passed = (
            ai_energy_audit_status != "FAIL_STRUCTURAL"
        )
        ai_energy_audit_strict_passed = ai_energy_audit_status == "PASS_STRICT"
        ai_energy_audit_not_failed = ai_energy_audit_status != "FAIL_STRUCTURAL"
        ai_energy_abs_strict_passed = bool(
            ai_execution_abs_gap_gwh <= ai_energy_audit_abs_tol_gwh
        )
        ai_energy_rel_strict_passed = bool(
            ai_execution_relative_abs_gap <= ai_energy_audit_rel_tol
        )
        ai_energy_abs_warn_passed = bool(
            ai_execution_abs_gap_gwh <= ai_energy_warn_abs_tol_gwh
        )
        ai_energy_rel_warn_passed = bool(
            ai_execution_relative_abs_gap <= ai_energy_warn_rel_tol
        )
        ai_energy_abs_review_passed = bool(
            ai_execution_abs_gap_gwh <= ai_energy_review_abs_tol_gwh
        )
        ai_energy_rel_review_passed = bool(
            ai_execution_relative_abs_gap <= ai_energy_review_rel_tol
        )
        cluster_od_allocation_abs_gap_gwh = (
            abs(float(cluster_od_allocation_gap_gwh))
            if cluster_od_allocation_gap_gwh is not None
            else None
        )
        cluster_od_allocation_relative_abs_gap = (
            float(cluster_od_allocation_abs_gap_gwh / abs(od_source_total_gwh))
            if cluster_od_allocation_abs_gap_gwh is not None
            and od_source_total_gwh is not None
            and abs(od_source_total_gwh) > 1e-9
            else None
        )
        if cluster_od_allocation_abs_gap_gwh is None:
            cluster_od_allocation_audit_status = "NOT_APPLICABLE"
        else:
            cluster_od_allocation_audit_status = classify_energy_gap(
                abs_gap_gwh=cluster_od_allocation_abs_gap_gwh,
                rel_gap=(
                    cluster_od_allocation_relative_abs_gap
                    if cluster_od_allocation_relative_abs_gap is not None
                    else 0.0
                ),
                strict_abs_gwh=ai_energy_audit_abs_tol_gwh,
                strict_rel=ai_energy_audit_rel_tol,
                warn_abs_gwh=ai_energy_warn_abs_tol_gwh,
                warn_rel=ai_energy_warn_rel_tol,
                review_abs_gwh=ai_energy_review_abs_tol_gwh,
                review_rel=ai_energy_review_rel_tol,
                fail_abs_gwh=ai_energy_fail_abs_tol_gwh,
                fail_rel=ai_energy_fail_rel_tol,
            )
        cluster_od_allocation_audit_strict_passed = (
            None
            if cluster_od_allocation_audit_status == "NOT_APPLICABLE"
            else cluster_od_allocation_audit_status == "PASS_STRICT"
        )
        cluster_od_allocation_audit_not_failed = (
            None
            if cluster_od_allocation_audit_status == "NOT_APPLICABLE"
            else cluster_od_allocation_audit_status != "FAIL_STRUCTURAL"
        )
        cluster_od_allocation_audit_passed = (
            cluster_od_allocation_audit_status != "FAIL_STRUCTURAL"
        )
        cluster_od_solution_audit = ai_interface_metadata.get(
            "cluster_od_solution_audit", {}
        )
        destination_power_cap_violation_gw = float(
            cluster_od_solution_audit.get("destination_power_cap_max_violation_gw", 0.0)
            or 0.0
        )
        destination_power_cap_audit_status = classify_power_gap(
            gap_gw=destination_power_cap_violation_gw,
            strict_gw=ai_power_audit_tol_gw,
            warn_gw=ai_power_warn_tol_gw,
            review_gw=ai_power_review_tol_gw,
            fail_gw=ai_power_fail_tol_gw,
        )
        destination_power_cap_audit_passed = (
            destination_power_cap_audit_status != "FAIL_STRUCTURAL"
        )
        destination_power_cap_audit_strict_passed = (
            destination_power_cap_audit_status == "PASS_STRICT"
        )
        destination_power_cap_audit_not_failed = (
            destination_power_cap_audit_status != "FAIL_STRUCTURAL"
        )
        s4_deadline_audit_key = (
            "s4_zone_runtime_transfer_deadline_audit"
            if ai_s4_zone_runtime_transfer_active
            else "s4_od_destination_deadline_audit"
        )
        s4_deadline_audit = ai_interface_metadata.get(s4_deadline_audit_key, {})
        if ai_operational_scenario == "S4":
            no_advance = float(
                s4_deadline_audit.get("max_no_advance_violation_gwh", 0.0)
            )
            deadline = float(
                s4_deadline_audit.get("max_deadline_violation_gwh", 0.0)
            )
            no_advance_audit_status = classify_temporal_gap(
                gap_gwh=no_advance,
                strict_gwh=ai_temporal_audit_tol_gwh,
                warn_gwh=ai_temporal_warn_tol_gwh,
                review_gwh=ai_temporal_review_tol_gwh,
                fail_gwh=ai_temporal_fail_tol_gwh,
            )
            deadline_audit_status = classify_temporal_gap(
                gap_gwh=deadline,
                strict_gwh=ai_temporal_audit_tol_gwh,
                warn_gwh=ai_temporal_warn_tol_gwh,
                review_gwh=ai_temporal_review_tol_gwh,
                fail_gwh=ai_temporal_fail_tol_gwh,
            )
            no_advance_audit_passed = (
                no_advance_audit_status != "FAIL_STRUCTURAL"
            )
            deadline_audit_passed = deadline_audit_status != "FAIL_STRUCTURAL"
            no_advance_audit_strict_passed = (
                no_advance_audit_status == "PASS_STRICT"
            )
            deadline_audit_strict_passed = (
                deadline_audit_status == "PASS_STRICT"
            )
            no_advance_audit_not_failed = (
                no_advance_audit_status != "FAIL_STRUCTURAL"
            )
            deadline_audit_not_failed = (
                deadline_audit_status != "FAIL_STRUCTURAL"
            )
            s4_deadline_audit["ai_temporal_audit_tol_gwh"] = float(
                ai_temporal_audit_tol_gwh
            )
            s4_deadline_audit["ai_temporal_warn_tol_gwh"] = float(
                ai_temporal_warn_tol_gwh
            )
            s4_deadline_audit["ai_temporal_review_tol_gwh"] = float(
                ai_temporal_review_tol_gwh
            )
            s4_deadline_audit["ai_temporal_fail_tol_gwh"] = float(
                ai_temporal_fail_tol_gwh
            )
            s4_deadline_audit["no_advance_audit_strict_passed"] = (
                no_advance_audit_strict_passed
            )
            s4_deadline_audit["deadline_audit_strict_passed"] = (
                deadline_audit_strict_passed
            )
            s4_deadline_audit["no_advance_audit_not_failed"] = (
                no_advance_audit_not_failed
            )
            s4_deadline_audit["deadline_audit_not_failed"] = (
                deadline_audit_not_failed
            )
            s4_deadline_audit["no_advance_audit_passed"] = no_advance_audit_passed
            s4_deadline_audit["deadline_audit_passed"] = deadline_audit_passed
            s4_deadline_audit["no_advance_audit_status"] = no_advance_audit_status
            s4_deadline_audit["deadline_audit_status"] = deadline_audit_status
            ai_interface_metadata[s4_deadline_audit_key] = s4_deadline_audit
            if ai_s4_zone_runtime_transfer_active:
                ai_interface_metadata.pop("s4_od_destination_deadline_audit", None)
        else:
            no_advance = None
            deadline = None
            no_advance_audit_status = "NOT_APPLICABLE"
            deadline_audit_status = "NOT_APPLICABLE"
            no_advance_audit_passed = None
            deadline_audit_passed = None
            no_advance_audit_strict_passed = None
            deadline_audit_strict_passed = None
            no_advance_audit_not_failed = None
            deadline_audit_not_failed = None
            ai_interface_metadata.pop("s4_od_destination_deadline_audit", None)
            ai_interface_metadata.pop("s4_zone_runtime_transfer_deadline_audit", None)
        if cluster_od_solution_audit:
            cluster_od_solution_audit["ai_power_audit_tol_gw"] = float(
                ai_power_audit_tol_gw
            )
            cluster_od_solution_audit["ai_power_warn_tol_gw"] = float(
                ai_power_warn_tol_gw
            )
            cluster_od_solution_audit["ai_power_review_tol_gw"] = float(
                ai_power_review_tol_gw
            )
            cluster_od_solution_audit["ai_power_fail_tol_gw"] = float(
                ai_power_fail_tol_gw
            )
            cluster_od_solution_audit["destination_power_cap_audit_passed"] = (
                destination_power_cap_audit_passed
            )
            cluster_od_solution_audit["destination_power_cap_audit_status"] = (
                destination_power_cap_audit_status
            )
            ai_interface_metadata["cluster_od_solution_audit"] = (
                cluster_od_solution_audit
            )
            if ai_operational_scenario == "S4":
                ai_interface_metadata["s4_od_solution_audit"] = cluster_od_solution_audit
            elif ai_operational_scenario == "S1":
                ai_interface_metadata["s1_od_solution_audit"] = cluster_od_solution_audit
        audit_statuses = [
            ai_energy_audit_status,
            cluster_od_allocation_audit_status,
            destination_power_cap_audit_status,
        ]
        if ai_operational_scenario == "S4":
            audit_statuses.extend(
                [
                    no_advance_audit_status,
                    deadline_audit_status,
                ]
            )
        overall_ai_audit_status = combine_audit_status(audit_statuses)
        usable_for_main_analysis = overall_ai_audit_status in {
            "PASS_STRICT",
            "WARN_NUMERICAL_RESIDUAL",
        }
        requires_manual_review = (
            overall_ai_audit_status == "WARN_REVIEW_REQUIRED"
        )
        hard_failed = overall_ai_audit_status == "FAIL_STRUCTURAL"
        ai_execution_audit = {
            "scenario": ai_operational_scenario,
            "effective_ai_scenario": ai_interface_metadata.get(
                "effective_ai_scenario"
            ),
            "use_external_ai_load": bool(use_external_ai_load),
            "use_source_cluster_ai_interface": bool(use_source_cluster_ai_interface),
            "s4_zone_runtime_transfer_enabled": bool(
                ai_s4_zone_runtime_transfer
            ),
            "s4_temporal_audit_key": (
                s4_deadline_audit_key
                if ai_operational_scenario == "S4"
                else "NOT_APPLICABLE"
            ),
            "s4_temporal_audit_granularity": (
                s4_deadline_audit.get("granularity")
                if ai_operational_scenario == "S4"
                else None
            ),
            "strict_postsolve_ai_audit": bool(strict_postsolve_ai_audit),
            "raise_on_structural_ai_audit_fail": bool(
                raise_on_structural_ai_audit_fail
            ),
            "ai_audit_tol_gwh": float(ai_audit_tol_gwh),
            "ai_energy_audit_abs_tol_gwh": float(ai_energy_audit_abs_tol_gwh),
            "ai_energy_audit_rel_tol": float(ai_energy_audit_rel_tol),
            "ai_temporal_audit_tol_gwh": float(ai_temporal_audit_tol_gwh),
            "ai_power_audit_tol_gw": float(ai_power_audit_tol_gw),
            "audit_policy": {
                "policy_name": "tiered_ai_postsolve_audit",
                "strict_thresholds_are_report_only": True,
                "review_warning_abs_gap_gwh": 0.1,
                "hard_fail_requires_structural_gap": True,
                "energy_fail_rule": (
                    "FAIL only if abs_gap_gwh > AI_ENERGY_FAIL_ABS_TOL_GWH "
                    "and relative_abs_gap > AI_ENERGY_FAIL_REL_TOL"
                ),
                "temporal_fail_rule": (
                    "FAIL only if no-advance/deadline violation exceeds "
                    "AI_TEMPORAL_FAIL_TOL_GWH"
                ),
                "power_fail_rule": (
                    "FAIL only if destination power-cap violation exceeds "
                    "AI_POWER_FAIL_TOL_GW"
                ),
            },
            "audit_thresholds": {
                "ai_energy_audit_abs_tol_gwh": float(ai_energy_audit_abs_tol_gwh),
                "ai_energy_audit_rel_tol": float(ai_energy_audit_rel_tol),
                "ai_energy_warn_abs_tol_gwh": float(ai_energy_warn_abs_tol_gwh),
                "ai_energy_warn_rel_tol": float(ai_energy_warn_rel_tol),
                "ai_energy_review_abs_tol_gwh": float(ai_energy_review_abs_tol_gwh),
                "ai_energy_review_rel_tol": float(ai_energy_review_rel_tol),
                "ai_energy_fail_abs_tol_gwh": float(ai_energy_fail_abs_tol_gwh),
                "ai_energy_fail_rel_tol": float(ai_energy_fail_rel_tol),
                "ai_temporal_audit_tol_gwh": float(ai_temporal_audit_tol_gwh),
                "ai_temporal_warn_tol_gwh": float(ai_temporal_warn_tol_gwh),
                "ai_temporal_review_tol_gwh": float(ai_temporal_review_tol_gwh),
                "ai_temporal_fail_tol_gwh": float(ai_temporal_fail_tol_gwh),
                "ai_power_audit_tol_gw": float(ai_power_audit_tol_gw),
                "ai_power_warn_tol_gw": float(ai_power_warn_tol_gw),
                "ai_power_review_tol_gw": float(ai_power_review_tol_gw),
                "ai_power_fail_tol_gw": float(ai_power_fail_tol_gw),
            },
            "hours": int(HOURS),
            "unit": "GWh over modeled horizon",
            "conservation_basis": conservation_basis,
            "arrival_total_gwh": float(conservation_arrival_total_gwh),
            "run_total_gwh": float(conservation_run_total_gwh),
            "gap_gwh": ai_execution_gap_gwh,
            "abs_gap_gwh": float(ai_execution_abs_gap_gwh),
            "relative_gap": float(
                ai_execution_gap_gwh / conservation_arrival_total_gwh
                if abs(conservation_arrival_total_gwh) > 1e-9
                else 0.0
            ),
            "relative_abs_gap": float(ai_execution_relative_abs_gap),
            "ai_energy_audit_status": ai_energy_audit_status,
            "ai_energy_audit_strict_passed": bool(
                ai_energy_audit_strict_passed
            ),
            "ai_energy_audit_not_failed": bool(ai_energy_audit_not_failed),
            "ai_energy_audit_passed": bool(ai_energy_audit_not_failed),
            "ai_energy_audit_passed_note": (
                "Backward-compatible alias for ai_energy_audit_not_failed; "
                "use ai_energy_audit_status and ai_energy_audit_strict_passed "
                "to distinguish strict pass from warning states."
            ),
            "ai_energy_abs_strict_passed": bool(ai_energy_abs_strict_passed),
            "ai_energy_rel_strict_passed": bool(ai_energy_rel_strict_passed),
            "ai_energy_abs_warn_passed": bool(ai_energy_abs_warn_passed),
            "ai_energy_rel_warn_passed": bool(ai_energy_rel_warn_passed),
            "ai_energy_abs_review_passed": bool(ai_energy_abs_review_passed),
            "ai_energy_rel_review_passed": bool(ai_energy_rel_review_passed),
            "input_flexible_arrival_total_gwh": flexible_arrival_total_gwh,
            "planned_flexible_total_gwh": planned_flexible_total_gwh,
            "batch_run_total_gwh": batch_run_total_gwh,
            "s1_profiled_total_gwh": s1_profiled_total_gwh,
            "arrival_by_province_gwh": conservation_arrival_by_province_gwh,
            "run_by_province_gwh": conservation_run_by_province_gwh,
            "planned_flexible_by_province_gwh": planned_flexible_by_province_gwh,
            "batch_run_by_province_gwh": batch_run_by_province_gwh,
            "s1_profiled_by_province_gwh": s1_profiled_by_province_gwh,
            "s0_reconstructed_run_total_gwh": s0_reconstructed_run_total_gwh,
            "s0_reconstruction_used_in_power_balance": bool(
                s0_use_cluster_reconstruction
                and ai_operational_scenario == "S0"
                and s0_cluster_fixed_flexible_load_mw
            ),
            "cluster_od_source_total_gwh": od_source_total_gwh,
            "cluster_od_allocated_total_gwh": od_allocated_total_gwh,
            "cluster_od_allocation_gap_gwh": cluster_od_allocation_gap_gwh,
            "cluster_od_allocation_abs_gap_gwh": cluster_od_allocation_abs_gap_gwh,
            "cluster_od_allocation_relative_abs_gap": (
                cluster_od_allocation_relative_abs_gap
            ),
            "cluster_od_allocation_audit_status": (
                cluster_od_allocation_audit_status
            ),
            "cluster_od_allocation_audit_strict_passed": (
                cluster_od_allocation_audit_strict_passed
            ),
            "cluster_od_allocation_audit_not_failed": (
                cluster_od_allocation_audit_not_failed
            ),
            "cluster_od_allocation_audit_passed": (
                cluster_od_allocation_audit_not_failed
            ),
            "destination_power_cap_max_violation_gw": (
                destination_power_cap_violation_gw
            ),
            "destination_power_cap_audit_status": (
                destination_power_cap_audit_status
            ),
            "destination_power_cap_audit_strict_passed": bool(
                destination_power_cap_audit_strict_passed
            ),
            "destination_power_cap_audit_not_failed": bool(
                destination_power_cap_audit_not_failed
            ),
            "destination_power_cap_audit_passed": bool(
                destination_power_cap_audit_not_failed
            ),
            "s4_temporal_audit_key": (
                s4_deadline_audit_key
                if ai_operational_scenario == "S4"
                else None
            ),
            "s4_temporal_audit_granularity": (
                s4_deadline_audit.get("granularity")
                if ai_operational_scenario == "S4"
                else None
            ),
            "s4_zone_runtime_transfer_enabled": bool(
                ai_s4_zone_runtime_transfer and ai_operational_scenario == "S4"
            ),
            "s4_od_no_advance_max_violation_gwh": (
                no_advance if ai_operational_scenario == "S4" else None
            ),
            "s4_od_deadline_max_violation_gwh": (
                deadline if ai_operational_scenario == "S4" else None
            ),
            "s4_od_no_advance_max_hour_by_destination": (
                s4_deadline_audit.get(
                    "no_advance_max_hour_by_destination",
                    s4_deadline_audit.get("no_advance_max_hour_by_zone"),
                )
                if ai_operational_scenario == "S4"
                else None
            ),
            "s4_od_deadline_max_hour_by_destination": (
                s4_deadline_audit.get(
                    "deadline_max_hour_by_destination",
                    s4_deadline_audit.get("deadline_max_hour_by_zone"),
                )
                if ai_operational_scenario == "S4"
                else None
            ),
            "s4_zone_no_advance_max_hour_by_zone": (
                s4_deadline_audit.get("no_advance_max_hour_by_zone")
                if ai_operational_scenario == "S4" and ai_s4_zone_runtime_transfer_active
                else None
            ),
            "s4_zone_deadline_max_hour_by_zone": (
                s4_deadline_audit.get("deadline_max_hour_by_zone")
                if ai_operational_scenario == "S4" and ai_s4_zone_runtime_transfer_active
                else None
            ),
            "s4_zone_no_advance_audit_status": (
                no_advance_audit_status
                if ai_operational_scenario == "S4" and ai_s4_zone_runtime_transfer_active
                else None
            ),
            "s4_zone_deadline_audit_status": (
                deadline_audit_status
                if ai_operational_scenario == "S4" and ai_s4_zone_runtime_transfer_active
                else None
            ),
            "s4_destination_field_deprecated_note": (
                "For S4_ZONE_RUNTIME_TRANSFER, destination-named temporal fields "
                "are retained only for backward compatibility; use the zone-named "
                "fields above as the primary source of truth."
            ),
            "s4_od_no_advance_audit_status": (
                no_advance_audit_status
                if ai_operational_scenario == "S4"
                else "NOT_APPLICABLE"
            ),
            "s4_od_deadline_audit_status": (
                deadline_audit_status
                if ai_operational_scenario == "S4"
                else "NOT_APPLICABLE"
            ),
            "s4_od_no_advance_audit_strict_passed": (
                no_advance_audit_strict_passed
                if ai_operational_scenario == "S4"
                else None
            ),
            "s4_od_deadline_audit_strict_passed": (
                deadline_audit_strict_passed
                if ai_operational_scenario == "S4"
                else None
            ),
            "s4_od_no_advance_audit_not_failed": (
                no_advance_audit_not_failed
                if ai_operational_scenario == "S4"
                else None
            ),
            "s4_od_deadline_audit_not_failed": (
                deadline_audit_not_failed
                if ai_operational_scenario == "S4"
                else None
            ),
            "s4_od_no_advance_audit_passed": (
                no_advance_audit_not_failed
                if ai_operational_scenario == "S4"
                else None
            ),
            "s4_od_deadline_audit_passed": (
                deadline_audit_not_failed
                if ai_operational_scenario == "S4"
                else None
            ),
            "overall_ai_audit_status": overall_ai_audit_status,
            "usable_for_main_analysis": bool(usable_for_main_analysis),
            "usable_for_main_analysis_rule": (
                "WARN_REVIEW_REQUIRED completes postprocessing but requires manual "
                "review before use in main analysis."
            ),
            "requires_manual_review": bool(requires_manual_review),
            "hard_failed": bool(hard_failed),
            "cluster_od_field_note": (
                "cluster_od_* fields apply to both S1-OD and S4-OD. "
                "s4_od_* fields are retained only for backward compatibility and are "
                "populated only for S4."
            ),
            "s4_od_source_total_gwh": (
                od_source_total_gwh if ai_operational_scenario == "S4" else None
            ),
            "s4_od_allocated_total_gwh": (
                od_allocated_total_gwh if ai_operational_scenario == "S4" else None
            ),
            "s4_od_allocation_gap_gwh": (
                cluster_od_allocation_gap_gwh
                if ai_operational_scenario == "S4"
                else None
            ),
        }
        ai_interface_metadata["ai_execution_audit"] = ai_execution_audit
        with open(os.path.join(res_dir, "ai_execution_audit.json"), "w", encoding="utf-8") as f:
            json.dump(ai_execution_audit, f, indent=2, ensure_ascii=False)
        if ai_s4_zone_runtime_transfer_active:
            with open(
                os.path.join(res_dir, "s4_zone_runtime_transfer_audit.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "config": ai_interface_metadata.get(
                            "s4_zone_runtime_transfer_config", {}
                        ),
                        "solution": ai_interface_metadata.get(
                            "s4_zone_runtime_transfer_solution_audit", {}
                        ),
                        "deadline": ai_interface_metadata.get(
                            "s4_zone_runtime_transfer_deadline_audit", {}
                        ),
                        "ai_execution_audit_summary": {
                            "overall_ai_audit_status": overall_ai_audit_status,
                            "ai_energy_audit_status": ai_energy_audit_status,
                            "no_advance_audit_status": no_advance_audit_status,
                            "deadline_audit_status": deadline_audit_status,
                            "destination_power_cap_audit_status": (
                                destination_power_cap_audit_status
                            ),
                        },
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        if s0_reconstruction_audit is not None:
            with open(
                os.path.join(res_dir, "s0_cluster_reconstruction_audit.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(s0_reconstruction_audit, f, indent=2, ensure_ascii=False)
        if energy_accounting_mwh:
            with open(os.path.join(res_dir, "energy_accounting.json"), "w", encoding="utf-8") as f:
                json.dump(energy_accounting_mwh, f, indent=2, ensure_ascii=False)
        with open(os.path.join(res_dir, "cost_components.json"), "w", encoding="utf-8") as f:
            json.dump(cost_components, f, indent=2, ensure_ascii=False)
        with open(os.path.join(res_dir, "ai_load_interface_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(ai_interface_metadata, f, indent=2, ensure_ascii=False)
        if overall_ai_audit_status in {
            "WARN_NUMERICAL_RESIDUAL",
            "WARN_REVIEW_REQUIRED",
        }:
            print(
                "WARNING: AI post-solve audit did not pass strict thresholds, "
                "but is below structural hard-fail threshold. "
                f"overall_ai_audit_status={overall_ai_audit_status}; "
                f"ai_energy_audit_status={ai_energy_audit_status}; "
                f"cluster_od_allocation_audit_status={cluster_od_allocation_audit_status}; "
                f"destination_power_cap_audit_status={destination_power_cap_audit_status}; "
                f"s4_od_no_advance_audit_status="
                f"{no_advance_audit_status if ai_operational_scenario == 'S4' else 'NA'}; "
                f"s4_od_deadline_audit_status="
                f"{deadline_audit_status if ai_operational_scenario == 'S4' else 'NA'}."
            )
        with open(os.path.join(res_dir_pkl, "results_underground_h2.pkl"), "wb") as f:
            pickle.dump(results_underground_h2, f)
        with open(os.path.join(res_dir_pkl, "results_dischar_phs.pkl"), "wb") as f:
            pickle.dump(results_dischar_phs, f)
        with open(os.path.join(res_dir_pkl, "results_dischar_bat.pkl"), "wb") as f:
            pickle.dump(results_dischar_bat, f)

        with open(os.path.join(res_dir_pkl, "results_dischar_h2.pkl"), "wb") as f:
            pickle.dump(results_dischar_h2, f)

        with open(os.path.join(res_dir_pkl, "results_tot_energy_h2.pkl"), "wb") as f:
            pickle.dump(results_tot_energy_h2, f)
        with open(os.path.join(res_dir_pkl, "results_tot_energy_phs.pkl"), "wb") as f:
            pickle.dump(results_tot_energy_phs, f)
        with open(os.path.join(res_dir_pkl, "results_tot_energy_bat.pkl"), "wb") as f:
            pickle.dump(results_tot_energy_bat, f)
        with open(os.path.join(res_dir_pkl, "results_ru.pkl"), "wb") as f:
            pickle.dump(results_ru, f)
        with open(os.path.join(res_dir_pkl, "results_rd.pkl"), "wb") as f:
            pickle.dump(results_rd, f)
        with open(os.path.join(res_dir_pkl, "results_load_conv.pkl"), "wb") as f:
            pickle.dump(results_load_conv, f)

        with open(os.path.join(res_dir_pkl, "results_trans_out.pkl"), "wb") as f:
            pickle.dump(results_trans_out, f)
        with open(os.path.join(res_dir_pkl, "results_charge_phs.pkl"), "wb") as f:
            pickle.dump(results_charge_phs, f)
        with open(os.path.join(res_dir_pkl, "results_charge_bat.pkl"), "wb") as f:
            pickle.dump(results_charge_bat, f)
        with open(os.path.join(res_dir_pkl, "results_charge_h2.pkl"), "wb") as f:
            pickle.dump(results_charge_h2, f)
        with open(os.path.join(res_dir_pkl, "results_load_trans.pkl"), "wb") as f:
            pickle.dump(results_load_trans, f)

        with open(os.path.join(res_dir_pkl, "results_trans_AC_cap.pkl"), "wb") as f:
            pickle.dump(results_trans_AC_cap, f)
        with open(os.path.join(res_dir_pkl, "results_trans_DC_cap.pkl"), "wb") as f:
            pickle.dump(results_trans_DC_cap, f)
        with open(
            os.path.join(res_dir_pkl, "results_trans_AC_installed_expansion_cap.pkl"),
            "wb",
        ) as f:
            pickle.dump(results_trans_AC_installed_expansion_cap, f)
        with open(
            os.path.join(res_dir_pkl, "results_trans_DC_installed_expansion_cap.pkl"),
            "wb",
        ) as f:
            pickle.dump(results_trans_DC_installed_expansion_cap, f)


        with open(os.path.join(res_dir_pkl, "results_load_trans_AC.pkl"), "wb") as f:
            pickle.dump(results_load_trans_AC, f)

        with open(os.path.join(res_dir_pkl, "results_load_trans_DC.pkl"), "wb") as f:
            pickle.dump(results_load_trans_DC, f)

        with open(os.path.join(res_dir_pkl, "results_load_trans_DC_installed.pkl"), "wb") as f:
            pickle.dump(results_load_trans_DC_installed, f)

        with open(os.path.join(res_dir_pkl, "results_load_trans_AC_installed.pkl"), "wb") as f:
            pickle.dump(results_load_trans_AC_installed, f)

        result_dict = {}
        result_dict = {
            "PRO_under_h2_CAP": PRO_under_h2_CAP,
            "PRO_AI_LOAD": PRO_AI_LOAD,
            "AI_BATCH_RUN": AI_BATCH_RUN,
            "AI_OD_FLOW": AI_OD_FLOW,
            "AI_RT_OD_FLOW": AI_RT_OD_FLOW,
            "AI_RT_ZONE_ROUTE": AI_RT_ZONE_ROUTE,
            "AI_ZONE_BATCH_RUN": results_AI_zone_batch_run,
            "ai_s4_od_params": ai_s4_od_params,
            "results_AI_s1_profiled_load": results_AI_s1_profiled_load,
            "results_AI_implied_province_od": results_AI_implied_province_od,
            "pv_lcoe": pv_lcoe,
            "wind_lcoe": wind_lcoe,
            "HOURS": HOURS,
            "wind_cell": wind_cell,
            "pv_cell": pv_cell,
            "wind_cf": wind_cf,
            "pv_cf": pv_cf,
            "installed_cap_data": installed_cap_data,
            "Params": Params,
            "LOAD_DEMAND": LOAD_DEMAND,
            "var_solar_cap": var_solar_cap,
            "var_wind_cap": var_wind_cap,
            "sum_solar_hours": sum_solar_hours,
            "sum_wind_hours": sum_wind_hours,
            "sum_solar_cell": sum_solar_cell,
            "sum_wind_cell": sum_wind_cell,
            "new_install_coal": new_install_coal,
            "load_shedding": load_shedding,
            "energy_h2": energy_h2,
            "energy_phs": energy_phs,
            "energy_bat": energy_bat,
            "tot_energy_h2": tot_energy_h2,
            "tot_energy_bat": tot_energy_bat,
            "tot_energy_phs": tot_energy_phs,
            "dischar_h2": dischar_h2,
            "dischar_bat": dischar_bat,
            "dischar_phs": dischar_phs,
            "charge_h2": charge_h2,
            "charge_bat": charge_bat,
            "charge_phs": charge_phs,
            "cap_dis_h2": cap_dis_h2,
            "cap_ch_h2": cap_ch_h2,
            "cap_bat": cap_bat,
            "cap_phs": cap_phs,
            "x_solar": x_solar,
            "x_wind": x_wind,
            "inter_solar": inter_solar,
            "inter_wind": inter_wind,
            "curtail_solar": curtail_solar,
            "curtail_wind": curtail_wind,
            "vre_curtail_summary": results_vre_curtail_summary,
            "trans_pair_in_AC_installed": trans_pair_in_AC_installed,
            "trans_pair_in_DC_installed": trans_pair_in_DC_installed,
            "trans_pair_in_DC": trans_pair_in_DC,
            "trans_pair_in_AC": trans_pair_in_AC,
            "ru": ru,
            "rd": rd,
            "trans_out": trans_out,
            "load_conv": load_conv,
            "trans_cap_DC": trans_cap_DC,
            "trans_cap_AC": trans_cap_AC,
            "load_trans_DC": load_trans_DC,
            "load_trans_AC": load_trans_AC,
            "load_trans_AC_installed": load_trans_AC_installed,
            "load_trans_DC_installed": load_trans_DC_installed,
            "trans_pair_out_AC": trans_pair_out_AC,
            "trans_pair_out_DC": trans_pair_out_DC,
            "trans_pair_out_DC_installed": trans_pair_out_DC_installed,
            "trans_pair_out_AC_installed": trans_pair_out_AC_installed,
        }
        audit_summary_row = {
            "solver_status": solver_audit.get("status_name"),
            "overall_ai_audit_status": overall_ai_audit_status,
            "effective_ai_scenario": ai_interface_metadata.get(
                "effective_ai_scenario"
            ),
            "s4_zone_runtime_transfer_enabled": bool(
                ai_s4_zone_runtime_transfer
                and ai_operational_scenario == "S4"
            ),
            "s4_zone_runtime_transfer_active": bool(
                ai_s4_zone_runtime_transfer_active
            ),
            "s4_temporal_audit_granularity": ai_interface_metadata.get(
                "ai_execution_audit", {}
            ).get("s4_temporal_audit_granularity"),
            "ai_penalty_in_objective": bool(include_ai_penalty_in_objective),
            "include_ai_penalty_in_objective": bool(include_ai_penalty_in_objective),
            "compute_ai_penalty_diagnostic": bool(compute_ai_penalty_diagnostic),
            "reported_power_system_cost": float(
                cost_components.get(
                    "reported_power_system_cost", cost_components["objective"]
                )
            ),
            "total_ai_preference_penalty": float(
                cost_components.get("total_ai_preference_penalty", 0.0)
            ),
            "network_tier_policy_enabled": bool(use_network_tier_caps),
            "ai_use_destination_tier_fallback": bool(
                ai_use_destination_tier_fallback
            ),
            "cluster_od_zone_combined_cap_min_margin_gw": (
                float(
                    min(
                        item["margin_gw"]
                        for item in ai_interface_metadata.get(
                            "cluster_od_zone_combined_cap_feasibility", {}
                        ).values()
                    )
                )
                if ai_interface_metadata.get(
                    "cluster_od_zone_combined_cap_feasibility"
                )
                else None
            ),
            "cluster_od_zone_combined_cap_all_feasible": (
                all(
                    item["margin_gw"] >= -1e-6
                    for item in ai_interface_metadata.get(
                        "cluster_od_zone_combined_cap_feasibility", {}
                    ).values()
                )
                if ai_interface_metadata.get(
                    "cluster_od_zone_combined_cap_feasibility"
                )
                else None
            ),
            "od_arc_min_capacity_ratio": (
                ai_interface_metadata.get("od_arc_capacity_audit", {}).get(
                    "min_cluster_capacity_ratio"
                )
            ),
            "od_arc_infeasible_cluster_count": int(
                len(
                    ai_interface_metadata.get("od_arc_capacity_audit", {}).get(
                        "infeasible_clusters", []
                    )
                )
            ),
            "usable_for_main_analysis": bool(usable_for_main_analysis),
            "requires_manual_review": bool(requires_manual_review),
            "hard_failed": bool(hard_failed),
            "ai_energy_audit_status": ai_energy_audit_status,
            "ai_energy_gap_gwh": float(ai_execution_gap_gwh),
            "ai_energy_abs_gap_gwh": float(ai_execution_abs_gap_gwh),
            "ai_energy_relative_abs_gap": float(ai_execution_relative_abs_gap),
            "ai_energy_abs_strict_passed": bool(ai_energy_abs_strict_passed),
            "ai_energy_rel_strict_passed": bool(ai_energy_rel_strict_passed),
            "ai_energy_abs_warn_passed": bool(ai_energy_abs_warn_passed),
            "ai_energy_rel_warn_passed": bool(ai_energy_rel_warn_passed),
            "ai_energy_abs_review_passed": bool(ai_energy_abs_review_passed),
            "ai_energy_rel_review_passed": bool(ai_energy_rel_review_passed),
            "cluster_od_allocation_audit_status": cluster_od_allocation_audit_status,
            "cluster_od_allocation_gap_gwh": (
                float(cluster_od_allocation_gap_gwh)
                if cluster_od_allocation_gap_gwh is not None
                else None
            ),
            "cluster_od_allocation_abs_gap_gwh": (
                float(cluster_od_allocation_abs_gap_gwh)
                if cluster_od_allocation_abs_gap_gwh is not None
                else None
            ),
            "cluster_od_allocation_relative_abs_gap": (
                float(cluster_od_allocation_relative_abs_gap)
                if cluster_od_allocation_relative_abs_gap is not None
                else None
            ),
            "cluster_od_allocation_audit_strict_passed": (
                cluster_od_allocation_audit_strict_passed
            ),
            "cluster_od_allocation_audit_not_failed": (
                cluster_od_allocation_audit_not_failed
            ),
            "destination_power_cap_audit_status": destination_power_cap_audit_status,
            "destination_power_cap_max_violation_gw": float(
                destination_power_cap_violation_gw
            ),
            "destination_power_cap_audit_strict_passed": bool(
                destination_power_cap_audit_strict_passed
            ),
            "destination_power_cap_audit_not_failed": bool(
                destination_power_cap_audit_not_failed
            ),
            "s4_no_advance_status": (
                no_advance_audit_status if ai_operational_scenario == "S4" else "NA"
            ),
            "s4_no_advance_max_violation_gwh": (
                float(no_advance) if ai_operational_scenario == "S4" else None
            ),
            "s4_no_advance_strict_passed": (
                no_advance_audit_strict_passed
                if ai_operational_scenario == "S4"
                else None
            ),
            "s4_no_advance_not_failed": (
                no_advance_audit_not_failed
                if ai_operational_scenario == "S4"
                else None
            ),
            "s4_deadline_status": (
                deadline_audit_status if ai_operational_scenario == "S4" else "NA"
            ),
            "s4_deadline_max_violation_gwh": (
                float(deadline) if ai_operational_scenario == "S4" else None
            ),
            "s4_deadline_strict_passed": (
                deadline_audit_strict_passed
                if ai_operational_scenario == "S4"
                else None
            ),
            "s4_deadline_not_failed": (
                deadline_audit_not_failed
                if ai_operational_scenario == "S4"
                else None
            ),
        }
        pd.DataFrame([audit_summary_row]).to_csv(
            os.path.join(res_dir, "audit_summary.csv"),
            index=False,
        )
        ai_od_flow_pkl_required = bool(
            use_source_cluster_ai_interface and ai_operational_scenario in {"S1", "S4"}
        )
        required_result_files = [
            os.path.join(res_dir, "ai_execution_audit.json"),
            os.path.join(res_dir, "energy_accounting.json"),
            os.path.join(res_dir, "cost_components.json"),
            os.path.join(res_dir, "ai_load_interface_metadata.json"),
            os.path.join(res_dir, "audit_summary.csv"),
            os.path.join(res_dir_pkl, "load_demand.pkl"),
            os.path.join(res_dir_pkl, "results_AI_load.pkl"),
            os.path.join(res_dir_pkl, "results_AI_batch_run.pkl"),
            os.path.join(res_dir_pkl, "results_AI_batch_cum.pkl"),
        ]
        if ai_od_flow_pkl_required:
            required_result_files.append(
                os.path.join(res_dir_pkl, "results_AI_od_flow.pkl")
            )
        if ai_od_flow_pkl_required:
            required_result_files.append(
                os.path.join(res_dir_pkl, "results_AI_rt_od_flow.pkl")
            )
        if ai_s4_zone_runtime_transfer_active:
            required_result_files.extend([
                os.path.join(res_dir_pkl, "results_AI_rt_zone_route.pkl"),
                os.path.join(res_dir_pkl, "results_AI_zone_batch_run.pkl"),
                os.path.join(res_dir, "s4_zone_runtime_transfer_audit.json"),
            ])
        required_result_files_missing = [
            path for path in required_result_files if not os.path.exists(path)
        ]
        run_status = {
            "solver_status": solver_audit.get("status_name"),
            "solver_acceptable": bool(
                model.status in acceptable_status and model.SolCount > 0
            ),
            "overall_ai_audit_status": overall_ai_audit_status,
            "ai_penalty_in_objective": bool(include_ai_penalty_in_objective),
            "include_ai_penalty_in_objective": bool(include_ai_penalty_in_objective),
            "compute_ai_penalty_diagnostic": bool(compute_ai_penalty_diagnostic),
            "recommended_cost_field": "reported_power_system_cost",
            "network_tier_policy_enabled": bool(use_network_tier_caps),
            "ai_od_network_policy_file": ai_od_network_policy_file or None,
            "ai_use_destination_tier_fallback": bool(
                ai_use_destination_tier_fallback
            ),
            "cluster_od_zone_combined_cap_all_feasible": (
                all(
                    item["margin_gw"] >= -1e-6
                    for item in ai_interface_metadata.get(
                        "cluster_od_zone_combined_cap_feasibility", {}
                    ).values()
                )
                if ai_interface_metadata.get(
                    "cluster_od_zone_combined_cap_feasibility"
                )
                else None
            ),
            "cluster_od_zone_combined_cap_min_margin_gw": (
                float(
                    min(
                        item["margin_gw"]
                        for item in ai_interface_metadata.get(
                            "cluster_od_zone_combined_cap_feasibility", {}
                        ).values()
                    )
                )
                if ai_interface_metadata.get(
                    "cluster_od_zone_combined_cap_feasibility"
                )
                else None
            ),
            "usable_for_main_analysis": bool(usable_for_main_analysis),
            "requires_manual_review": bool(requires_manual_review),
            "hard_failed": bool(hard_failed),
            "strict_postsolve_ai_audit": bool(strict_postsolve_ai_audit),
            "raise_on_structural_ai_audit_fail": bool(
                raise_on_structural_ai_audit_fail
            ),
            "slurm_should_fail_on_structural_ai_audit": bool(
                strict_postsolve_ai_audit
                and use_external_ai_load
                and hard_failed
                and raise_on_structural_ai_audit_fail
            ),
            "ai_execution_audit_written": os.path.exists(
                os.path.join(res_dir, "ai_execution_audit.json")
            ),
            "s4_zone_runtime_transfer_audit_written": os.path.exists(
                os.path.join(res_dir, "s4_zone_runtime_transfer_audit.json")
            ),
            "audit_summary_written": os.path.exists(
                os.path.join(res_dir, "audit_summary.csv")
            ),
            "s0_cluster_reconstruction_audit_written": os.path.exists(
                os.path.join(res_dir, "s0_cluster_reconstruction_audit.json")
            ),
            "energy_accounting_written": os.path.exists(
                os.path.join(res_dir, "energy_accounting.json")
            ),
            "cost_components_written": os.path.exists(
                os.path.join(res_dir, "cost_components.json")
            ),
            "ai_load_interface_metadata_written": os.path.exists(
                os.path.join(res_dir, "ai_load_interface_metadata.json")
            ),
            "pkl_dir_exists": os.path.isdir(res_dir_pkl),
            "key_ai_load_pkl_written": os.path.exists(
                os.path.join(res_dir_pkl, "results_AI_load.pkl")
            ),
            "key_ai_batch_run_pkl_written": os.path.exists(
                os.path.join(res_dir_pkl, "results_AI_batch_run.pkl")
            ),
            "key_ai_batch_cum_pkl_written": os.path.exists(
                os.path.join(res_dir_pkl, "results_AI_batch_cum.pkl")
            ),
            "key_ai_rt_zone_route_pkl_written": os.path.exists(
                os.path.join(res_dir_pkl, "results_AI_rt_zone_route.pkl")
            ),
            "key_ai_zone_batch_run_pkl_written": os.path.exists(
                os.path.join(res_dir_pkl, "results_AI_zone_batch_run.pkl")
            ),
            "ai_od_flow_expected": bool(
                use_source_cluster_ai_interface
                and ai_operational_scenario in {"S1", "S4"}
            ),
            "key_ai_od_flow_pkl_written": os.path.exists(
                os.path.join(res_dir_pkl, "results_AI_od_flow.pkl")
            ),
            "key_ai_od_flow_pkl_required": ai_od_flow_pkl_required,
            "key_load_demand_pkl_written": os.path.exists(
                os.path.join(res_dir_pkl, "load_demand.pkl")
            ),
            "key_cost_components_json_written": os.path.exists(
                os.path.join(res_dir, "cost_components.json")
            ),
            "required_result_files_count": int(len(required_result_files)),
            "required_result_files_missing_count": int(
                len(required_result_files_missing)
            ),
            "required_result_files_missing": required_result_files_missing,
            "required_result_files_complete": bool(
                len(required_result_files_missing) == 0
            ),
            "note": (
                "Strict AI audit thresholds are retained for reporting. "
                "Runs below structural hard-fail thresholds are completed with warnings. "
                "When FAIL_STRUCTURAL occurs, result files are still written; whether "
                "the process raises an exception is controlled by "
                "RAISE_ON_STRUCTURAL_AI_AUDIT_FAIL."
            ),
            "usable_for_main_analysis_rule": (
                "WARN_REVIEW_REQUIRED completes postprocessing but requires manual "
                "review before use in main analysis."
            ),
        }
        with open(os.path.join(res_dir, "run_status.json"), "w", encoding="utf-8") as f:
            json.dump(run_status, f, indent=2, ensure_ascii=False)
        if strict_postsolve_ai_audit and use_external_ai_load and hard_failed:
            msg = (
                "AI audit structural failure after postprocess completed: "
                f"overall_ai_audit_status={overall_ai_audit_status}; "
                f"ai_energy_audit_status={ai_energy_audit_status}; "
                f"cluster_od_allocation_audit_status={cluster_od_allocation_audit_status}; "
                f"destination_power_cap_audit_status={destination_power_cap_audit_status}; "
                f"s4_od_no_advance_audit_status="
                f"{no_advance_audit_status if ai_operational_scenario == 'S4' else 'NA'}; "
                f"s4_od_deadline_audit_status="
                f"{deadline_audit_status if ai_operational_scenario == 'S4' else 'NA'}. "
                "All diagnostic and result files have been written. "
                "See ai_execution_audit.json and run_status.json for details."
            )
            if raise_on_structural_ai_audit_fail:
                raise RuntimeError(msg)
            print("WARNING:", msg)
        return result_dict

    else:
        # No usable feasible solution. Write a minimal status file instead of
        # silently returning None (the original V4 silent-exit bug).
        os.makedirs(args.output_dir, exist_ok=True)
        res_dir = os.path.join(args.output_dir, args.Mode)
        os.makedirs(res_dir, exist_ok=True)
        fail_status = {
            "solver_status": solver_status_names.get(model.status, "OTHER"),
            "solver_terminal_status": int(model.status),
            "sol_count": int(model.SolCount),
            "solver_acceptable": False,
            "reason": (
                "Solver terminated without a usable feasible solution "
                "(status not in OPTIMAL/SUBOPTIMAL/TIME_LIMIT-with-solution). "
                "No result files written. Inspect solver log and tolerances."
            ),
            "required_result_files_complete": False,
        }
        with open(
            os.path.join(res_dir, "run_status.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(fail_status, f, indent=2, ensure_ascii=False)
        print(
            "ERROR: no usable feasible solution; wrote run_status.json only. "
            f"status={int(model.status)} SolCount={model.SolCount}"
        )
        return {}


if __name__ == "__main__":
    config = get_config()
    model_result = National_energy_model(config)
