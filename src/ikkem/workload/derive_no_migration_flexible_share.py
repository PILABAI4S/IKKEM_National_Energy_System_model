#!/usr/bin/env python3
"""Derive a no-migration 2020 AI baseline with a target flexible share."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def derive(input_csv: Path, output_csv: Path, flexible_share: float) -> None:
    if not 0.0 <= flexible_share <= 1.0:
        raise ValueError(f"flexible_share must be in [0, 1], got {flexible_share}")

    df = pd.read_csv(input_csv)
    required = {"fixed_ai_load_mw", "flexible_ai_load_mw"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    out = df.copy()
    total_ai_mw = out["fixed_ai_load_mw"].astype(float) + out[
        "flexible_ai_load_mw"
    ].astype(float)
    out["flexible_ai_load_mw"] = total_ai_mw * flexible_share
    out["fixed_ai_load_mw"] = total_ai_mw * (1.0 - flexible_share)

    if "allocation_version" in out.columns:
        suffix = f"flex{int(round(flexible_share * 100)):02d}"
        out["allocation_version"] = (
            out["allocation_version"].astype(str).str.replace(
                r"_flex\d+$", "", regex=True
            )
            + f"_{suffix}"
        )
    if "ai_local_retention_min" in out.columns:
        out["ai_local_retention_min"] = 1.0
    if "ai_hosting_upper_bound_mw" in out.columns:
        out["ai_hosting_upper_bound_mw"] = out["flexible_ai_load_mw"]
    if "local_retention_floor_mw" in out.columns:
        out["local_retention_floor_mw"] = out["flexible_ai_load_mw"]
    if "evidence_cap_raw_share" in out.columns:
        pool = float(out["flexible_ai_load_mw"].sum())
        out["evidence_cap_raw_share"] = (
            out["flexible_ai_load_mw"] / pool if pool > 0 else 0.0
        )
    if "ai_max_host_share" in out.columns:
        pool = float(out["flexible_ai_load_mw"].sum())
        out["ai_max_host_share"] = (
            out["flexible_ai_load_mw"] / pool if pool > 0 else 0.0
        )
    if "method_note" in out.columns:
        out["method_note"] = (
            "AI-BAU no-migration baseline derived from the same provincial total "
            f"AI load with flexible_share={flexible_share:.3f}. Each province "
            "must host exactly its own flexible AI demand via local-retention "
            "and hosting-upper-bound equality."
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    total_gw = float(total_ai_mw.sum()) / 1000.0
    flex_gw = float(out["flexible_ai_load_mw"].sum()) / 1000.0
    fixed_gw = float(out["fixed_ai_load_mw"].sum()) / 1000.0
    print("DERIVE_NO_MIGRATION_FLEXIBLE_SHARE_OK")
    print(f"input={input_csv}")
    print(f"output={output_csv}")
    print(f"flexible_share={flexible_share:.6f}")
    print(f"total_ai_gw={total_gw:.12f}")
    print(f"fixed_ai_gw={fixed_gw:.12f}")
    print(f"flexible_ai_gw={flex_gw:.12f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--flexible-share", type=float, required=True)
    args = parser.parse_args()
    derive(args.input_csv, args.output_csv, args.flexible_share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
