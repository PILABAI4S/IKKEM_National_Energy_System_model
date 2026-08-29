"""Inspect an external AI-load table without a paper configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from ikkem.workload.ai_load_interface import load_external_ai_load


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--provinces", nargs="+", required=True)
    parser.add_argument("--hours", type=int, required=True)
    args = parser.parse_args()

    workload = load_external_ai_load(args.input_csv, args.provinces, args.hours)
    print(f"flexible_pool_gw={workload.flexible_pool_gw:.6f}")
    print(f"province_count={len(workload.fixed_ai_load_mw)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
