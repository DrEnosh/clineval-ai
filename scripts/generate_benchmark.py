"""Generate the synthetic benchmark and write it to data/synthetic/.

Usage:
    python scripts/generate_benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.benchmark import BENCHMARK_VERSION, generate_default_benchmark, save_benchmark
from app.core.config import settings


def main() -> int:
    cases = generate_default_benchmark()
    out_path = settings.benchmark_dir / f"benchmark_{BENCHMARK_VERSION}.json"
    save_benchmark(cases, out_path)

    by_type: dict[str, int] = {}
    for case in cases:
        by_type[case.case_type.value] = by_type.get(case.case_type.value, 0) + 1

    print(f"Wrote {len(cases)} synthetic benchmark cases to {out_path}")
    print(f"Benchmark version: {BENCHMARK_VERSION}")
    for case_type, count in sorted(by_type.items()):
        print(f"  {case_type:<26} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
