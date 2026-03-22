from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import run as run_module


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _recompute_category(
    category_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    actual_path = category_dir / "inference_results.jsonl"
    result_path = category_dir / "inference_result.json"
    validation_config_path = category_dir / "validation_config.json"

    if not actual_path.exists() or not result_path.exists() or not validation_config_path.exists():
        return {"category_dir": str(category_dir), "updated": False, "reason": "missing_required_files"}

    result_payload = _read_json(result_path)
    validation_config = _read_json(validation_config_path)
    reference_path_value = validation_config.get("reference_inference_artifact")
    if not reference_path_value:
        return {
            "category_dir": str(category_dir),
            "updated": False,
            "reason": "missing_reference_inference_artifact",
        }

    reference_path = Path(reference_path_value)
    if not reference_path.exists():
        return {
            "category_dir": str(category_dir),
            "updated": False,
            "reason": "missing_reference_records",
        }

    expected_records = _iter_jsonl(reference_path)
    actual_records = _iter_jsonl(actual_path)
    pair_count = min(len(expected_records), len(actual_records))

    exact_match_count = sum(
        1
        for index in range(pair_count)
        if run_module._results_exact_match(expected_records[index], actual_records[index])
    )
    mismatch_count = pair_count - exact_match_count
    request_failure_count = sum(
        1
        for record in actual_records[:pair_count]
        if bool((record.get("metadata") or {}).get("request_failed"))
    )

    result_summary = result_payload.get("summary") if isinstance(result_payload.get("summary"), dict) else {}
    result_summary["n_prompts"] = pair_count
    result_summary["exact_match_count"] = exact_match_count
    result_summary["mismatch_count"] = mismatch_count
    result_summary["request_failure_count"] = request_failure_count
    result_payload["summary"] = result_summary
    result_payload.pop("records", None)
    result_payload.pop("records_omitted", None)
    result_payload.pop("records_count", None)

    if not dry_run:
        _write_json(result_path, result_payload)

    return {
        "category_dir": str(category_dir),
        "updated": True,
        "n_prompts": pair_count,
        "exact_match_count": exact_match_count,
        "mismatch_count": mismatch_count,
        "request_failure_count": request_failure_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute verification diff artifacts with shared distance logic.")
    parser.add_argument("--results-root", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results_root = args.results_root
    category_dirs = sorted(path for path in results_root.glob("inference_*") if path.is_dir())

    if not category_dirs:
        print(f"No inference_* directories found under {results_root}")
        return

    updated = 0
    skipped = 0
    for category_dir in category_dirs:
        outcome = _recompute_category(
            category_dir,
            dry_run=args.dry_run,
        )
        if outcome.get("updated"):
            updated += 1
            print(
                "UPDATED",
                category_dir.name,
                f"exact={outcome['exact_match_count']}",
                f"mismatch={outcome['mismatch_count']}",
                f"requests_failed={outcome['request_failure_count']}",
            )
        else:
            skipped += 1
            print("SKIPPED", category_dir.name, outcome.get("reason"))

    print(f"Done. updated={updated}, skipped={skipped}, dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
