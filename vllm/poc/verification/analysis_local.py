from __future__ import annotations

import json
import os
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from metrics_local import distance2 as shared_distance2


@dataclass(frozen=True)
class SessionSummary:
	artifact_dir: Path
	bucket: str
	label: str
	inference_model: str
	reference_model: str
	timestamp: str | None
	raw_records: int
	matched_pairs: int
	unmatched_actual: int
	unmatched_reference: int
	used_items: int
	dropped_token_mismatch: int
	total_time_seconds: float | None
	average_time_per_prompt_seconds: float | None
	output_tokens_per_second: float | None
	mean_distance: float | None
	median_distance: float | None
	max_distance: float | None


def load_json(path: Path) -> Any:
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	with path.open("r", encoding="utf-8") as handle:
		for line in handle:
			line = line.strip()
			if line:
				rows.append(json.loads(line))
	return rows


def default_verification_dir() -> Path:
	env_path = os.environ.get("VLLM_VERIFICATION_DIR", "").strip()
	if env_path:
		candidate = Path(env_path).expanduser().resolve()
		if candidate.exists():
			return candidate

	module_dir = Path(__file__).resolve().parent
	candidates = [
		module_dir,
		Path.cwd() / "vllm" / "poc" / "verification",
		Path.cwd() / "poc" / "verification",
		Path.cwd(),
	]
	for candidate in candidates:
		if candidate.exists():
			return candidate.resolve()
	return module_dir


def compute_length(item: dict[str, Any], length_mode: str = "text_chars") -> int:
	inference_result = item.get("inference_result") or {}
	if length_mode == "text_chars":
		return len(str(inference_result.get("text", "")))
	if length_mode == "generated_tokens":
		return len(inference_result.get("results") or [])
	raise ValueError(f"Unsupported length_mode: {length_mode}")


def distance2(
	inference_result: dict[str, Any],
	validation_result: dict[str, Any],
	distance_floor_tokens: int = 100,
) -> tuple[float, float]:
	return shared_distance2(
		inference_result,
		validation_result,
		distance_floor_tokens=distance_floor_tokens,
	)


def process_data(
	items: list[dict[str, Any]],
	distance_floor_tokens: int = 100,
	length_mode: str = "text_chars",
	token_mismatch_policy: str = "drop",
) -> tuple[list[dict[str, Any]], list[float], list[float], list[int], int]:
	distances: list[float] = []
	top_k_matches_ratios: list[float] = []
	lengths: list[int] = []
	cleaned_items: list[dict[str, Any]] = []
	dropped_token_mismatch = 0

	for item in items:
		distance, top_k_matches_ratio = distance2(
			item["inference_result"],
			item["validation_result"],
			distance_floor_tokens=distance_floor_tokens,
		)
		if distance == -1.0:
			if token_mismatch_policy == "drop":
				dropped_token_mismatch += 1
				continue
			if token_mismatch_policy == "max_distance":
				distance = 1.0
				top_k_matches_ratio = 0.0
			else:
				raise ValueError(
					f"Unsupported token_mismatch_policy: {token_mismatch_policy}"
				)

		length = compute_length(item, length_mode=length_mode)
		item_with_metrics = dict(item)
		item_with_metrics["computed_distance"] = distance
		item_with_metrics["computed_length"] = length
		cleaned_items.append(item_with_metrics)
		distances.append(distance)
		top_k_matches_ratios.append(top_k_matches_ratio)
		lengths.append(length)

	print(
		f"Processed {len(cleaned_items)} / {len(items)} items "
		f"(dropped token mismatches: {dropped_token_mismatch})"
	)
	return cleaned_items, distances, top_k_matches_ratios, lengths, dropped_token_mismatch


def _artifact_timestamp(name: str) -> str | None:
	parts = name.split("_")
	if len(parts) < 4:
		return None
	raw_value = f"{parts[-2]}_{parts[-1]}"
	try:
		parsed = datetime.strptime(raw_value, "%Y%m%d_%H%M%S")
		return parsed.strftime("%Y-%m-%d %H:%M:%S")
	except ValueError:
		return raw_value


def _artifact_kind(name: str) -> str | None:
	parts = name.split("_")
	if len(parts) < 2:
		return None
	return parts[1]


def _group_label(bucket: str, artifact_dir: Path) -> str:
	kind = _artifact_kind(artifact_dir.name) or artifact_dir.name
	timestamp = artifact_dir.name.split("_", 2)[-1] if kind != artifact_dir.name else artifact_dir.name
	return f"{bucket}_{kind}_{timestamp}"


def discover_artifact_dirs(results_root: Path | None = None) -> dict[str, list[Path]]:
	verification_dir = default_verification_dir()
	root = results_root.resolve() if results_root is not None else (verification_dir / "results").resolve()
	if not root.exists():
		raise FileNotFoundError(f"Results root does not exist: {root}")

	buckets: dict[str, list[Path]] = {"honest": [], "fraud": []}
	for artifact_dir in sorted(root.glob("inference_*")):
		if not artifact_dir.is_dir():
			continue
		kind = _artifact_kind(artifact_dir.name)
		if kind == "fp8":
			buckets["honest"].append(artifact_dir)
		elif kind == "int4":
			buckets["fraud"].append(artifact_dir)
	return buckets


def _load_reference_bundle(reference_config_dir: Path) -> tuple[list[dict[str, Any]], str]:
	records_path = reference_config_dir / "inference_results.jsonl"
	config_path = reference_config_dir / "inference_config.json"
	if not records_path.exists():
		raise FileNotFoundError(f"Missing FP8 reference records: {records_path}")
	if not config_path.exists():
		raise FileNotFoundError(f"Missing FP8 reference config: {config_path}")

	config = load_json(config_path)
	reference_model = str((config.get("model_info") or {}).get("name") or "").strip()
	return load_jsonl(records_path), reference_model


def _record_key(record: dict[str, Any]) -> tuple[str, str]:
	return str(record.get("prompt") or ""), str(record.get("language") or "")


def pair_actual_to_reference(
	actual_records: list[dict[str, Any]],
	reference_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
	reference_index: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
	for record in reference_records:
		reference_index[_record_key(record)].append(record)

	items: list[dict[str, Any]] = []
	unmatched_actual = 0
	for actual_record in actual_records:
		key = _record_key(actual_record)
		if not reference_index[key]:
			unmatched_actual += 1
			continue
		reference_record = reference_index[key].popleft()
		items.append(
			{
				"prompt": actual_record.get("prompt"),
				"language": actual_record.get("language"),
				"inference_result": actual_record.get("inference_result") or {},
				"validation_result": reference_record.get("inference_result") or {},
				"inference_model": actual_record.get("inference_model") or {},
				"validation_model": reference_record.get("inference_model") or {},
				"metadata": {
					"actual_metadata": actual_record.get("metadata") or {},
					"reference_metadata": reference_record.get("metadata") or {},
				},
			}
		)

	unmatched_reference = sum(len(queue) for queue in reference_index.values())
	match_stats = {
		"raw_records": len(actual_records),
		"matched_pairs": len(items),
		"unmatched_actual": unmatched_actual,
		"unmatched_reference": unmatched_reference,
	}
	return items, match_stats


def _coerce_float(value: Any) -> float | None:
	if isinstance(value, (int, float)):
		return float(value)
	return None


def _session_summary_from_dataset(dataset: dict[str, Any]) -> SessionSummary:
	distances = dataset["distances"]
	performance = dataset["performance"]
	return SessionSummary(
		artifact_dir=dataset["artifact_dir"],
		bucket=dataset["bucket"],
		label=dataset["label"],
		inference_model=dataset["inference_model"],
		reference_model=dataset["reference_model"],
		timestamp=dataset["timestamp"],
		raw_records=dataset["match_stats"]["raw_records"],
		matched_pairs=dataset["match_stats"]["matched_pairs"],
		unmatched_actual=dataset["match_stats"]["unmatched_actual"],
		unmatched_reference=dataset["match_stats"]["unmatched_reference"],
		used_items=len(dataset["items"]),
		dropped_token_mismatch=dataset["dropped_token_mismatch"],
		total_time_seconds=_coerce_float(performance.get("total_time_seconds")),
		average_time_per_prompt_seconds=_coerce_float(
			performance.get("average_time_per_prompt_seconds")
		),
		output_tokens_per_second=_coerce_float(
			performance.get("output_tokens_per_second")
		),
		mean_distance=float(np.mean(distances)) if distances else None,
		median_distance=float(np.median(distances)) if distances else None,
		max_distance=float(np.max(distances)) if distances else None,
	)


def load_session_dataset(
	artifact_dir: Path,
	reference_records: list[dict[str, Any]],
	reference_model: str,
	distance_floor_tokens: int = 100,
	length_mode: str = "text_chars",
	token_mismatch_policy: str = "drop",
) -> dict[str, Any]:
	validation_cfg = load_json(artifact_dir / "validation_config.json")
	actual_records = load_jsonl(artifact_dir / "inference_results.jsonl")
	items, match_stats = pair_actual_to_reference(actual_records, reference_records)
	processed_items, distances, top_k_matches_ratios, lengths, dropped_token_mismatch = process_data(
		items,
		distance_floor_tokens=distance_floor_tokens,
		length_mode=length_mode,
		token_mismatch_policy=token_mismatch_policy,
	)

	inference_model = str(
		(validation_cfg.get("validation_model_info") or {}).get("name")
		or ((actual_records[0].get("inference_model") or {}).get("name") if actual_records else "")
	).strip()
	bucket = "honest" if _artifact_kind(artifact_dir.name) == "fp8" else "fraud"
	dataset = {
		"artifact_dir": artifact_dir,
		"bucket": bucket,
		"label": _group_label(bucket, artifact_dir),
		"timestamp": _artifact_timestamp(artifact_dir.name),
		"inference_model": inference_model,
		"reference_model": reference_model,
		"items": processed_items,
		"distances": distances,
		"top_k_matches_ratios": top_k_matches_ratios,
		"lengths": lengths,
		"match_stats": match_stats,
		"dropped_token_mismatch": dropped_token_mismatch,
		"performance": validation_cfg.get("performance") or {},
	}
	dataset["summary"] = _session_summary_from_dataset(dataset)
	return dataset


def load_benchmark_datasets(
	results_root: Path | None = None,
	reference_config_dir: Path | None = None,
	distance_floor_tokens: int = 100,
	length_mode: str = "text_chars",
	token_mismatch_policy: str = "drop",
) -> dict[str, list[dict[str, Any]]]:
	verification_dir = default_verification_dir()
	reference_dir = (
		reference_config_dir.resolve()
		if reference_config_dir is not None
		else (verification_dir / "configs" / "inference_fp8").resolve()
	)
	reference_records, reference_model = _load_reference_bundle(reference_dir)
	buckets = discover_artifact_dirs(results_root=results_root)

	datasets: dict[str, list[dict[str, Any]]] = {"honest": [], "fraud": []}
	for bucket, artifact_dirs in buckets.items():
		for artifact_dir in artifact_dirs:
			datasets[bucket].append(
				load_session_dataset(
					artifact_dir=artifact_dir,
					reference_records=reference_records,
					reference_model=reference_model,
					distance_floor_tokens=distance_floor_tokens,
					length_mode=length_mode,
					token_mismatch_policy=token_mismatch_policy,
				)
			)
	return datasets


def classify_data(distances: list[float], lower_bound: float, upper_bound: float) -> list[str]:
	classifications: list[str] = []
	for distance in distances:
		if distance < lower_bound:
			classifications.append("accepted")
		elif distance > upper_bound:
			classifications.append("fraud")
		else:
			classifications.append("questionable")
	return classifications


def _binary_f1(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
	tp = int(np.sum((labels_true == 1) & (labels_pred == 1)))
	fp = int(np.sum((labels_true == 0) & (labels_pred == 1)))
	fn = int(np.sum((labels_true == 1) & (labels_pred == 0)))
	if tp == 0:
		return 0.0
	precision = tp / (tp + fp) if (tp + fp) else 0.0
	recall = tp / (tp + fn) if (tp + fn) else 0.0
	if precision == 0.0 or recall == 0.0:
		return 0.0
	return 2.0 * precision * recall / (precision + recall)


def evaluate_bound(
	lower: float,
	upper_candidates: np.ndarray,
	distances_val: np.ndarray,
	distances_quant: np.ndarray,
) -> tuple[float, float, float] | None:
	if np.any(distances_val > lower):
		return None

	all_distances = np.concatenate([distances_val, distances_quant])
	labels_true = np.array([0] * len(distances_val) + [1] * len(distances_quant), dtype=int)
	best_f1 = -1.0
	optimal_upper: float | None = None
	for upper in upper_candidates:
		labels_pred = np.where(all_distances < lower, 0, 1)
		labels_pred[(all_distances >= lower) & (all_distances <= upper)] = 1
		current_f1 = _binary_f1(labels_true, labels_pred)
		if current_f1 > best_f1:
			best_f1 = current_f1
			optimal_upper = float(upper)

	if optimal_upper is None:
		return None
	return float(lower), optimal_upper, float(best_f1)


def find_optimal_bounds(
	honest_distances: list[float],
	fraud_distances: list[float],
	step: float = 0.001,
) -> tuple[float, float]:
	distances_val = np.asarray(honest_distances, dtype=float)
	distances_quant = np.asarray(fraud_distances, dtype=float)
	if distances_val.size == 0 or distances_quant.size == 0:
		raise ValueError("Need both honest and fraud distances to find optimal bounds")

	all_distances = np.concatenate([distances_val, distances_quant])
	min_dist = float(all_distances.min())
	max_dist = float(all_distances.max())
	search_space = np.arange(min_dist, max_dist + step, step)

	results: list[tuple[float, float, float]] = []
	for lower in search_space:
		candidate = evaluate_bound(
			lower=float(lower),
			upper_candidates=search_space[search_space > lower],
			distances_val=distances_val,
			distances_quant=distances_quant,
		)
		if candidate is not None:
			results.append(candidate)

	if not results:
		raise ValueError(
			"No valid bounds found under the constraint that no honest distances exceed the lower bound"
		)

	optimal_lower, optimal_upper, best_f1 = max(results, key=lambda row: row[2])
	print(f"Optimal Lower Bound: {optimal_lower:.6f}")
	print(f"Optimal Upper Bound: {optimal_upper:.6f}")
	print(f"Best F1-Score: {best_f1:.4f}")
	return optimal_lower, optimal_upper


def build_threshold_payload(
	results_root: Path | None = None,
	reference_config_dir: Path | None = None,
	distance_floor_tokens: int = 100,
	length_mode: str = "text_chars",
	token_mismatch_policy: str = "drop",
	step: float = 0.001,
) -> dict[str, Any]:
	datasets = load_benchmark_datasets(
		results_root=results_root,
		reference_config_dir=reference_config_dir,
		distance_floor_tokens=distance_floor_tokens,
		length_mode=length_mode,
		token_mismatch_policy=token_mismatch_policy,
	)

	honest_items_dict: dict[str, list[dict[str, Any]]] = {}
	honest_distances_dict: dict[str, list[float]] = {}
	fraud_items_dict: dict[str, list[dict[str, Any]]] = {}
	fraud_distances_dict: dict[str, list[float]] = {}
	honest_distances: list[float] = []
	fraud_distances: list[float] = []
	session_summaries: list[SessionSummary] = []

	for dataset in datasets["honest"]:
		honest_items_dict[dataset["label"]] = dataset["items"]
		honest_distances_dict[dataset["label"]] = dataset["distances"]
		honest_distances.extend(dataset["distances"])
		session_summaries.append(dataset["summary"])

	for dataset in datasets["fraud"]:
		fraud_items_dict[dataset["label"]] = dataset["items"]
		fraud_distances_dict[dataset["label"]] = dataset["distances"]
		fraud_distances.extend(dataset["distances"])
		session_summaries.append(dataset["summary"])

	optimal_lower: float | None = None
	optimal_upper: float | None = None
	honest_classifications: list[str] = []
	fraud_classifications: list[str] = []
	share_of_fraud_found = 0.0
	if honest_distances and fraud_distances:
		optimal_lower, optimal_upper = find_optimal_bounds(
			honest_distances,
			fraud_distances,
			step=step,
		)
		honest_classifications = classify_data(honest_distances, optimal_lower, optimal_upper)
		fraud_classifications = classify_data(fraud_distances, optimal_lower, optimal_upper)
		if fraud_classifications:
			share_of_fraud_found = float(
				np.mean(np.array(fraud_classifications, dtype=object) == "fraud")
			)

	return {
		"datasets": datasets,
		"honest_items_dict": honest_items_dict,
		"honest_distances_dict": honest_distances_dict,
		"fraud_items_dict": fraud_items_dict,
		"fraud_distances_dict": fraud_distances_dict,
		"honest_distances": honest_distances,
		"fraud_distances": fraud_distances,
		"optimal_lower": optimal_lower,
		"optimal_upper": optimal_upper,
		"honest_classifications": honest_classifications,
		"fraud_classifications": fraud_classifications,
		"share_of_fraud_found": share_of_fraud_found,
		"session_summaries": session_summaries,
	}


def session_summaries_to_rows(session_summaries: list[SessionSummary]) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	for summary in session_summaries:
		rows.append(
			{
				"bucket": summary.bucket,
				"label": summary.label,
				"artifact_dir": str(summary.artifact_dir),
				"timestamp": summary.timestamp,
				"inference_model": summary.inference_model,
				"reference_model": summary.reference_model,
				"raw_records": summary.raw_records,
				"matched_pairs": summary.matched_pairs,
				"unmatched_actual": summary.unmatched_actual,
				"unmatched_reference": summary.unmatched_reference,
				"used_items": summary.used_items,
				"dropped_token_mismatch": summary.dropped_token_mismatch,
				"total_time_seconds": summary.total_time_seconds,
				"average_time_per_prompt_seconds": summary.average_time_per_prompt_seconds,
				"output_tokens_per_second": summary.output_tokens_per_second,
				"mean_distance": summary.mean_distance,
				"median_distance": summary.median_distance,
				"max_distance": summary.max_distance,
			}
		)
	return rows


def plot_classification_results(
	distances: list[float],
	classifications: list[str],
	lower_bound: float,
	upper_bound: float,
	title_prefix: str = "",
	languages: list[str] | None = None,
) -> None:
	classification_counts = Counter(classifications)
	labels = ["accepted", "questionable", "fraud"]
	counts = [classification_counts.get(label, 0) for label in labels]

	plt.figure(figsize=(14, 6))

	plt.subplot(1, 2, 1)
	plt.bar(labels, counts, color=["green", "orange", "red"])
	plt.title(f"{title_prefix} Classification Counts")
	plt.xlabel("Classification")
	plt.ylabel("Count")

	plt.subplot(1, 2, 2)
	color_map = {"accepted": "green", "questionable": "orange", "fraud": "red"}
	marker_size = 36

	if languages is not None:
		if len(languages) != len(classifications) or len(distances) != len(classifications):
			raise ValueError("Lengths of languages, classifications, and distances must match")

		seen_languages: list[str] = []
		for language in languages:
			if language not in seen_languages:
				seen_languages.append(language)

		fixed_marker_map = {"sp": "^", "en": "o", "ch": "s", "ar": "D", "hi": "P", "unk": "X"}
		fallback_markers = ["v", "*", "h", "<", ">", "8"]
		marker_map = dict(fixed_marker_map)
		fallback_index = 0
		for language in seen_languages:
			if language not in marker_map:
				marker_map[language] = fallback_markers[fallback_index % len(fallback_markers)]
				fallback_index += 1

		for cls in labels:
			class_indices = [index for index, value in enumerate(classifications) if value == cls]
			if not class_indices:
				continue
			for language in seen_languages:
				indices = [index for index in class_indices if languages[index] == language]
				if not indices:
					continue
				plt.scatter(
					indices,
					[distances[index] for index in indices],
					c=color_map[cls],
					marker=marker_map[language],
					alpha=0.6,
					s=marker_size,
				)

		class_handles = [
			Line2D(
				[0],
				[0],
				marker="o",
				color=color_map[label],
				linestyle="None",
				markersize=8,
				label=f"{label.capitalize()} ({classification_counts.get(label, 0)})",
			)
			for label in labels
		]
		language_name_map = {"sp": "Spanish", "en": "English", "ch": "Chinese", "ar": "Arabic", "hi": "Hindi", "unk": "Unknown"}
		language_handles = [
			Line2D(
				[0],
				[0],
				marker=marker_map[language],
				color="black",
				linestyle="None",
				markersize=8,
				label=language_name_map.get(language, language),
			)
			for language in seen_languages
		]
		legend1 = plt.legend(handles=class_handles, title="Classification", loc="upper left")
		plt.gca().add_artist(legend1)
		plt.legend(handles=language_handles, title="Languages", loc="upper right")
	else:
		for label in classification_counts:
			indices = [index for index, value in enumerate(classifications) if value == label]
			plt.scatter(
				indices,
				[distances[index] for index in indices],
				c=color_map[label],
				alpha=0.5,
				s=marker_size,
				label=f"{label.capitalize()} ({classification_counts[label]})",
			)

	plt.axhline(lower_bound, color="blue", linestyle="--", label="_nolegend_")
	plt.axhline(upper_bound, color="purple", linestyle="--", label="_nolegend_")

	if languages is None:
		plt.legend(loc="upper right")

	bounds_handles = [
		Line2D([0], [0], color="blue", linestyle="--", linewidth=2, label=f"Lower: {lower_bound:.6f}"),
		Line2D([0], [0], color="purple", linestyle="--", linewidth=2, label=f"Upper: {upper_bound:.6f}"),
	]
	if languages is not None:
		plt.legend(handles=bounds_handles, title="Bounds", loc="lower right")
	else:
		legend2 = plt.legend(handles=bounds_handles, title="Bounds", loc="center right")
		plt.gca().add_artist(legend2)

	plt.title(f"{title_prefix} Distances Classification")
	plt.xlabel("Item Index")
	plt.ylabel("Distance")
	plt.tight_layout()
	plt.show()


def plot_length_vs_distance_comparison(
	name: str,
	honest_items_dict: dict[str, list[dict[str, Any]]],
	honest_distances_dict: dict[str, list[float]],
	fraud_items_dict: dict[str, list[dict[str, Any]]],
	fraud_distances_dict: dict[str, list[float]],
	bounds: tuple[float, float] | None = None,
) -> None:
	honest_keys = list(honest_items_dict.keys())
	fraud_keys = list(fraud_items_dict.keys())

	if set(honest_keys) != set(honest_distances_dict.keys()):
		raise ValueError("honest_items_dict and honest_distances_dict must have the same keys")
	if set(fraud_keys) != set(fraud_distances_dict.keys()):
		raise ValueError("fraud_items_dict and fraud_distances_dict must have the same keys")

	honest_lengths = {
		key: [int(item.get("computed_length", compute_length(item))) for item in honest_items_dict[key]]
		for key in honest_keys
	}
	fraud_lengths = {
		key: [int(item.get("computed_length", compute_length(item))) for item in fraud_items_dict[key]]
		for key in fraud_keys
	}

	fig, ax = plt.subplots(figsize=(10, 6))
	marker_size = 36
	honest_palette = ["#0B3D91", "#87CEFA", "#20B2AA"]
	honest_color_by_key = {
		key: honest_palette[index % len(honest_palette)]
		for index, key in enumerate(honest_keys)
	}
	if fraud_keys:
		fraud_palette = [plt.cm.Reds(value) for value in np.linspace(0.8, 0.4, len(fraud_keys))]
	else:
		fraud_palette = []
	fraud_color_by_key = {key: fraud_palette[index] for index, key in enumerate(fraud_keys)}

	def _norm_lang(item: dict[str, Any]) -> str:
		return str(item.get("language") or "unk")

	all_languages: list[str] = []
	for source in [honest_items_dict, fraud_items_dict]:
		for items in source.values():
			for item in items:
				language = _norm_lang(item)
				if language not in all_languages:
					all_languages.append(language)

	fixed_marker_map = {"sp": "^", "en": "o", "ch": "s", "ar": "D", "hi": "P", "unk": "X"}
	marker_map = dict(fixed_marker_map)
	fallback_markers = ["v", "*", "h", "<", ">", "8"]
	fallback_index = 0
	for language in all_languages:
		if language not in marker_map:
			marker_map[language] = fallback_markers[fallback_index % len(fallback_markers)]
			fallback_index += 1

	def _scatter_group(
		group_name: str,
		items_dict: dict[str, list[dict[str, Any]]],
		lengths_dict: dict[str, list[int]],
		distances_dict: dict[str, list[float]],
		color: Any,
		legend_label: str,
	) -> Line2D:
		languages = [_norm_lang(item) for item in items_dict[group_name]]
		xs = lengths_dict[group_name]
		ys = distances_dict[group_name]
		for language in all_languages:
			indices = [index for index, value in enumerate(languages) if value == language]
			if not indices:
				continue
			ax.scatter(
				[xs[index] for index in indices],
				[ys[index] for index in indices],
				color=color,
				marker=marker_map[language],
				alpha=0.6,
				s=marker_size,
			)

		return Line2D(
			[0],
			[0],
			marker="o",
			color="w",
			markerfacecolor=color,
			markeredgecolor=color,
			markersize=8,
			linestyle="None",
			label=legend_label,
		)

	group_handles: list[Line2D] = []
	for key in honest_keys:
		group_handles.append(
			_scatter_group(
				key,
				honest_items_dict,
				honest_lengths,
				honest_distances_dict,
				honest_color_by_key[key],
				f"Honest - {key}",
			)
		)
	for key in fraud_keys:
		group_handles.append(
			_scatter_group(
				key,
				fraud_items_dict,
				fraud_lengths,
				fraud_distances_dict,
				fraud_color_by_key[key],
				f"Fraud - {key}",
			)
		)

	if group_handles:
		legend1 = ax.legend(handles=group_handles, title="Groups", loc="upper left", fontsize=8)
		ax.add_artist(legend1)

	language_name_map = {"sp": "Spanish", "en": "English", "ch": "Chinese", "ar": "Arabic", "hi": "Hindi", "unk": "Unknown"}
	language_handles = [
		Line2D(
			[0],
			[0],
			marker=marker_map[language],
			color="black",
			linestyle="None",
			markersize=8,
			label=language_name_map.get(language, language),
		)
		for language in all_languages
	]
	if language_handles:
		legend2 = ax.legend(handles=language_handles, title="Languages", loc="upper right", fontsize=8)
		ax.add_artist(legend2)

	if bounds is not None:
		lower, upper = bounds
		ax.axhline(lower, color="blue", linestyle="--", linewidth=1.8)
		ax.axhline(upper, color="purple", linestyle="--", linewidth=1.8)
		bounds_handles = [
			Line2D([0], [0], color="blue", linestyle="--", linewidth=2, label=f"Lower: {lower:.6f}"),
			Line2D([0], [0], color="purple", linestyle="--", linewidth=2, label=f"Upper: {upper:.6f}"),
		]
		ax.legend(handles=bounds_handles, title="Bounds", loc="lower right", fontsize=8)

	ax.set_title(f"{name} - Length vs Distance Comparison")
	ax.set_xlabel("Length (characters)")
	ax.set_ylabel("Distance")
	ax.grid(True, alpha=0.3)
	fig.tight_layout()
	plt.show()


def plot_violin_comparison(
	distributions_dict: dict[str, list[float]],
	title: str = "Distance Distributions",
	ylabel: str = "Distance",
	figsize: tuple[int, int] = (10, 6),
	show: bool = True,
	ylim: tuple[float, float] | None = None,
) -> None:
	if not isinstance(distributions_dict, dict) or not distributions_dict:
		print("Nothing to plot: empty or invalid input.")
		return

	group_names: list[str] = []
	group_values: list[np.ndarray] = []
	for name, values in distributions_dict.items():
		array = np.asarray(values, dtype=float)
		array = array[np.isfinite(array)]
		if array.size == 0:
			continue
		group_names.append(name)
		group_values.append(array)

	if not group_values:
		print("Nothing to plot: all groups are empty after cleaning.")
		return

	fig, ax = plt.subplots(figsize=figsize)
	parts = ax.violinplot(
		group_values,
		positions=range(len(group_names)),
		showmeans=False,
		showmedians=True,
		showextrema=True,
	)
	for body in parts["bodies"]:
		body.set_facecolor("#8dd3c7")
		body.set_alpha(0.7)
		body.set_edgecolor("black")
		body.set_linewidth(1)

	for part_name in ("cbars", "cmins", "cmaxes", "cmedians"):
		if part_name in parts:
			artist = parts[part_name]
			artist.set_edgecolor("black")
			artist.set_linewidth(1)

	ax.set_title(title)
	ax.set_xlabel("Group")
	ax.set_ylabel(ylabel)
	ax.set_xticks(range(len(group_names)))
	ax.set_xticklabels(group_names, rotation=45, ha="right")
	if ylim is not None:
		ax.set_ylim(ylim)
	plt.tight_layout()
	if show:
		plt.show()

	print("Mean and std per group:")
	for name, array in sorted(zip(group_names, group_values), key=lambda item: float(np.mean(item[1]))):
		mean = float(np.mean(array))
		std = float(np.std(array, ddof=1)) if array.size > 1 else float("nan")
		print(f"  {name}: n={int(array.size)}, mean={mean:.6f}, std={std:.6f}")


plot_length_vs_distance = plot_length_vs_distance_comparison
