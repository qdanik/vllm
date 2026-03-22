from __future__ import annotations

from typing import Any


def coerce_float(value: Any) -> float | None:
	if isinstance(value, (int, float)):
		return float(value)
	return None


def token_distance2(
	inf_position_logprobs: dict[str, Any],
	val_position_logprobs: dict[str, Any],
) -> tuple[float, int]:
	inf_map = inf_position_logprobs.get("logprobs") or {}
	val_map = val_position_logprobs.get("logprobs") or {}
	dist = 0.0
	n_matches = 0

	if not val_map:
		return float(len(inf_map)), 0

	sorted_logprobs = sorted(val_map.values())
	if len(sorted_logprobs) >= 2:
		min_val_logprob_1 = sorted_logprobs[0]
		min_val_logprob_2 = sorted_logprobs[1]
	else:
		min_val_logprob_1 = sorted_logprobs[0]
		min_val_logprob_2 = min_val_logprob_1 - 1.0

	for token, inf_logprob in inf_map.items():
		inf_logprob = coerce_float(inf_logprob)
		if inf_logprob is None:
			continue
		if token in val_map:
			val_logprob = coerce_float(val_map[token])
			n_matches += 1
		else:
			val_logprob = min_val_logprob_1 - (min_val_logprob_2 - min_val_logprob_1)

		if val_logprob is None:
			continue
		denom = 1e-10 + abs(inf_logprob) + abs(val_logprob)
		dist += abs(inf_logprob - val_logprob) / denom / 2.0

	return dist, n_matches


def distance2_from_results(
	inf_results: list[dict[str, Any]],
	val_results: list[dict[str, Any]],
	distance_floor_tokens: int = 100,
) -> tuple[float, float]:
	if [item.get("token") for item in inf_results] != [item.get("token") for item in val_results]:
		return -1.0, -1.0

	if not inf_results:
		return 0.0, 0.0

	total_dist = 0.0
	total_n_matches = 0
	top_k = len((inf_results[0].get("logprobs") or {}).keys()) or 1

	for inf_position, val_position in zip(inf_results, val_results):
		dist, n_matches = token_distance2(inf_position, val_position)
		total_dist += dist
		total_n_matches += n_matches

	matches_ratio = total_n_matches / (len(inf_results) * top_k)
	norm_tokens = max(distance_floor_tokens, len(inf_results)) if distance_floor_tokens > 0 else len(inf_results)
	total_dist = (total_dist + 1.0) / (norm_tokens * top_k + 1.0)
	return total_dist, matches_ratio


def distance2(
	inference_result: dict[str, Any],
	validation_result: dict[str, Any],
	distance_floor_tokens: int = 100,
) -> tuple[float, float]:
	inf_results = inference_result.get("results") or []
	val_results = validation_result.get("results") or []
	return distance2_from_results(
		inf_results,
		val_results,
		distance_floor_tokens=distance_floor_tokens,
	)
