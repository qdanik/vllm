#!/usr/bin/env python3
"""End-to-end verification harness for vLLM image validation.

This script starts a local OpenAI-compatible vLLM server, runs a PoC
artifact generation round, then validates inference quality against the
reference datasets stored in ``vllm/poc/verification/configs``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from typing import Any, Iterable

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_TIMEOUT_S = 1800
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_TP_SIZE = 1
DEFAULT_PP_SIZE = 1
DEFAULT_RESULTS_DIR = pathlib.Path(__file__).resolve().parent / "results"
DEFAULT_CONFIGS_DIR = pathlib.Path(__file__).resolve().parent / "configs"
DEFAULT_INFERENCE_CATEGORY = "inference_fp8"
DEFAULT_LIMIT_PROMPTS = 10
DEFAULT_POC_DURATION_S = 120
DEFAULT_POC_SEQ_LEN = 1024
DEFAULT_POC_K_DIM = 12
DEFAULT_POC_NONCES = "1,3,5,7,9"
DEFAULT_LOGPROBS_MODE = "processed_logprobs"
DEFAULT_POST_READY_DELAY_S = 2.0
DEFAULT_INFERENCE_REQUEST_DELAY_S = 0.0
DEFAULT_BLOCK_HASH = "8d148df1530d06a3412acd3deda4db16bae780eefdd160e081e6f878417de92a"
DEFAULT_PUBLIC_KEY = (
	"02e0f3b6b7f832ead7af2a235b9b27715a4d586b0fa108e735f0676a5086479225"
)
DEFAULT_BLOCK_HEIGHT = 2732723
_API_SERVER_FLAG_CACHE: set[str] | None = None

_CONSENSUS_ARG_FIELDS = {
	"model",
	"dtype",
	"tensor_parallel_size",
	"pipeline_parallel_size",
	"trust_remote_code",
	"performance_mode",
	"kv_cache_dtype",
	"logprobs_mode",
	"max_model_len",
	"attention_backend",
	"optimization_level",
	"disable_custom_all_reduce",
	"enforce_eager",
	"enable_cuda_compatibility",
	"cuda_compatibility_path",
	"additional_server_arg",
}


def _timestamp() -> str:
	return dt.datetime.now(dt.timezone.utc).isoformat()


def _safe_name(value: str) -> str:
	safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
	return safe.strip("._") or "unnamed"


def _preview_text(value: Any, limit: int = 240) -> str:
	text = str(value or "")
	text = re.sub(r"\s+", " ", text).strip()
	if len(text) <= limit:
		return text
	return text[: limit - 3] + "..."


def _parse_nonces(raw: str) -> list[int]:
	return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _json_dump(path: pathlib.Path, payload: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def _jsonl_dump(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		for row in rows:
			handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _read_json(path: pathlib.Path) -> dict[str, Any]:
	return json.loads(path.read_text())


def _iter_jsonl(path: pathlib.Path) -> Iterable[dict[str, Any]]:
	with path.open("r", encoding="utf-8") as handle:
		for line in handle:
			line = line.strip()
			if line:
				yield json.loads(line)


def _build_model_info_payload(
	name: str,
	url: str,
	deploy_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
	return {
		"name": name,
		"url": url,
		"deploy_params": dict(deploy_params or {}),
	}


def _post_json(
	url: str,
	payload: dict[str, Any],
	headers: dict[str, str],
	timeout_s: int = 60,
) -> dict[str, Any]:
	body = json.dumps(payload).encode("utf-8")
	req = urllib.request.Request(url, data=body, headers=headers, method="POST")
	with urllib.request.urlopen(req, timeout=timeout_s) as resp:
		return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, headers: dict[str, str], timeout_s: int = 30) -> dict[str, Any]:
	req = urllib.request.Request(url, headers=headers, method="GET")
	with urllib.request.urlopen(req, timeout=timeout_s) as resp:
		return json.loads(resp.read().decode("utf-8"))


def _get_status(
	url: str,
	headers: dict[str, str],
	timeout_s: int = 10,
) -> tuple[int, str]:
	req = urllib.request.Request(url, headers=headers, method="GET")
	try:
		with urllib.request.urlopen(req, timeout=timeout_s) as resp:
			return resp.status, resp.read().decode("utf-8")
	except urllib.error.HTTPError as exc:
		body = exc.read().decode("utf-8") if exc.fp else ""
		return exc.code, body


def _client_base_url(host: str, port: int) -> str:
	client_host = host
	if host in {"0.0.0.0", "::", "[::]"}:
		client_host = "127.0.0.1"
	return f"http://{client_host}:{port}"


def _wait_for_health(
	base_url: str,
	headers: dict[str, str],
	timeout_s: int,
	server: subprocess.Popen[str],
	model: str,
	server_tail: deque[str],
) -> dict[str, Any]:
	deadline = time.time() + timeout_s
	last_error = "unknown"
	last_report_at = 0.0
	while time.time() < deadline:
		if server.poll() is not None:
			tail = "\n".join(server_tail)
			raise RuntimeError(
				"API server exited before becoming healthy "
				f"(exit_code={server.returncode})\n"
				f"Last server log lines:\n{tail}"
			)
		now = time.time()
		if now - last_report_at >= 10.0:
			elapsed = int(now - (deadline - timeout_s))
			remaining = max(0, int(deadline - now))
			print(
				"Waiting for server readiness: "
				f"elapsed={elapsed}s remaining={remaining}s last_error={last_error}",
				flush=True,
			)
			last_report_at = now
		try:
			status_code, _ = _get_status(f"{base_url}/health", headers, timeout_s=5)
			if status_code != 200:
				last_error = f"HTTP {status_code}"
				time.sleep(0.5)
				continue

			models_response = _get_json(f"{base_url}/v1/models", headers, timeout_s=5)
			served_model_ids = [
				item.get("id")
				for item in models_response.get("data", [])
				if isinstance(item, dict) and item.get("id")
			]
			warmup_model = str(served_model_ids[0]) if served_model_ids else model
			return _warmup_chat(base_url, headers, warmup_model)
		except Exception as exc:  # noqa: BLE001
			last_error = str(exc)
		time.sleep(0.5)
	tail = "\n".join(server_tail)
	raise RuntimeError(
		"Server did not become ready for inference within timeout "
		f"({timeout_s}s): {last_error}\n"
		f"Last server log lines:\n{tail}"
	)


def _get_api_server_supported_flags() -> set[str]:
	global _API_SERVER_FLAG_CACHE
	if _API_SERVER_FLAG_CACHE is not None:
		return _API_SERVER_FLAG_CACHE

	try:
		proc = subprocess.run(
			[sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--help"],
			capture_output=True,
			text=True,
			timeout=30,
			check=False,
		)
		help_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
		_API_SERVER_FLAG_CACHE = set(re.findall(r"--[a-z0-9][a-z0-9-]*", help_text))
	except Exception:  # noqa: BLE001
		_API_SERVER_FLAG_CACHE = set()

	return _API_SERVER_FLAG_CACHE


def _supports_api_server_flag(flag: str) -> bool:
	supported = _get_api_server_supported_flags()
	if not supported:
		return True
	return flag in supported


def _append_optional_flag(cmd: list[str], flag: str, value: str | None = None) -> None:
	if not _supports_api_server_flag(flag):
		print(f"Warning: api_server does not support {flag}; skipping.", flush=True)
		return
	cmd.append(flag)
	if value is not None:
		cmd.append(value)


def _start_log_pump(
	stream: Any,
	log_path: pathlib.Path,
	tail: deque[str],
) -> threading.Thread:
	log_path.parent.mkdir(parents=True, exist_ok=True)

	def _pump() -> None:
		with log_path.open("w", encoding="utf-8") as sink:
			for raw_line in iter(stream.readline, ""):
				line = raw_line.rstrip("\n")
				sink.write(raw_line)
				sink.flush()
				tail.append(line)
				print(raw_line, end="", flush=True)

	worker = threading.Thread(target=_pump, daemon=True)
	worker.start()
	return worker


def _start_server(
	args: argparse.Namespace,
	server_log_path: pathlib.Path,
) -> tuple[subprocess.Popen[str], deque[str], threading.Thread | None]:
	env = os.environ.copy()
	env.setdefault("PYTHONUNBUFFERED", "1")
	if args.enable_cuda_compatibility:
		env["VLLM_ENABLE_CUDA_COMPATIBILITY"] = "1"
	if args.cuda_compatibility_path:
		env["VLLM_CUDA_COMPATIBILITY_PATH"] = args.cuda_compatibility_path

	cmd = [
		sys.executable,
		"-m",
		"vllm.entrypoints.openai.api_server",
		"--model",
		args.model,
		"--host",
		args.host,
		"--port",
		str(args.port),
		"--tensor-parallel-size",
		str(args.tensor_parallel_size),
	]
	_append_optional_flag(cmd, "--log-error-stack")

	if args.pipeline_parallel_size > 1:
		_append_optional_flag(
			cmd, "--pipeline-parallel-size", str(args.pipeline_parallel_size)
		)
	if args.trust_remote_code:
		cmd.append("--trust-remote-code")
	if args.dtype:
		_append_optional_flag(cmd, "--dtype", args.dtype)
	if args.performance_mode:
		_append_optional_flag(cmd, "--performance-mode", args.performance_mode)
	if args.kv_cache_dtype:
		_append_optional_flag(cmd, "--kv-cache-dtype", args.kv_cache_dtype)
	if args.optimization_level is not None:
		_append_optional_flag(cmd, "--optimization-level", str(args.optimization_level))
	if args.attention_backend:
		_append_optional_flag(cmd, "--attention-backend", args.attention_backend)
	if args.logprobs_mode:
		_append_optional_flag(cmd, "--logprobs-mode", args.logprobs_mode)
	if args.max_model_len is not None:
		_append_optional_flag(cmd, "--max-model-len", str(args.max_model_len))
	if args.disable_custom_all_reduce:
		_append_optional_flag(cmd, "--disable-custom-all-reduce")
	if args.enforce_eager:
		_append_optional_flag(cmd, "--enforce-eager")
	for extra_arg in args.additional_server_arg:
		cmd.append(extra_arg)

	process = subprocess.Popen(
		cmd,
		env=env,
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		text=True,
		bufsize=1,
	)
	tail: deque[str] = deque(maxlen=400)
	pump = None
	if process.stdout is not None:
		pump = _start_log_pump(process.stdout, server_log_path, tail)
	return process, tail, pump


def _stop_server(
	server: subprocess.Popen[str] | None,
	pump: threading.Thread | None,
) -> None:
	if server is None:
		return
	if server.poll() is None:
		server.send_signal(signal.SIGINT)
		try:
			server.wait(timeout=15)
		except subprocess.TimeoutExpired:
			server.kill()
			server.wait(timeout=5)
	if pump is not None:
		pump.join(timeout=2)


def _with_server_overrides(
	args: argparse.Namespace,
	overrides: dict[str, Any],
	*,
	override_existing: bool,
) -> argparse.Namespace:
	effective = argparse.Namespace(**vars(args))
	for key, value in overrides.items():
		if not hasattr(effective, key):
			continue
		if not override_existing and not _arg_has_default_value(effective, key):
			continue
		setattr(effective, key, value)
	return effective


def _start_ready_server(
	args: argparse.Namespace,
	server_log_path: pathlib.Path,
	headers: dict[str, str],
) -> tuple[
	subprocess.Popen[str],
	deque[str],
	threading.Thread | None,
	str,
	dict[str, Any],
	dict[str, Any],
]:
	base_url = _client_base_url(args.host, args.port)
	server, server_tail, pump = _start_server(args, server_log_path)
	try:
		warmup_response = _wait_for_health(
			base_url,
			headers,
			args.timeout_s,
			server,
			args.model,
			server_tail,
		)
		runtime_probe = _build_runtime_probe(base_url, headers)
		return server, server_tail, pump, base_url, warmup_response, runtime_probe
	except Exception:
		_stop_server(server, pump)
		raise


def _collect_gpu_info() -> dict[str, Any]:
	if not shutil_which("nvidia-smi"):
		return {
			"available": False,
			"gpus": [],
			"summary": "unknown",
			"vram_summary": "unknown",
		}

	query = [
		"nvidia-smi",
		"--query-gpu=name,memory.total,driver_version",
		"--format=csv,noheader,nounits",
	]
	proc = subprocess.run(query, capture_output=True, text=True, check=False)
	if proc.returncode != 0:
		return {
			"available": False,
			"gpus": [],
			"summary": "unknown",
			"vram_summary": "unknown",
			"error": proc.stderr.strip(),
		}

	gpus: list[dict[str, Any]] = []
	for line in proc.stdout.splitlines():
		parts = [part.strip() for part in line.split(",")]
		if len(parts) < 3:
			continue
		name, memory_total_mb, driver_version = parts[0], parts[1], parts[2]
		total_gib = round(float(memory_total_mb) / 1024.0, 1)
		gpus.append(
			{
				"name": name,
				"memory_total_mb": int(float(memory_total_mb)),
				"memory_total_gib": total_gib,
				"driver_version": driver_version,
			}
		)

	if not gpus:
		return {
			"available": False,
			"gpus": [],
			"summary": "unknown",
			"vram_summary": "unknown",
		}

	grouped: dict[tuple[str, float], int] = {}
	for gpu in gpus:
		key = (str(gpu["name"]), float(gpu["memory_total_gib"]))
		grouped[key] = grouped.get(key, 0) + 1

	summary_parts = []
	vram_parts = []
	for (name, memory_gib), count in grouped.items():
		summary_parts.append(f"{count}x{name}" if count > 1 else name)
		vram_parts.append(
			f"{count}x{memory_gib:.1f}GiB" if count > 1 else f"{memory_gib:.1f}GiB"
		)

	return {
		"available": True,
		"gpus": gpus,
		"summary": ", ".join(summary_parts),
		"vram_summary": ", ".join(vram_parts),
	}


def shutil_which(cmd: str) -> str | None:
	return subprocess.run(
		["/usr/bin/env", "sh", "-c", f"command -v {cmd}"],
		capture_output=True,
		text=True,
		check=False,
	).stdout.strip() or None


def _build_runtime_probe(base_url: str, headers: dict[str, str]) -> dict[str, Any]:
	models_url = f"{base_url}/v1/models"
	models_response = _get_json(models_url, headers, timeout_s=30)
	health_status_code, health_body = _get_status(f"{base_url}/health", headers)
	version_status_code, version_body = _get_status(f"{base_url}/version", headers)
	served_model_ids = [
		item.get("id")
		for item in models_response.get("data", [])
		if isinstance(item, dict) and item.get("id")
	]
	max_model_len = None
	if models_response.get("data"):
		first = models_response["data"][0]
		if isinstance(first, dict):
			max_model_len = first.get("max_model_len")

	return {
		"base_url": base_url,
		"models_url": models_url,
		"served_model_ids": served_model_ids,
		"raw_models_response": models_response,
		"health_status_code": health_status_code,
		"health_body": health_body,
		"version_status_code": version_status_code,
		"version_body": version_body,
		"max_model_len": max_model_len,
		"timestamp": _timestamp(),
	}


def _print_capability_line(
	args: argparse.Namespace,
	runtime_probe: dict[str, Any],
	gpu_info: dict[str, Any],
) -> None:
	line = (
		f"GPU={gpu_info.get('summary', 'unknown')} | "
		f"Model={args.model} | "
		f"DType={args.dtype or 'auto'} | "
		f"KV Cache={args.kv_cache_dtype or 'auto'} | "
		f"TP={args.tensor_parallel_size} | "
		f"PP={args.pipeline_parallel_size} | "
		f"Max Seq Len={runtime_probe.get('max_model_len', 'unknown')} | "
		f"VRAM={gpu_info.get('vram_summary', 'unknown')} | "
		f"PoC Seq Len={args.poc_seq_len} | "
		f"Attn Backend={args.attention_backend or 'auto'} | "
		f"Tools={args.tools_label or 'auto'}"
	)
	print(line, flush=True)


def _warmup_chat(base_url: str, headers: dict[str, str], model: str) -> dict[str, Any]:
	payload = {
		"model": model,
		"messages": [{"role": "user", "content": "Ping"}],
		"max_tokens": 8,
		"temperature": 0.0,
	}
	return _post_json(f"{base_url}/v1/chat/completions", payload, headers, timeout_s=30)


def _run_poc_phase(
	args: argparse.Namespace,
	base_url: str,
	headers: dict[str, str],
	results_root: pathlib.Path,
	runtime_probe: dict[str, Any],
	gpu_info: dict[str, Any],
) -> dict[str, Any]:
	model_timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
	model_dir = results_root / f"{_safe_name(args.model)}_{model_timestamp}"
	init_payload = {
		"block_hash": args.poc_block_hash,
		"block_height": args.poc_block_height,
		"public_key": args.poc_public_key,
		"node_id": args.poc_node_id,
		"node_count": args.poc_node_count,
		"group_id": args.poc_group_id,
		"n_groups": args.poc_n_groups,
		"batch_size": args.poc_batch_size,
		"params": {
			"model": args.model,
			"seq_len": args.poc_seq_len,
			"k_dim": args.poc_k_dim,
		},
		"url": None,
	}
	poc_config_payload = {
		"timestamp": _timestamp(),
		"model": args.model,
		"model_output_dir": str(model_dir),
		"base_url": base_url,
		"duration_seconds": args.poc_duration_seconds,
		"init_payload": init_payload,
		"runtime_probe": runtime_probe,
		"gpu_info": gpu_info,
		"server": {
			"dtype": args.dtype or "auto",
			"kv_cache_dtype": args.kv_cache_dtype or "auto",
			"tensor_parallel_size": args.tensor_parallel_size,
			"pipeline_parallel_size": args.pipeline_parallel_size,
			"attention_backend": args.attention_backend or "auto",
			"enforce_eager": args.enforce_eager,
			"additional_server_args": args.additional_server_arg,
		},
	}
	_json_dump(model_dir / "config.json", poc_config_payload)

	print("Starting PoC generation...", flush=True)
	init_response = _post_json(
		f"{base_url}/api/v1/pow/init/generate",
		init_payload,
		headers,
		timeout_s=30,
	)

	status_samples: list[dict[str, Any]] = []
	start_time = time.time()
	while time.time() - start_time < args.poc_duration_seconds:
		sample = _get_json(f"{base_url}/api/v1/pow/status", headers, timeout_s=15)
		sample["timestamp"] = _timestamp()
		status_samples.append(sample)
		time.sleep(args.poc_status_poll_interval)

	stop_response = _post_json(
		f"{base_url}/api/v1/pow/stop",
		{},
		headers,
		timeout_s=30,
	)

	generate_payload = {
		"block_hash": args.poc_block_hash,
		"block_height": args.poc_block_height,
		"public_key": args.poc_public_key,
		"node_id": args.poc_node_id,
		"node_count": args.poc_node_count,
		"nonces": _parse_nonces(args.poc_nonces),
		"params": {
			"model": args.model,
			"seq_len": args.poc_seq_len,
			"k_dim": args.poc_k_dim,
		},
		"wait": True,
		"url": None,
		"validation": None,
		"stat_test": None,
	}
	generate_response = _post_json(
		f"{base_url}/api/v1/pow/generate",
		generate_payload,
		headers,
		timeout_s=args.poc_generate_timeout_seconds,
	)

	validation_response = None
	if isinstance(generate_response, dict) and generate_response.get("artifacts"):
		validate_payload = dict(generate_payload)
		validate_payload["validation"] = {
			"artifacts": generate_response["artifacts"],
		}
		validation_response = _post_json(
			f"{base_url}/api/v1/pow/generate",
			validate_payload,
			headers,
			timeout_s=args.poc_generate_timeout_seconds,
		)

	artifacts_payload = {
		"timestamp": _timestamp(),
		"model_output_dir": str(model_dir),
		"init_response": init_response,
		"status_samples": status_samples,
		"stop_response": stop_response,
		"generate_payload": generate_payload,
		"generate_response": generate_response,
		"validation_response": validation_response,
		"summary": {
			"status_samples": len(status_samples),
			"artifacts_count": len(generate_response.get("artifacts", []))
			if isinstance(generate_response, dict)
			else 0,
			"max_nonces_per_second": max(
				(
					float(sample.get("stats", {}).get("nonces_per_second", 0.0))
					for sample in status_samples
				),
				default=0.0,
			),
			"total_processed_last": (
				status_samples[-1].get("stats", {}).get("total_processed", 0)
				if status_samples
				else 0
			),
		},
	}
	_json_dump(model_dir / "artifacts.json", artifacts_payload)
	return artifacts_payload


def _normalize_message_content(content: Any) -> str:
	if isinstance(content, str):
		return content
	if isinstance(content, list):
		chunks: list[str] = []
		for item in content:
			if isinstance(item, dict):
				text = item.get("text")
				if isinstance(text, str):
					chunks.append(text)
		return "".join(chunks)
	return ""


def _prepare_messages(prompt: str) -> list[dict[str, str]]:
	return [
		{
			"role": "system",
			"content": "You are a helpful assistant. Response clear, correct and complete.",
		},
		{"role": "user", "content": prompt},
	]


def _build_enforced_tokens(expected_record: dict[str, Any]) -> dict[str, Any] | None:
	results = expected_record.get("inference_result", {}).get("results")
	if not isinstance(results, list) or not results:
		return None

	tokens: list[dict[str, Any]] = []
	for item in results:
		if not isinstance(item, dict):
			continue
		token = item.get("token")
		logprobs = item.get("logprobs")
		if token is None or not isinstance(logprobs, dict) or not logprobs:
			continue
		tokens.append(
			{
				"token": str(token),
				"top_tokens": [str(candidate) for candidate in logprobs.keys()],
			}
		)

	if not tokens:
		return None
	return {"tokens": tokens}


def _normalize_logprobs(choice: dict[str, Any]) -> list[dict[str, Any]]:
	content = choice.get("logprobs", {}).get("content")
	if not isinstance(content, list):
		return []

	normalized: list[dict[str, Any]] = []
	for item in content:
		if not isinstance(item, dict):
			continue
		token = str(item.get("token", ""))
		token_logprob = item.get("logprob")
		top_logprobs = item.get("top_logprobs")
		logprobs: dict[str, Any] = {}
		if token and token_logprob is not None:
			logprobs[token] = token_logprob
		if isinstance(top_logprobs, list):
			for candidate in top_logprobs:
				if not isinstance(candidate, dict):
					continue
				candidate_token = candidate.get("token")
				candidate_logprob = candidate.get("logprob")
				if candidate_token is not None and candidate_logprob is not None:
					logprobs[str(candidate_token)] = candidate_logprob
		normalized.append({"token": token, "logprobs": logprobs})
	return normalized


def _build_failed_verification_record(
	served_model: str,
	base_url: str,
	expected_record: dict[str, Any],
	request_params: dict[str, Any],
	*,
	error_type: str,
	error_message: str,
	error_code: Any = None,
	raw_response: dict[str, Any] | None = None,
) -> dict[str, Any]:
	response_id = raw_response.get("id") if isinstance(raw_response, dict) else None
	return {
		"prompt": expected_record.get("prompt"),
		"language": expected_record.get("language"),
		"inference_result": {
			"text": "",
			"results": [],
		},
		"inference_model": {
			"name": served_model,
			"url": base_url,
			"deploy_params": {},
		},
		"request_params": request_params,
		"metadata": {
			"finish_reason": "error",
			"usage": {},
			"response_id": response_id,
			"request_failed": True,
			"request_error_type": error_type,
			"request_error_code": error_code,
			"request_error_message": error_message,
		},
		"raw_response": raw_response or {},
	}


def _request_with_retries(
	url: str,
	payload: dict[str, Any],
	headers: dict[str, str],
	timeout_seconds: int,
	max_attempts: int,
	retry_backoff_start: float,
	retry_backoff_multiplier: float,
) -> dict[str, Any]:
	delay = retry_backoff_start
	last_error: Exception | None = None
	for attempt in range(1, max_attempts + 1):
		try:
			return _post_json(url, payload, headers, timeout_s=timeout_seconds)
		except Exception as exc:  # noqa: BLE001
			last_error = exc
			if attempt == max_attempts:
				break
			time.sleep(delay)
			delay *= retry_backoff_multiplier
	if last_error is None:
		raise RuntimeError("request failed without captured exception")
	raise last_error


def _run_single_verification_request(
	base_url: str,
	headers: dict[str, str],
	served_model: str,
	expected_record: dict[str, Any],
	request_params: dict[str, Any],
) -> dict[str, Any]:
	payload: dict[str, Any] = {
		"model": served_model,
		"messages": _prepare_messages(str(expected_record.get("prompt", ""))),
		"max_tokens": request_params.get("max_tokens"),
		"temperature": request_params.get("temperature"),
		"seed": request_params.get("seed"),
		"stream": False,
		"n": 1,
		"skip_special_tokens": False,
	}
	additional_params = request_params.get("additional_params") or {}
	payload.update(additional_params)
	enforced_tokens = _build_enforced_tokens(expected_record)
	if enforced_tokens is not None:
		payload["enforced_tokens"] = enforced_tokens

	top_logprobs = request_params.get("top_logprobs")
	if top_logprobs:
		payload["logprobs"] = True
		payload["top_logprobs"] = top_logprobs
	if request_params.get("top_p") is not None:
		payload["top_p"] = request_params["top_p"]
	if request_params.get("top_k") is not None:
		payload["top_k"] = request_params["top_k"]
	if request_params.get("repetition_penalty") is not None:
		payload["repetition_penalty"] = request_params["repetition_penalty"]

	try:
		response = _request_with_retries(
			f"{base_url}/v1/chat/completions",
			payload,
			headers,
			timeout_seconds=int(request_params.get("timeout_seconds", 300)),
			max_attempts=int(request_params.get("retries_max_attempts", 3)),
			retry_backoff_start=float(request_params.get("retry_backoff_seconds_start", 1.0)),
			retry_backoff_multiplier=float(request_params.get("retry_backoff_multiplier", 2.0)),
		)
	except Exception as exc:  # noqa: BLE001
		return _build_failed_verification_record(
			served_model,
			base_url,
			expected_record,
			request_params,
			error_type=type(exc).__name__,
			error_message=str(exc),
		)

	error = response.get("error")
	if isinstance(error, dict):
		error_message = str(error.get("message") or "")
		error_type = str(error.get("type") or "APIError")
		error_code = error.get("code")
		return _build_failed_verification_record(
			served_model,
			base_url,
			expected_record,
			request_params,
			error_type=error_type,
			error_message=error_message,
			error_code=error_code,
			raw_response=response,
		)

	choices = response.get("choices") or []
	first_choice = choices[0] if choices else {}
	message = first_choice.get("message") if isinstance(first_choice, dict) else {}
	text = _normalize_message_content(message.get("content") if isinstance(message, dict) else "")
	usage = response.get("usage") or {}

	return {
		"prompt": expected_record.get("prompt"),
		"language": expected_record.get("language"),
		"inference_result": {
			"text": text,
			"results": _normalize_logprobs(first_choice) if isinstance(first_choice, dict) else [],
		},
		"inference_model": {
			"name": served_model,
			"url": base_url,
			"deploy_params": {},
		},
		"request_params": request_params,
		"metadata": {
			"finish_reason": first_choice.get("finish_reason") if isinstance(first_choice, dict) else None,
			"usage": usage,
			"response_id": response.get("id"),
		},
		"raw_response": response,
	}


def _sleep_with_message(delay_s: float, reason: str) -> None:
	if delay_s <= 0:
		return
	print(f"Waiting {delay_s:.2f}s before {reason}...", flush=True)
	time.sleep(delay_s)


def _results_exact_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
	expected_result = expected.get("inference_result") or {}
	actual_result = actual.get("inference_result") or {}
	return expected_result == actual_result


def _select_categories(
	configs_dir: pathlib.Path,
	requested_categories: list[str],
) -> list[pathlib.Path]:
	categories = sorted(
		path for path in configs_dir.iterdir() if path.is_dir() and (path / "inference_config.json").exists()
	)
	if requested_categories:
		selected: list[pathlib.Path] = []
		for name in requested_categories:
			match = configs_dir / name
			if not match.exists():
				raise FileNotFoundError(f"Verification category not found: {name}")
			selected.append(match)
		return selected

	return categories


def _nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
	current: Any = payload
	for key in path:
		if not isinstance(current, dict):
			return None
		current = current.get(key)
	return current


def _extract_legacy_max_model_len(config: dict[str, Any]) -> int | None:
	runtime_probe = config.get("vllm_runtime_probe")
	if not isinstance(runtime_probe, dict):
		return None

	max_model_len = runtime_probe.get("max_model_len")
	if isinstance(max_model_len, int):
		return max_model_len

	models_response = runtime_probe.get("raw_models_response")
	if not isinstance(models_response, dict):
		return None
	data = models_response.get("data")
	if not isinstance(data, list) or not data:
		return None
	first = data[0]
	if not isinstance(first, dict):
		return None
	value = first.get("max_model_len")
	return value if isinstance(value, int) else None


def _extract_consensus_server_params(config: dict[str, Any]) -> dict[str, Any]:
	model_name = _nested_get(config, ("model_info", "name"))
	deploy_params = _nested_get(config, ("model_info", "deploy_params"))
	deploy_params = deploy_params if isinstance(deploy_params, dict) else {}

	params: dict[str, Any] = {}
	if isinstance(model_name, str) and model_name.strip():
		params["model"] = model_name.strip()

	for key in _CONSENSUS_ARG_FIELDS - {"model"}:
		if key in deploy_params and deploy_params[key] is not None:
			params[key] = deploy_params[key]

	if "max_model_len" not in params:
		legacy_max_model_len = _extract_legacy_max_model_len(config)
		if legacy_max_model_len is not None:
			params["max_model_len"] = legacy_max_model_len

	return params


def _collect_consensus_server_params(
	configs_dir: pathlib.Path,
	requested_categories: list[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
	category_dirs = _select_categories(configs_dir, requested_categories)
	merged_params: dict[str, Any] = {}
	category_params: dict[str, dict[str, Any]] = {}
	conflicts: dict[str, list[dict[str, Any]]] = {}

	for category_dir in category_dirs:
		config = _read_json(category_dir / "inference_config.json")
		params = _extract_consensus_server_params(config)
		category_params[category_dir.name] = params
		for key, value in params.items():
			if key not in merged_params:
				merged_params[key] = value
				continue
			if merged_params[key] != value:
				conflicts.setdefault(key, []).append(
					{
						"category": category_dir.name,
						"value": value,
					}
				)

	for key, items in conflicts.items():
		items.insert(
			0,
			{
				"category": "baseline",
				"value": merged_params[key],
			},
		)
		merged_params.pop(key, None)

	return merged_params, category_params, conflicts


def _arg_has_default_value(args: argparse.Namespace, field_name: str) -> bool:
	return getattr(args, field_name) == build_parser().get_default(field_name)


def _apply_consensus_server_params(
	args: argparse.Namespace,
	consensus_params: dict[str, Any],
) -> dict[str, Any]:
	applied: dict[str, Any] = {}
	for key, value in consensus_params.items():
		if not hasattr(args, key):
			continue
		if not _arg_has_default_value(args, key):
			continue
		setattr(args, key, value)
		applied[key] = value
	return applied


def _run_inference_verification(
	args: argparse.Namespace,
	headers: dict[str, str],
	results_root: pathlib.Path,
	session_dir: pathlib.Path,
) -> dict[str, Any]:
	results: dict[str, Any] = {}
	categories = _select_categories(args.configs_dir, args.category)

	for category_dir in categories:
		category_name = category_dir.name
		print(f"Running verification category: {category_name}", flush=True)
		category_timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
		category_output_dir = results_root / f"{category_name}_{category_timestamp}"
		expected_config = _read_json(category_dir / "inference_config.json")
		category_server_params = _extract_consensus_server_params(expected_config)
		category_validation_params = dict(category_server_params)
		if not args.allow_category_model_override:
			category_validation_params.pop("model", None)
		category_args = _with_server_overrides(
			args,
			category_validation_params,
			override_existing=False,
		)
		category_server_log_path = session_dir / f"server_{category_name}.log"
		(
			server,
			server_tail,
			pump,
			base_url,
			_,
			runtime_probe,
		) = _start_ready_server(category_args, category_server_log_path, headers)
		expected_records = list(_iter_jsonl(category_dir / "inference_results.jsonl"))
		if args.limit_prompts is not None:
			expected_records = expected_records[: args.limit_prompts]

		try:
			served_model = category_args.model
			served_ids = runtime_probe.get("served_model_ids") or []
			if served_ids:
				served_model = str(served_ids[0])

			validation_model_info = _build_model_info_payload(
				served_model,
				base_url.rstrip("/") + "/",
				{
					key: value
					for key, value in category_server_params.items()
					if key != "model"
				},
			)

			request_params = dict(expected_config.get("request_params", {}))
			actual_records: list[dict[str, Any]] = []
			started_at = time.time()
			_sleep_with_message(
				args.inference_request_delay_seconds,
				f"starting verification category {category_name}",
			)
			for index, expected_record in enumerate(expected_records, start=1):
				try:
					actual_record = _run_single_verification_request(
						base_url,
						headers,
						served_model,
						expected_record,
						request_params,
					)
				except Exception as exc:  # noqa: BLE001
					category_output_dir.mkdir(parents=True, exist_ok=True)
					failure_payload = {
						"timestamp": _timestamp(),
						"category": category_name,
						"prompt_index": index,
						"prompt_preview": _preview_text(expected_record.get("prompt")),
						"served_model": served_model,
						"request_params": request_params,
						"server_running": server.poll() is None,
						"server_returncode": server.poll(),
						"error_type": type(exc).__name__,
						"error": str(exc),
						"server_tail": list(server_tail),
						"server_log": str(category_server_log_path),
						"server_params": category_server_params,
					}
					_json_dump(category_output_dir / "failure.json", failure_payload)
					raise RuntimeError(
						"Verification request failed "
						f"(category={category_name}, prompt_index={index}, "
						f"prompt={_preview_text(expected_record.get('prompt'))!r}, "
						f"server_running={server.poll() is None}, "
						f"server_returncode={server.poll()})"
					) from exc
				actual_record["metadata"] = actual_record.get("metadata") or {}
				actual_record["metadata"]["prompt_index"] = index
				actual_record["metadata"]["server_log"] = str(category_server_log_path)
				actual_record["metadata"]["server_params"] = category_server_params
				actual_records.append(actual_record)
				if index < len(expected_records):
					_sleep_with_message(
						args.inference_request_delay_seconds,
						f"sending prompt {index + 1} for category {category_name}",
					)

			duration = time.time() - started_at
			exact_match_count = sum(
				1
				for expected_record, actual_record in zip(expected_records, actual_records)
				if _results_exact_match(expected_record, actual_record)
			)
			mismatch_count = len(expected_records) - exact_match_count
			request_failure_count = sum(
				1
				for record in actual_records
				if bool((record.get("metadata") or {}).get("request_failed"))
			)
			total_output_tokens = sum(
				len(record.get("inference_result", {}).get("results", []))
				for record in actual_records
			)
			category_config_payload = {
				"timestamp": _timestamp(),
				"category": category_name,
				"category_output_dir": str(category_output_dir),
				"served_model": served_model,
				"base_url": base_url,
				"server_log": str(category_server_log_path),
				"server_params": category_server_params,
				"execution_config": {
					"source": str(category_dir / "inference_config.json"),
					"request_params": request_params,
					"metadata": {
						"exp_name": expected_config.get("exp_name"),
						"n_prompts": expected_config.get("n_prompts"),
						"languages_used": expected_config.get("languages_used"),
						"multilingual": expected_config.get("multilingual"),
					},
				},
				"runtime_probe": runtime_probe,
				"run": {
					"n_prompts": len(expected_records),
					"duration_seconds": round(duration, 3),
				},
			}
			analysis_validation_config_payload = {
				"timestamp": _timestamp(),
				"artifact_dir": str(category_output_dir),
				"source_inference_artifact": str((category_output_dir / "inference_results.jsonl").resolve()),
				"reference_inference_artifact": str((category_dir / "inference_results.jsonl").resolve()),
				"n_items": len(expected_records),
				"validation_model_info": validation_model_info,
				"request_params": request_params,
				"vllm_runtime_probe": runtime_probe,
				"config_check_passed": True,
				"config_diff_keys": [],
				"cli": {
					"validation_url": base_url,
					"validation_model": served_model,
					"max_workers": 1,
					"wait_timeout_s": args.timeout_s,
					"max_attempts": request_params.get("retries_max_attempts", 3),
					"retry_backoff_start_s": request_params.get("retry_backoff_seconds_start", 1.0),
					"retry_backoff_mult": request_params.get("retry_backoff_multiplier", 2.0),
					"artifact_tag": "",
				},
				"performance": {
					"total_time_seconds": round(duration, 3),
					"n_prompts": len(expected_records),
					"total_output_tokens": total_output_tokens,
					"output_tokens_per_second": round(total_output_tokens / duration, 2)
					if duration > 0
					else 0,
					"average_time_per_prompt_seconds": round(duration / len(expected_records), 3)
					if expected_records
					else 0,
				},
			}
			category_result_payload = {
				"timestamp": _timestamp(),
				"category": category_name,
				"category_output_dir": str(category_output_dir),
				"summary": {
					"n_prompts": len(expected_records),
					"exact_match_count": exact_match_count,
					"mismatch_count": mismatch_count,
					"request_failure_count": request_failure_count,
					"duration_seconds": round(duration, 3),
				},
			}

			_json_dump(category_output_dir / "inference_config.json", expected_config)
			_json_dump(category_output_dir / "validation_config.json", analysis_validation_config_payload)
			_jsonl_dump(category_output_dir / "inference_results.jsonl", actual_records)
			_json_dump(category_output_dir / "config.json", category_config_payload)
			_json_dump(category_output_dir / "inference_result.json", category_result_payload)
			results[category_name] = {
				"n_prompts": len(expected_records),
				"exact_match_count": exact_match_count,
				"mismatch_count": mismatch_count,
				"request_failure_count": request_failure_count,
				"exact_match_rate": round(exact_match_count / len(expected_records), 6)
				if expected_records
				else 0.0,
				"category_output_dir": str(category_output_dir),
				"inference_model": str(expected_config.get("model_info", {}).get("name", "")),
				"validation_model": served_model,
			}
		finally:
			_stop_server(server, pump)

	return results


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", default=DEFAULT_MODEL)
	parser.add_argument("--host", default=DEFAULT_HOST)
	parser.add_argument("--port", type=int, default=DEFAULT_PORT)
	parser.add_argument("--tensor-parallel-size", type=int, default=DEFAULT_TP_SIZE)
	parser.add_argument("--pipeline-parallel-size", type=int, default=DEFAULT_PP_SIZE)
	parser.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
	parser.add_argument("--trust-remote-code", action="store_true")
	parser.add_argument("--dtype", default="")
	parser.add_argument("--api-key", default="")
	parser.add_argument("--results-dir", type=pathlib.Path, default=DEFAULT_RESULTS_DIR)
	parser.add_argument("--configs-dir", type=pathlib.Path, default=DEFAULT_CONFIGS_DIR)
	parser.add_argument("--category", action="append", default=[])
	parser.add_argument("--limit-prompts", type=int, default=DEFAULT_LIMIT_PROMPTS)
	parser.add_argument("--skip-poc", action="store_true")
	parser.add_argument("--skip-inference", action="store_true")
	parser.add_argument(
		"--allow-category-model-override",
		action="store_true",
		help=(
			"Allow category inference_config to override validation model per category. "
			"By default validation model stays fixed from CLI --model."
		),
	)
	parser.add_argument("--tools-label", default="")
	parser.add_argument(
		"--performance-mode",
		choices=["balanced", "interactivity", "throughput"],
		default="",
	)
	parser.add_argument(
		"--logprobs-mode",
		choices=["processed_logits", "processed_logprobs", "raw_logits", "raw_logprobs"],
		default=DEFAULT_LOGPROBS_MODE,
	)
	parser.add_argument("--kv-cache-dtype", default="")
	parser.add_argument("--max-model-len", type=int, default=None)
	parser.add_argument("--attention-backend", default="")
	parser.add_argument("--optimization-level", type=int, default=None)
	parser.add_argument("--disable-custom-all-reduce", action="store_true")
	parser.add_argument("--enforce-eager", action="store_true")
	parser.add_argument("--enable-cuda-compatibility", action="store_true")
	parser.add_argument("--cuda-compatibility-path", default="")
	parser.add_argument(
		"--post-ready-delay-seconds",
		type=float,
		default=DEFAULT_POST_READY_DELAY_S,
	)
	parser.add_argument(
		"--inference-request-delay-seconds",
		type=float,
		default=DEFAULT_INFERENCE_REQUEST_DELAY_S,
	)
	parser.add_argument(
		"--additional-server-arg",
		action="append",
		default=[],
		help="Repeatable passthrough arg for api_server.",
	)
	parser.add_argument("--poc-duration-seconds", type=int, default=DEFAULT_POC_DURATION_S)
	parser.add_argument("--poc-seq-len", type=int, default=DEFAULT_POC_SEQ_LEN)
	parser.add_argument("--poc-k-dim", type=int, default=DEFAULT_POC_K_DIM)
	parser.add_argument("--poc-nonces", default=DEFAULT_POC_NONCES)
	parser.add_argument("--poc-node-id", type=int, default=0)
	parser.add_argument("--poc-node-count", type=int, default=1)
	parser.add_argument("--poc-group-id", type=int, default=0)
	parser.add_argument("--poc-n-groups", type=int, default=1)
	parser.add_argument("--poc-batch-size", type=int, default=None)
	parser.add_argument("--poc-status-poll-interval", type=float, default=2.0)
	parser.add_argument("--poc-generate-timeout-seconds", type=int, default=180)
	parser.add_argument("--poc-block-hash", default=DEFAULT_BLOCK_HASH)
	parser.add_argument("--poc-public-key", default=DEFAULT_PUBLIC_KEY)
	parser.add_argument("--poc-block-height", type=int, default=DEFAULT_BLOCK_HEIGHT)
	return parser


def main() -> int:
	args = build_parser().parse_args()
	args.results_dir.mkdir(parents=True, exist_ok=True)
	args.configs_dir = args.configs_dir.resolve()
	if not args.skip_inference and not args.category:
		args.category = [DEFAULT_INFERENCE_CATEGORY]
	headers = {"Content-Type": "application/json"}
	if args.api_key:
		headers["Authorization"] = f"Bearer {args.api_key}"

	session_dir = args.results_dir / f"session_{_safe_name(args.model)}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
	session_dir.mkdir(parents=True, exist_ok=True)
	poc_server: subprocess.Popen[str] | None = None
	poc_server_tail: deque[str] | None = None
	poc_pump: threading.Thread | None = None
	poc_server_log_path = session_dir / "server_poc.log"
	poc_base_url: str | None = None
	poc_runtime_probe: dict[str, Any] | None = None
	startup_model = args.model

	try:
		gpu_info = _collect_gpu_info()
		if not args.skip_poc:
			(
				poc_server,
				poc_server_tail,
				poc_pump,
				poc_base_url,
				_,
				poc_runtime_probe,
			) = _start_ready_server(args, poc_server_log_path, headers)
			_print_capability_line(args, poc_runtime_probe, gpu_info)
			_sleep_with_message(args.post_ready_delay_seconds, "starting workload")

		summary: dict[str, Any] = {
			"timestamp": _timestamp(),
			"model": startup_model,
			"gpu_info": gpu_info,
			"session_dir": str(session_dir),
			"phases": {},
		}

		if not args.skip_poc:
			summary["phases"]["poc"] = _run_poc_phase(
				args,
				poc_base_url,
				headers,
				args.results_dir,
				poc_runtime_probe,
				gpu_info,
			).get("summary", {})
			summary["poc_server_log"] = str(poc_server_log_path)
			summary["poc_runtime_probe"] = poc_runtime_probe
			summary["poc_base_url"] = poc_base_url
			_stop_server(poc_server, poc_pump)
			poc_server = None
			poc_pump = None
			poc_server_tail = None

		if not args.skip_inference:
			summary["phases"]["inference"] = _run_inference_verification(
				args,
				headers,
				args.results_dir,
				session_dir,
			)

		_json_dump(session_dir / "summary.json", summary)
		print(json.dumps(summary, indent=2), flush=True)
		return 0
	except urllib.error.HTTPError as exc:
		body = exc.read().decode("utf-8") if exc.fp else ""
		print(f"HTTP error: {exc.code} {exc.reason}\n{body}", file=sys.stderr)
		return 2
	finally:
		_stop_server(poc_server, poc_pump)


if __name__ == "__main__":
	raise SystemExit(main())
