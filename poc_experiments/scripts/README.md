# Shared scripts and patches

Canonical versions used across experiments. If a script changed for a specific experiment, the modified copy lives inside that experiment folder.

## Collectors

| script | endpoint targeted | output | notes |
|---|---|---|---|
| `collect_aot.py` | `POST /api/v1/pow/generate` (one-shot, wait=true) | single JSON `{artifacts: [...]}` | scheduler path, does NOT exercise `legacy_poc_runner` patch — kept for the 001 reproduction only |
| `collect_legacy.py` | `POST /api/v1/pow/legacy_poc` callback + `POST /stop` | single JSON `{artifacts: [...]}` | what experiments 002–004 used |
| `collect_artifacts.py` | `POST /api/v1/pow/init/generate` callback + `POST /stop` | `<dir>/nonces_1000.json` + `config.json` | kaitakuai's script; expects URL prefix sed-fix `/inference/pow → /pow` on local containers. Flags: `--block-hash` / `--public-key` (seed `generate_inputs`, must match between runs you want to L2-compare); `--batch-size` (forwarded to engine — set `32` to match 0.17.0 default). |

## Plot helper

- `plot_aot.py` — generic per-nonce L2 + histogram. Args `--aot1` / `--aot0` are just two JSONs; metric is symmetric.

## Patches

- `aot_workaround.patch` — applies the `POC_USE_AOT_COMPILED_WORKAROUND` env var to `qdanik refactor/0.17.0` (modifies `vllm/poc/env.py` and `vllm/poc/server/legacy_poc_runner.py`). Already merged upstream into the branch (commit `83b0db8 AOT`) — kept for reference / older checkouts.
- `kaitakuai_aot_env.patch` — minimal env-var-only patch on top of `kaitakuai fix/poc-dummy-input-ids`. Used by experiment 005. Kept for historical reproducibility — current branches (`qdanik/compiled-0.15.1`) already bake these changes in, so newer experiments just clone the branch.
