# PoC artifact-divergence experiments

Chronological log of cross-GPU / cross-AOT PoC tests. Folders are numbered `NNN_...` for sort order. Each experiment has its own `README.md` describing branch, commit, parameters, finding, and how to reproduce.

## Index

| # | folder | branch | endpoint | model | what was tested |
|---|---|---|---|---|---|
| 001 | [001_v0.17.0_generate_path_false_start](001_v0.17.0_generate_path_false_start/) | qdanik refactor/0.17.0 | `/api/v1/pow/generate` | Qwen2.5-7B | first try — turned out to be the wrong endpoint (scheduler path, not our patched runner). Identical AOT=1/AOT=0 was a tautology. |
| 002 | [002_v0.17.0_legacy_poc_qwen25_rtx](002_v0.17.0_legacy_poc_qwen25_rtx/) | qdanik refactor/0.17.0 | `/api/v1/pow/legacy_poc` | Qwen2.5-7B | RTX, correct endpoint. AOT=1 works (500 nonces); AOT=0 crashes with `assert input_ids is not None` in `qwen2.py:583`. Workaround mandatory for Qwen2.x. |
| 003 | [003_v0.17.0_legacy_poc_qwen3_rtx](003_v0.17.0_legacy_poc_qwen3_rtx/) | qdanik refactor/0.17.0 | `/api/v1/pow/legacy_poc` | Qwen3-0.6B | RTX, Qwen3 has no hard assert → both AOT modes run. L2=0 bit-exact (eager path ignores `input_ids` when `inputs_embeds` is given). |
| 004 | [004_v0.17.0_cross_gpu_h100_vs_rtx](004_v0.17.0_cross_gpu_h100_vs_rtx/) | qdanik refactor/0.17.0 | `/api/v1/pow/legacy_poc` | Qwen3-0.6B, Qwen2.5-7B | H100 vs RTX cross-GPU. AOT toggle still no effect on either GPU. Cross-GPU drift ~0.037 L2 (well below threshold 0.4), one Qwen2.5 outlier at 0.84. |
| 005 | [005_v0.15.1_kaitakuai_compiled_path](005_v0.15.1_kaitakuai_compiled_path/) | kaitakuai fix/poc-dummy-input-ids | `/api/v1/pow/init/generate` | Qwen2.5-7B | First test of the **compiled** PoC path (older 0.15.1 has no `skip_compiled=has_poc` routing). Workaround now truly critical: AOT=0 crashes in CUDA-graph capture. Cross-GPU drift jumps to ~0.22 mean, 5% over threshold — torch.compile adds noise. |

## Shared assets

- [`scripts/`](scripts/) — collectors, plot helper, patches. The canonical versions; per-experiment scripts (custom plots) live inside the experiment folder.
- See also memory notes in `~/.claude/projects/.../memory/` (referenced via `MEMORY.md`): the architectural map (`project_poc_aot_landscape`), reproducible recipe (`project_poc_test_methodology`), GPU server inventory (`reference_gpu_test_servers`).

## Folder naming convention

`NNN_<vllm-version>_<short-description>` — `NNN` is a sequential 3-digit number, never reused even if an experiment is abandoned. Future experiments append `006_…`, `007_…`, etc.

## When adding a new experiment

1. Create `poc_experiments/NNN_<descriptor>/` with a `README.md` (use one of the existing as template).
2. Drop raw JSONs + any custom plot scripts + the resulting PNG/CSV into it.
3. Update the table above with one line.
4. If a shared collector/patch script changed, update `scripts/` and note the new version (or copy the used version into the experiment folder if reproducibility matters).
5. Run any new comparison against an existing experiment by loading both JSONs through `scripts/plot_aot.py` (per-nonce L2) — its arg names are `--aot1` / `--aot0` but it doesn't care which is which.
