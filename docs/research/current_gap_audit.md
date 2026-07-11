# Current gap audit

**Date:** 2026-07-11
**Commit audited:** `ce46966` (3 commits, `main`, nothing pushed)
**Baseline verified before writing this:** `pytest` 264 passed · `ruff check` clean ·
`ruff format --check` clean · `mypy` clean (51 source files).

## Status vocabulary

| Status | Meaning |
|---|---|
| `implemented_and_live_verified` | Ran end-to-end against **real data and/or a real model**, output inspected |
| `implemented_unit_tested_only` | Code exists and is tested, but never exercised against the real external thing |
| `partial` | Some of the promised behaviour exists; named gaps remain |
| `stub` | Shape/schema exists, logic does not |
| `missing` | Not written at all |
| `blocked` | Cannot be done here; the blocker is named |

## Headline finding

The repository today **validates infrastructure, not financial capability**. It proves a pipeline
can execute and that FinQA converts correctly. It has **never invoked a real language model**. The
only provider is `mock`, and the mock *reads the gold answer out of the request*. Consequently:

> **No claim about any model's financial ability can currently be supported by this repository.**

Two defects rise above the rest, and are fixed first (see `docs/research/validity_threats.md`):

1. **Gold-answer leakage into the request object.** `ModelRequest.simulation_context` carries
   `gold_answer` and `gold_numeric_value`. That object is serialized into `predictions.jsonl` **and
   hashed into the response-cache key**. Today only a `provider == "mock"` branch keeps it empty for
   real providers — a convention, not a guarantee.
2. **Mock runs are indistinguishable from model runs.** There is no `run_type`, no
   `eligible_for_leaderboard`, no `--allow-mock` gate. A mock run writes the same artifacts and
   lands on the same leaderboard as a real one would.

## Component audit

### Core plumbing

| Component | Status | Evidence / gap |
|---|---|---|
| Canonical schemas (`schemas/`) | `implemented_and_live_verified` | Real FinQA data (6,251/883/1,147) converts into `CanonicalSample` with zero errors |
| Registries (dataset / provider / metric) | `implemented_unit_tested_only` | Work, but only ever resolved `mock` + `smoke` + `finqa` |
| Response cache (`execution/cache.py`) | `implemented_unit_tested_only` | Hash stability + all 4 cache modes tested; never exercised against a real provider |
| Retry / backoff (`execution/retry.py`) | `implemented_unit_tested_only` | Only ever retried the mock's synthetic `ProviderTimeoutError` |
| Run engine (`execution/engine.py`) | `partial` | Single request per sample. **No conversation turn-chaining** (ConvFinQA needs it); no tool loop; no retrieval step |
| Run artifacts (18 files) | `partial` | All 18 files are written, but `gates.json` and `confidence_intervals.json` are honestly-empty placeholders (`evaluated: false`) |
| CLI (`cli.py`) | `partial` | 14 commands work. Missing: `--allow-mock`, `--all-core`, `list-models`, eval-mode flags |

### Anti-fabrication protections (mission Phase 2)

| Protection | Status | Gap |
|---|---|---|
| Gold-answer leakage prevention | **`missing`** | The request object *contains* the gold answer. No test asserts otherwise |
| `--allow-mock` gate | **`missing`** | Mock runs freely, by default |
| `run_type` / `eligible_for_leaderboard` | **`missing`** | Fields do not exist |
| Mock watermark in reports | **`missing`** | — |
| Mock excluded from leaderboard | **`missing`** | `leaderboard` ingests every run dir indiscriminately |
| Comparable-run protection | **`missing`** | `compare` does not check benchmark version, split fingerprint, prompt profile, or eval mode |

### Evaluation modes (mission Phase 3)

| Mode | Status |
|---|---|
| `context_given` | `partial` — it is the *implicit and only* behaviour; not named, not recorded, not selectable |
| `retrieval_required` | `missing` — no retriever, no corpus index, no retrieval metrics |
| `tool_assisted` | `missing` — `ToolSpec` schema exists (`stub`); no executor, no tool loop |

### Datasets

| Benchmark | Status | Detail |
|---|---|---|
| `smoke` | `implemented_and_live_verified` | 10 in-repo synthetic samples. Not a real benchmark — a pipeline fixture |
| `finqa` | `partial` | Real data converts (6,251/883/1,147, zero errors). Execution accuracy works. **Program accuracy is not implemented** — no program-eliciting prompt profile exists, so there is nothing to compare. Difficulty/task slices missing |
| `tatqa` | `missing` | — |
| `finance_reasoning` | `missing` | — |
| `financebench` | `missing` | — |
| `convfinqa` | `missing` | — |
| `secque` | `missing` | — |
| `smb_cfo` | `missing` | The entire RU-language coverage of the platform depends on this |

### Model providers

| Provider | Status | Detail |
|---|---|---|
| `mock` | `implemented_and_live_verified` | …as a *simulator*. It is **not** evidence the benchmark evaluates models |
| `openai_compatible` | `missing` | — |
| `ollama` | `missing` | Ollama **is running locally** with real models — this is the shortest path to a real result |
| OpenAI / Anthropic / Gemini / OpenRouter | `missing` | Will be `blocked` at live verification: **no API keys exist in this environment** |
| vLLM / llama.cpp | `missing` | No server running locally; reachable through `openai_compatible` once that exists |
| Transformers (direct) | `missing` | Declared as an optional extra in `pyproject.toml` but never implemented |

### Metrics and scoring

| Component | Status | Detail |
|---|---|---|
| Numeric parser (`evaluation/numeric.py`) | `implemented_unit_tested_only` | Parses %, bps, parens-negative, K/M/B, currencies; only ever fed the mock's synthetic strings |
| `exact_match` | `implemented_unit_tested_only` | — |
| FinQA execution accuracy | `partial` | Uses a `1e-3` **tolerance**; the official evaluator uses strict `==` after round-5. Currently named as if it were the official metric. It is not |
| FinQA program accuracy | `missing` | Official algorithm is sympy symbolic equivalence — see `metric_parity.md` |
| Metric parity vs official evaluators | **`missing`** | No official evaluator has ever been run against this code |
| Capability dimensions | `partial` | 7 defined (mission asks for 10). Tag→dimension routing is a hardcoded dict |
| Macro-averaging (sample→task→benchmark→capability) | `missing` | Current rollup is a flat mean — dataset size dominates |
| Financial Core / RAG / Agent scores | `missing` | — |
| Finance Capability Index | `missing` | — |
| Critical gates | `stub` | `GatesReport(evaluated=False)` — shape only, honestly |
| Failure taxonomy (25 types) | `missing` | `failures.jsonl` currently just re-dumps failed metric rows |
| Bootstrap CIs / paired comparison | `missing` | `confidence_intervals.json` is an empty placeholder |
| Finance-readiness verdict | `missing` | — |

### Reporting

| Component | Status |
|---|---|
| `summary.md` / `report.html` | `partial` — written, but contain no gates, no CIs, no failure distribution, no worst examples |
| Leaderboard | `partial` — CSV/MD/HTML written; **does not separate real from mock**, nor mode, nor modality |

## What must be true before any capability claim is made

1. A real model has produced real answers to real financial questions. *(Not yet true.)*
2. The request that reached it provably contained no answer key. *(Not yet true.)*
3. The metric that graded it matches the official evaluator, or is explicitly labelled as not
   matching. *(Not yet true.)*
4. Mock output cannot reach a leaderboard. *(Not yet true.)*

Each of these is addressed, in this order, by the work that follows this audit.
