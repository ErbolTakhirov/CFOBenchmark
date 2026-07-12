# FinanceBench v0.1.0-rc1 — generated result tables

Generated from the run artifacts. The narrative report, with the findings, is [`report.md`](report.md).

Evaluator fingerprint `80ca8a678b1c4fa1`. Every run below was scored by **this** evaluator; runs scored by a different one are not on this page, because they are not comparable and averaging them would be a lie.

Hardware: NVIDIA GeForce GTX 1650, 4096 MiB — Linux-6.19.14+kali-amd64-x86_64-with-glibc2.42.

> **A dash means NOT MEASURED. It never means zero.** An `INSUFFICIENT_COVERAGE` index is a refusal, not a missing number: the run did not ask enough to support the claim the index makes, and the reason is printed next to it.

## Verdict

| model | run | FCI | verdict |
|---|---|---|---|
| `ollama/qwen2.5:3b` | `tool_paired_v1-structured_financial_v1-conte` | **INSUFFICIENT_COVERAGE** — a critical gate failed — a single index would let a strong average hide the kind of error that is not a near-miss in finance | NOT_FINANCE_READY |

## What was measured

| model | run | metric | value |
|---|---|---|---|
| `ollama/qwen2.5:3b` | `tool_paired_v1-structured_fina` | exact_match | 0.140 (n=150) |
| `ollama/qwen2.5:3b` | `tool_paired_v1-structured_fina` | finqa_answer_accuracy | 0.147 (n=75) |
| `ollama/qwen2.5:3b` | `tool_paired_v1-structured_fina` | tatqa_exact_match | 0.173 (n=75) |
| `ollama/qwen2.5:3b` | `tool_paired_v1-structured_fina` | tatqa_f1 | 0.268 (n=75) |
| `ollama/qwen2.5:3b` | `tool_paired_v1-structured_fina` | tatqa_scale_accuracy | 0.720 (n=75) |

## What was NOT measured

- **SECQUE analytical correctness: `NOT_EVALUATED`.** No available judge passes calibration. `llama3.2:3b` scores 75% accuracy with a **41% false-positive rate** against a 20% bar — it never rejects a good answer, and waves through two-thirds of answers that name the wrong company or contain a fabricated figure. This is a measurement, not an omission, and it is **never** reported as zero.
- **No API provider is live-verified.** OpenAI, Anthropic, Gemini and OpenRouter are implemented and unit-tested against a mocked transport. No API key exists in this environment, so **none of them has ever made a successful call**.
- **No multimodal run exists.** `multimodal_coverage: 0.0` in every run.

## Limitations

See [`docs/known_limitations.md`](../../docs/known_limitations.md).

---

**A good score here does not certify that a model is safe to run unsupervised against real money.** It means it did well on these questions, on this hardware, on this date.

