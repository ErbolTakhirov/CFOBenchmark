# Known limitations

Written so that a reader can discount the results correctly. Everything here is a real constraint on
what the numbers in this repository mean.

## The evaluation is small, and the sample counts are stated everywhere

This ran on a **GTX 1650 with 4 GB of VRAM**. The costs are measured, not estimated — from
`latency_ms` in the real run artifacts:

| benchmark | s/sample (3B) |
|---|---|
| TAT-QA | 16.8 |
| FinanceReasoning | 19.0 |
| FinQA | 35.8 |
| ConvFinQA | 37 |
| FinanceBench (context-given) | 62.5 |
| FinanceBench (retrieval) | **109.5** |
| SMB-CFO | **150.2** |

SMB-CFO at 300 samples on the 3B alone is **12.5 GPU-hours**. The evaluation matrix a reader would
*assume* from the benchmark list is 75–100 GPU-hours. It does not fit, so the release runs a **frozen,
stratified manifest** and states every count. A smaller honest number beats a larger invalid one.

**Consequence:** confidence intervals here are wide. Where an interval contains zero, the report says
"no significant difference" — which is a statement about *this sample size* as much as about the
models, and it is labelled as such (`underpowered: true` when n < 30).

## qwen2.5:7b does not fit in the GPU

4.7 GB of weights on a 4 GB card: it **spills to CPU**. Every 3B-vs-7B latency comparison in this repo
is therefore a measurement of *this machine*, not a general claim about 7B inference cost. Correctness
comparisons are unaffected.

## SECQUE's analytical correctness is NOT_EVALUATED

Not omitted — **measured to be unmeasurable with the instruments available here.**

`llama3.2:3b`, 48 calibration cases whose correct verdict is known by construction:

| what the answer was | judge got it right |
|---|---|
| the expert's own answer | 100% |
| correct but concise | 100% |
| minor rounding | 100% |
| a refusal, where the filing plainly answers it | 100% |
| the direction of travel inverted | 80% |
| fluent, unsupported boilerplate | 50% |
| **an invented figure** | **33%** |
| **the wrong company entirely** | **33%** |

**False-positive rate 41% against a bar of 20%. False negatives: zero.** It never rejects a good
answer, and it waves through two-thirds of answers about the wrong company. `qwen3:8b` was tried first
and returns an **empty string** — it is a thinking model and spends its whole budget inside `<think>`,
at 116 s/call. `qwen2.5:7b` is refused outright: same family as the candidate, and a model grading its
own family is not evidence.

The judge is also **not wired into `eval`** (`judge_config: null`); the only entry point is
`financebench calibrate-judge`. So SECQUE reports deterministic diagnostics only.

## SECQUE's denominators differ between models

`secque_filing_identification` is graded on **62 of 80** samples for the 7B but **56** for the 3B — the
metric is not-applicable when the model declines or produces nothing to check. The 3B's *higher* score
(0.607 vs 0.494) therefore sits on a **smaller, self-selected subset**, and
**"the 3B identifies filings better than the 7B" is not a supportable claim.**
`secque_comparison_direction` is graded on 17 (3B) / 12 (7B) samples and is flagged
`underpowered: true`.

## No API provider has ever been called

OpenAI, Anthropic, Gemini and OpenRouter are implemented and unit-tested against a mocked transport.
**No API key exists in this environment, so none of them has ever made a successful call.** They are
labelled `implemented_not_live_verified`, and `financebench verify-providers` labels each by what
happens when you call it — not by a class attribute.

## Our FinanceBench metrics are not official

FinanceBench (the dataset) **ships no evaluator**. `financebench_answer_accuracy` and
`financebench_unsupported_numeric_claim` are ours, and they are named so they cannot be mistaken for an
official metric. FinQA, TAT-QA and FinanceReasoning *do* have official evaluators, and ours are
parity-tested against them (17 tests, zero skips).

## The hallucination detector was too generous until v3

A 0.5% tolerance window *after scaling* meant an invented `987,654,321` counted as "supported" by a
filing's `983` (983 × 10⁶ is 0.47% away). On a 733-number SEC filing the candidate set is so dense that
almost any large invented figure lands near something — so the detector got **weaker the more evidence
a document contained**, which is backwards, and worst exactly where hallucination matters most.

v3 requires **leading-digit agreement**. **Every unsupported-claim rate measured before it is an
understatement.**

## What is deliberately not built

- **External market-data tools.** A benchmark whose scores depend on what a third-party endpoint
  returned on a Tuesday is not a benchmark; it is a snapshot of a Tuesday. Tools are deterministic from
  the sample's own inputs — currency conversion uses the *ledger's* FX table, as an accountant would.
- **Multimodal.** Zero multimodal runs exist. The coverage field says `0.0` and the README says so.

## This benchmark does not certify autonomous financial safety

A good score means a model did well on these questions, on this hardware, on this date. It does not
clear a model to act on real money without a human reading the output.

## A tool run cannot be re-scored cheaply

Every other run is cached: the response cache keys on the request, so re-scoring one under a new
evaluator (`financebench resume`) replays the model's saved answers and costs **nothing**. That is how
the SECQUE runs were moved onto a common fingerprint for free.

**Tool-assisted runs are deliberately not cached** (`engine.py`). An agent run is a *conversation*
whose later turns depend on what the tools said in the earlier ones — there is no single request to
hash. So "re-score the tool run" means **re-run every turn against the model**.

The practical consequence, and it is worth knowing before you bump a metric version: a fingerprint
change after a tool run has completed costs you that entire run again. Freeze the evaluator *before*
the tool experiment, not after.
