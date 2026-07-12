# FinanceBench v0.1.0-rc1 — release report

**qwen2.5:3b and qwen2.5:7b, on a GTX 1650 (4 GB).** Evaluator fingerprint `80ca8a678b1c4fa1`.

> **A dash means NOT MEASURED. It never means zero.**
> An `INSUFFICIENT_COVERAGE` index is a **refusal**, not a missing number: the run did not ask enough
> to support the claim an index makes, and the reason is printed beside it.

---

## The headline: tools do not help either model — and they *destroy* the small one

The paired experiment ran the **same 150 sample ids** (FinQA + TAT-QA, frozen manifest
`tool_paired_v1`, `id_hash=7c839cfbc46cb862`) four ways: each model answering directly, and each with a
calculator, a formula registry, a table query, a CSV query and an FX converter available. Same
questions, same evaluator (`80ca8a678b1c4fa1`), same prompt profile, temperature 0, zero provider
errors.

| | 3B direct | 3B + tools | 7B direct | 7B + tools |
|---|---|---|---|---|
| FinQA answer accuracy | 0.1467 | **0.0267** | **0.2667** | 0.2400 |
| TAT-QA exact match | 0.1733 | **0.0667** | **0.2667** | 0.2667 |
| exact match (all 150) | 0.1400 | **0.0600** | 0.1667 | 0.2000 |

### Paired bootstrap — the four comparisons that matter

| comparison | metric | difference | 95% CI | |
|---|---|---|---|---|
| 3B direct **vs** 3B + tools | FinQA | +0.120 | `[+0.040, +0.213]` | **tools HURT, significantly** |
| 3B direct **vs** 3B + tools | TAT-QA | +0.107 | `[+0.027, +0.200]` | **tools HURT, significantly** |
| 7B direct **vs** 7B + tools | FinQA | +0.027 | `[-0.080, +0.133]` | no significant difference |
| 7B direct **vs** 7B + tools | TAT-QA | +0.000 | `[-0.093, +0.093]` | **identical** |
| 3B + tools **vs** 7B direct | FinQA | −0.240 | `[-0.347, -0.133]` | **7B wins by 10×** |
| 3B + tools **vs** 7B direct | TAT-QA | −0.200 | `[-0.307, -0.093]` | **7B wins** |

**The two models fail differently, and the difference is the finding.**

### Why: the 3B never picks the tools up. The 7B does — and gains nothing.

| tool metric | 3B | 7B |
|---|---|---|
| `tool_invocation_rate` | **0.013** (2 of 150) | **0.193** (29 of 150) |
| `tool_argument_validity` | 0.500 (n=2) | **0.931** (n=29) |
| `tool_execution_success` | 0.500 (n=2) | 0.552 (n=29) |
| `tool_result_utilization` | 1.000 (n=1) | **0.688** (n=16) |
| `tool_error_recovery` | 0.000 (n=1) | 0.188 (n=16) |
| `tool_hallucination_rate` | 1.000 (n=2) | 1.000 (n=29) — neither ever invented a tool |
| `tool_security_rejection` | **1.000** (n=150) | **1.000** (n=150) — **the sandbox was never breached** |

The 3B calls a tool on **2 of 150 questions**. It is not misusing the toolbox; it is *ignoring* it —
and it still loses five-sixths of its FinQA answers. So the damage is done by the **agent scaffolding
itself**: the JSON envelope asking the model to choose between emitting a tool call and emitting an
answer. A 3B model asked to route its reasoning through a protocol gets worse at the underlying
arithmetic *while ignoring the protocol*, and pays 27% more tokens (207,473 vs 163,028) to do it.

The 7B is a different story. It invokes tools **15× more often**, forms valid arguments **93%** of the
time, and uses what comes back **69%** of the time. It can work the protocol. **And it gains nothing
for it** — every 7B direct-vs-tools interval contains zero.

The `n=1` and `n=2` in the 3B column are printed with their denominators and are **not claims about
anything**. A bare `tool_result_utilization = 1.000` for the 3B would be a lie by omission.

### The six questions, answered

**Does tool access improve correctness?** **No.** It significantly *reduces* it for the 3B, and makes
no significant difference for the 7B. Not one interval favours tools.

**Does it reduce catastrophic numeric errors?** No.

**Does the model actually invoke tools?** The 3B: almost never (1.3%). The 7B: sometimes (19.3%).

**Does it use correct tool output?** The 7B does, 69% of the time (n=16). The 3B executed one tool
successfully in the entire run.

**Does orchestration introduce new failures?** **Yes — and for the 3B that is the entire effect.**

**Can a smaller model with tools outperform a larger model without tools?**
**Emphatically no.** 3B+tools scores **0.027** against 7B-direct's **0.267** — *ten times worse*, with
the interval excluding zero on every metric. On this evidence, spending a small model's context budget
on a tool protocol is strictly worse than spending it on a bigger model.

---

## The release group: the only run that can earn a Finance Capability Index

220 frozen samples (`release_v0_1`, `id_hash=e9b815fdb84fb9ab`), stratified across six benchmarks and
across SMB-CFO's **24 task families**, plus 10 adversarial prompt-injection samples. This is the only
run whose coverage can support an index at all: the FCI is withheld unless **one** run asked about
SMB-CFO *and* document grounding *and* refusal together.

**qwen2.5:3b, 220/220 evaluated, zero provider errors:**

| | |
|---|---|
| **Finance Capability Index** | **WITHHELD** — a critical gate failed |
| verdict | **NOT_FINANCE_READY** |
| `numeric_accuracy` | **0.116** (gate: ≥ 0.50) — **FAIL, critical** |
| `invalid_output_rate` | **0.173** (gate: ≤ 0.10) — **FAIL** |
| `prompt_injection_obeyed_rate` | **0.000** (gate: 0.0) — **PASS** |
| `catastrophic_numeric_error_rate` | 0.036 (gate: ≤ 0.05) — PASS |
| `failed_refusal_rate` | 0.000 — PASS |
| `unsupported_claim_rate` | 0.000 — PASS |

The index is **refused, not asterisked.** `capabilities.json` records why in plain words.

### Two zeros, and they are different failures

`finance_reasoning_accuracy = 0.000` (n=40) and `smb_cfo_accuracy = 0.000` (n=37). Both are real. They
are not the same thing.

**FinanceReasoning** is a genuine reasoning failure: the model emits numbers, and they are simply
wrong — 1328 where the answer is 1152; 368 where it is 22.

**SMB-CFO is a FORMAT failure.** The model answers in *its own JSON shape*:

```json
{"monthly_budget": 2595, "category": "Cloud hosting"}
{"data": [{"supplier": "Northwind Studio", "amount": 2486.58}]}
```

when the prompt asked for `{"answer": ..., "numeric_value": ...}`. It may well hold the right number —
it simply cannot say it in the requested envelope, so nothing can be read out. That is counted
separately (**38 `invalid_structured_response`** of 220), and it is what fails the
`invalid_output_rate` gate.

The distinction matters because the fixes are opposite: one needs a better model, the other needs a
parser or a prompt. A single "accuracy: 0.0" would have sent you to fix the wrong one.

**The prompt-injection gate fired for the first time in this project's history** — the release manifest
is the first run to carry real injection samples — and the model **resisted every one**.

---

## SECQUE: both models, on identical stratified samples

80 tasks, stratified across Analysis / Comparison / Ratio / Risk. Same 80 sample ids for both models,
same evaluator.

| | qwen2.5:3b | qwen2.5:7b |
|---|---|---|
| `secque_numeric_agreement` | 0.080 (n=62, 18 n/a) | **0.115** (n=62, 18 n/a) |
| `secque_filing_identification` | 0.607 (n=56) | 0.494 (n=62) |
| `secque_unsupported_numeric_claim` | 0.938 (n=80) | 0.900 (n=80) |
| `exact_match` | 0.000 | 0.000 |
| **Financial Core Score** | **0.354** | **0.307** |
| **analytical correctness** | **`NOT_EVALUATED`** | **`NOT_EVALUATED`** |
| verdict | **NOT_FINANCE_READY** | **NOT_FINANCE_READY** |

**Both models agree with the expert analyst's figures roughly one time in ten**, and the 7B names the
**wrong company in 51% of its answers**. These are not models that can read a 10-K.

> ⚠️ **The 3B's higher filing-identification score is NOT a supportable claim.** It is graded on **56**
> of 80 samples versus the 7B's **62** — the metric is not-applicable when the model declines or
> produces nothing to check — so the 3B's number sits on a **smaller, self-selected subset**. Different
> denominators; not a comparison.

**Before this release, both models reported a Financial Core Score of 0.900.** Every SECQUE capability
dimension was being fed the *absence-of-hallucination* metric, so "document grounding" and "table/text
reasoning" were both reporting "did it avoid inventing a number" — which a model that emits no numbers
passes perfectly. Fixed; see `manual_validity_review.md`.

---

## Retrieval: the bottleneck is the model, not the retriever

The full 6-arm ablation over a real **12,013-page, 84-filing** corpus (no model in the loop):

| retriever | scope | page recall @20 | doc recall @20 |
|---|---|---|---|
| bm25 | open-corpus | 7.3% | 79.3% |
| bm25 | doc-scoped | 27.3% | 100% |
| dense | open-corpus | 2.0% | **27.3%** |
| dense | doc-scoped | 34.0% | 100% |
| hybrid | open-corpus | 4.7% | 67.3% |
| **hybrid** | **doc-scoped** | **38.7%** | 100% |

Dense retrieval is **dramatically worse at finding the company** (27.3% document recall vs BM25's
79.3%) — the questions name a company and a year, which is a lexical match, and the embedding blurs
exactly that. It is better at finding the *page* once the company is known. Hybrid gets both.

**But retrieval quality is not what is limiting the answers.** Fixing document scoping raised page
recall from 4.0% → 18.7% (**4.7×**) and produced **no statistically supported improvement in answer
accuracy** (4 → 2 correct of 89; the 95% CI contains the old value). Meanwhile
`generation_error_after_retrieval` rose from 2 → 7.

Reading those 7 failures by hand (`manual_validity_review.md`): **all of them are JSON-envelope
failures.** The retriever found the page, the model computed something, and then answered in its own
shape — `{"financial_metric": "Retention Ratio", "value": 0.31}` instead of the requested envelope. The
fix is a parser or a prompt. It is not an index.

| stage | value |
|---|---|
| retrieval succeeded (page recall @10, bm25 doc-scoped) | 18.7% |
| end-to-end answer accuracy | 2.25% |
| **oracle** (context handed to the model) | **23.6%** |

The gap between 23.6% and 2.25% is the retriever. The gap between 23.6% and 100% is the model.

---

## Coverage, and what is refused

| | |
|---|---|
| Evaluator fingerprint | `80ca8a678b1c4fa1` — every run on this page was scored by it |
| Frozen manifests | `tool_paired_v1` (150 ids), `release_v0_1` (220 ids, stratified across SMB-CFO's 24 task families + 10 injection samples) |
| Parity tests | **17 passed, ZERO skipped**, against the real upstream FinQA / TAT-QA / FinanceReasoning evaluators |
| Security tests | 411 passed |
| Sandbox gate | **PASSED** (`tool_security_rejection = 1.000` over 150 samples) |

### What is NOT measured

- **SECQUE analytical correctness: `NOT_EVALUATED`.** `llama3.2:3b` fails calibration with a **41%
  false-positive rate** against a 20% bar — it never rejects a good answer, and waves through
  two-thirds of answers that name the **wrong company** or contain an **invented figure**. `qwen3:8b`
  returns an empty string (a thinking model; it spends its budget inside `<think>`). `qwen2.5:7b` is
  refused: same family as the candidate. **This is a measurement, not an omission, and never a zero.**
- **No API provider is live-verified.** OpenAI, Anthropic, Gemini, OpenRouter are implemented and
  unit-tested against a mocked transport. **No API key exists here; none has ever made a call.**
- **No multimodal run exists.** `multimodal_coverage: 0.0` everywhere.

---

## Limitations

See [`docs/known_limitations.md`](../../docs/known_limitations.md). The load-bearing ones:

- **The 7B does not fit in the GPU** (4.7 GB of weights, 4 GB of VRAM) and spills to CPU. Every
  latency comparison here measures *this machine*, not the models.
- **The PDF text extraction strips spaces out of line items** — `Propertyandequipment,net`. The
  evidence is present and the model cannot find it. This depresses every FinanceBench number in this
  repo, and it is a *pipeline* limitation, not a model one.
- **`financebench_answer_accuracy` allows a 1% band**, which credits an answer wrong by $27M
  (`id_10285`: gold $12,645.00, model 12,672.0). I judge that too loose, and I have **not changed it** —
  tightening a tolerance after seeing which answers it credits is selecting a metric by its effect on
  the score, and that the effect would be to *lower* the score does not make it acceptable.

---

**A good score here does not certify that a model is safe to run unsupervised against real money.** It
means it did well on these questions, on this hardware, on this date. Neither of these models did.
