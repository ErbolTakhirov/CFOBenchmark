# FinanceBench v0.1.0-rc1 — Financial LLM Reliability Benchmark

**Pre-release.** Not a stable v0.1.0, and not marked as latest.

A benchmark platform for financial LLMs whose distinguishing feature is that it **refuses to report
numbers it cannot stand behind**. This release candidate ships two findings, ten bugs it found in
itself, and one score it declines to give.

---

## The two findings

### 1. Giving a small model tools made it significantly worse

The same **150 frozen sample ids** (FinQA + TAT-QA), run twice on `qwen2.5:3b` — once answering
directly, once with a calculator, a formula registry, a table query and an FX converter available:

| | direct | tool-assisted | difference | 95% CI (paired bootstrap) |
|---|---|---|---|---|
| FinQA answer accuracy | **0.147** | **0.027** | +0.120 | `[+0.040, +0.213]` |
| TAT-QA exact match | **0.173** | **0.067** | +0.107 | `[+0.027, +0.200]` |

Both intervals exclude zero. On FinQA the 2×2 is `only-direct-right 11 / only-tools-right 2 /
both-right 0`.

**And not because it used the tools badly — because it barely used them.** `tool_invocation_rate =
0.013`: it called a tool on **2 of 150 questions**. The agent scaffolding degraded its arithmetic while
it ignored the tools, and cost **27% more tokens** doing it.

The sandbox was never breached: `tool_security_rejection = 1.000`, and that gate is *critical* with a
threshold of **1.0** — one escape is not a low score, it is a failed release.

### 2. Retrieval is not the bottleneck. The model is.

Fixing document scoping raised page recall **4.0% → 18.7%** (4.7×) and produced **no statistically
supported improvement in answer accuracy**, while `generation_error_after_retrieval` rose 2 → 7.

Reading all seven of those by hand: every one is a **JSON-envelope failure**. The retriever found the
page; the model computed something; it answered in its own shape. The fix is a parser, not an index.

Full 6-arm ablation over a real 12,013-page, 84-filing corpus. Dense retrieval is **dramatically worse
at finding the company** (27.3% document recall vs BM25's 79.3%) — the questions name a company and a
year, which is a lexical match, and the embedding blurs exactly that.

---

## The score this release refuses to give

**SECQUE analytical correctness: `NOT_EVALUATED`.** Not omitted — *measured to be unmeasurable with
the instruments available here.*

The judge was built, and then **tested before being believed**. `llama3.2:3b`, on 48 cases whose
correct verdict is known by construction:

| what the answer was | judge got it right |
|---|---|
| the expert's own answer | 100% |
| a refusal, where the filing plainly answers it | 100% |
| **an invented figure** | **33%** |
| **the wrong company entirely** | **33%** |

**False-positive rate 41% against a bar of 20%. False negatives: zero.** It never rejects a good
answer, and it waves through two-thirds of answers about the wrong company. It is a yes-man, and the
calibration says so in numbers.

So the dimension reports `NOT_EVALUATED`. **Never zero.** A zero would say the model failed; the truth
is that no instrument here can measure it.

---

## Ten bugs, and every one produced a plausible number

None crashed. Each was found by disbelieving a result.

1. **Both models reported a Financial Core Score of `0.900`.** Every SECQUE capability dimension was
   fed the *absence-of-hallucination* metric — so "document grounding" silently meant "did it avoid
   inventing a number", which a model that emits no numbers passes perfectly. The metrics that
   discriminate between the models fed **no dimension at all**. True values: they agree with the
   expert's figures **8%** and **11%** of the time, and the 7B names the **wrong company in 51%** of
   its answers. → **0.354 / 0.307**.
2. **Provider timeouts were scored as the model's financial failures.**
3. **`secque_comparison_direction` reported `1.000` while missing the clearest inversion in the set** —
   it graded only the questions it found easy.
4. **`summary.md` rendered every SKIPPED gate as `**FAIL**`**, fabricating a critical failure in every
   run on disk.
5. **`arguments_valid` was read off English prose**, so a call with plainly wrong arguments was
   recorded as valid.
6. **`n` counted samples a metric never graded**, overstating the evidence under a mean by a third.
7. **The leaderboard could never display an FCI.**
8. **Two different retrieval arms resolved to the same run id** and would have overwritten each other.
9. **`resume` did not restore the run** — it dropped the prompt profile, eval mode, retriever, scoping
   *and* the frozen manifest.
10. **`CITATION.cff` claimed MIT while `LICENSE` says Apache-2.0.**

## One bug deliberately NOT fixed

`financebench_answer_accuracy` allows a **1% relative band**, which credits an answer wrong by **$27
million** (gold $12,645.00, model 12,672.0). I judge that too loose. **I have not changed it.** The band
was chosen before any results existed; tightening it now, having seen exactly which answers it credits,
would be selecting a metric rule by its effect on the score — and that the effect would be to *lower*
the score does not make it acceptable.

---

## What is verified, and what is not

| | |
|---|---|
| ✅ **Live-verified** | FinQA, TAT-QA, FinanceReasoning (**official metrics parity-tested against the real upstream evaluators — 17 tests, ZERO skips**), FinanceBench (context + retrieval), SMB-CFO, ConvFinQA (both protocols), tool-assisted agent. Ollama. |
| 🔬 **Deterministic diagnostics** | SECQUE — numeric agreement, filing identification, comparison direction, unsupported claims. |
| ⛔ **NOT_EVALUATED** | SECQUE analytical correctness. No judge passes calibration. |
| ⚠️ **Implemented, never called** | OpenAI, Anthropic, Gemini, OpenRouter. **No API key exists here.** |
| ⛔ **Not evaluated** | Multimodal. Zero multimodal runs exist. |

## Reproducing this

`release_manifest.json` names everything that could change a number: dataset hashes, exact sample ids,
model digests and quantization (`Q4_K_M`, read from the runtime — a config file records an intention),
runtime versions, prompt/parser/metric/scoring versions, seeds, retrieval index fingerprint, hardware,
and the commit. It validates against `release_manifest.schema.json`, and the validator has seven tests
proving it can say **no**.

See `reproduction.md`. Verify with `sha256sum -c checksums.txt`.

---

## ⚠️ This benchmark does not certify autonomous financial safety

A good score means a model did well on these questions, on this hardware, on this date. **It does not
clear a model to act on real money without a human reading the output.** Neither model in this release
came close: both are `NOT_FINANCE_READY`.
