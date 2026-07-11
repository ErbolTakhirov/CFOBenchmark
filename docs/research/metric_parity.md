# Official metric parity

**Rule this project holds itself to:** a metric may be called *official* only if it produces the
same number as the benchmark's own evaluator on the same predictions. Where behaviour differs, the
difference is written down here **and the metric is renamed** so it cannot be mistaken for the
official one.

Parity suites live in `tests/parity/`. They run the **real official evaluator code** (from the
clones in `/tmp/financebench-references/`, executed in an isolated venv with `numpy`, `scipy`,
`pandas`, `sympy`) over a fixed set of predictions, run this project's implementation over the same
predictions, and assert the outputs are identical. Fixtures are committed so CI fails on regression.

---

## FinQA

Official evaluator: `code/evaluate/evaluate.py`.

### Execution accuracy

```python
invalid_flag, exe_res = eval_program(pred_tokens, table)
if invalid_flag == 0:
    if exe_res == gold_res:          # <-- strict equality, NO tolerance
        exe_correct += 1
```

`eval_program` rounds its result to 5 decimals; the comparison against `qa["exe_ans"]` is then a
plain `==`. **The official metric takes a predicted *program*, not a free-text answer.**

This forces an honest split into two differently-named metrics:

| Metric name here | Input | Comparison | Official? |
|---|---|---|---|
| `finqa_execution_accuracy` | a predicted **program** (needs the `program_v1` prompt profile) | official: execute, round-5, strict `==` | **Yes** — parity-tested |
| `finqa_answer_accuracy` | a predicted **final answer** (the `structured_financial_v1` profile) | numeric match within a small tolerance | **No** — and it does not claim to be |

The second is a legitimate, useful metric for direct-answer prompting — it is simply *not the
official one*, because the official one has no defined behaviour for a free-text answer. The
previous name (`finqa_execution_accuracy` applied to free-text with a `1e-3` tolerance) conflated
the two, and has been corrected.

### Program accuracy

Official `equal_program(gold_tokens, pred_tokens)` is **symbolic equivalence**, not string equality:

1. Build a symbol map from the **gold** program: every non-`#` argument becomes `a0`, `a1`, …; every
   whole `table_*` step becomes a symbol.
2. Structurally validate the prediction: token `i` must be an op when `i % 4 == 0` and `)` when
   `(i+1) % 4 == 0`; a `#N` back-reference may not point at its own index or later.
3. **Reject any prediction that uses an operand absent from the gold's symbol map** — a model may
   not invent literals, even numerically correct ones.
4. Recursively expand each program's final step into an infix expression over those symbols, run
   both through sympy's `simplify(..., evaluate=False)`, and compare.

Consequence: `subtract(a,b), divide(#0,b)` and a differently-shaped-but-algebraically-identical
program both count as correct — which is why a string comparison would have been wrong, and why
`sympy` is a real dependency rather than an optional one.

**Prerequisite:** program accuracy is meaningless unless the model is *asked* for a program. The
`program_v1` prompt profile exists for exactly this. Program accuracy is reported only for runs
using it, and is absent (not zero, not faked) otherwise.

---

## TAT-QA

Official evaluator: `tatqa_eval.py` + `tatqa_metric.py` + `tatqa_utils.py`.
Prediction contract: a JSON dict `{uid: [answer, scale]}`.

### Exact match

Normalized-span-set equality:

```python
if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
    exact_match = 1.0
```

### Numeracy-aware F1 — Hungarian alignment is genuinely required

```python
from scipy.optimize import linear_sum_assignment
...
row_ind, col_ind = linear_sum_assignment(-scores)
```

This is quoted from the official `tatqa_metric.py`. The alignment is **real**, not an invention:
for multi-span answers the evaluator finds the optimal 1-to-1 matching between predicted and gold
spans, then averages the per-bag token-F1 over `max(len(gold), len(pred))` slots and rounds to 2
decimals. Any implementation using naive bag-overlap would disagree with the official numbers.

Two behaviours that are easy to get wrong and are therefore parity-tested explicitly:

- **`f1 := em` is forced** when the gold answer type is `arithmetic` or `count`.
- **Scale is folded into the compared string**, not scored separately inside EM/F1:
  `'%.4f' % (round(ans_num, 2) * scale_to_num(scale))`, where `scale_to_num` maps
  `thousand→1e3`, `million→1e6`, `billion→1e9`, **`percent→0.01`**. Scale accuracy is *additionally*
  tracked as its own number (`scale_em`).
- `add_percent_pred` appends an alternative prediction string so a model answering `0.2342` is still
  matched against a gold of `23.42` + `scale=percent`.

`scipy` is therefore a real dependency of the TAT-QA metric.

---

## FinanceReasoning

Official evaluator: `utils/evaluation_utils.py` (adapted from Yale-NLP FinanceMath).

```python
def within_eps(pred: float, gt: float):
    eps = abs(gt) * 0.002
    return pred >= gt - eps and pred <= gt + eps
```

**0.2 % relative tolerance**, inclusive on both sides. Note it is relative to the *gold*, so a gold
of `0` admits only an exact `0`.

The official `normalize()` additionally strips currency words/symbols, takes the right-hand side of
`=` and `≈`, strips `%`, maps `true/yes → True` and `false/no → False`, and drops units. It is
ported faithfully. Two deliberate deviations, recorded here:

| Official behaviour | Here | Why |
|---|---|---|
| `normalize()` starts with a bare `eval(prediction)` on model output | **Not ported.** A safe numeric parser is used instead | Executing model output as Python is an arbitrary-code-execution hole. The parity suite confirms identical results on all fixture predictions; a prediction that *only* the `eval` path could parse would be a divergence, and the suite would surface it |
| An LLM is used as a fallback answer extractor | Optional, off by default | An LLM in the *grading* path makes the metric non-deterministic. Runs using it are marked provisional |

---

## ConvFinQA

Shares FinQA's `eval_program` / `equal_program` (same authors, same code lineage), so the same
parity fixtures and the same two-metric split apply. Conversation-level metrics
(conversation completion accuracy, context-loss rate, error-propagation rate) are **ours** — the
official repo defines no such metric — and are named without the "official" claim.

---

## FinanceBench and SECQUE — no official metric exists

FinanceBench ships an `evaluation_playground.ipynb` and no evaluator. SECQUE's public artifact is a
dataset plus an agent-harness server. **Every metric this project reports on either benchmark is its
own**, is named `financebench_*` / `secque_*`, and is never presented as official or comparable to
published leaderboard numbers.

---

## Parity status

| Suite | Status |
|---|---|
| `tests/parity/test_finqa_parity.py` — execution accuracy | see `pytest tests/parity` |
| `tests/parity/test_finqa_parity.py` — program accuracy | see `pytest tests/parity` |
| `tests/parity/test_tatqa_parity.py` — EM / F1 / scale | see `pytest tests/parity` |
| `tests/parity/test_finance_reasoning_parity.py` — 0.2 % tolerance | see `pytest tests/parity` |
| `tests/parity/test_convfinqa_parity.py` — exec / program | see `pytest tests/parity` |

This table deliberately does not assert "PASSED" in prose. Run the suite; it is the claim.
