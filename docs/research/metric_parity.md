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

### The bug the parity suite caught

The official `scale_to_num` returns Python **`int`** for every scale except percent:

```python
def scale_to_num(scale):
    num = 1                                  # int, not 1.0
    if 'thousand' in scale: num = 1000       # int
    elif 'percent' in scale: num = 0.01      # float
    return num
```

That `int` flows through `to_number` into `round()` — and `round(-134, 4)` is the **int** `-134`
while `round(-134.0, 4)` is the **float** `-134.0`. `normalize_number` then `str()`s the result, so
typing those constants as floats (which is the natural thing to do) makes `normalize_answer("(134)")`
produce `"-134.0"` instead of `"-134"`, and it no longer exact-matches a gold of `"-134"`.

A silent, systematic deflation of EM and F1 on **every negative number in the dataset** — invisible
to inspection, and caught immediately by running the official code side by side. The int-ness is now
asserted by a type check in the test suite, and the docstring says not to "tidy" it away.

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
ported faithfully, with one deliberate deviation:

| Official behaviour | Here | Why |
|---|---|---|
| `normalize()` ends with a bare `eval(prediction)` on model output | A **safe AST evaluator**: numeric literals, tuples of them, and `+ - * / ** // %` only. No names, calls, attributes or subscripts | Running a language model's output through `eval` is arbitrary code execution — one adversarial dataset away from a compromised machine. The parity suite proves the two agree on every realistic prediction, because `eval` was only ever doing arithmetic |
| An LLM is used as a fallback answer extractor | Optional, off by default | An LLM in the *grading* path makes the metric non-deterministic. Runs using it are marked provisional |

### An upstream bug that is reproduced on purpose

Python parses `"1,234.56"` as the **tuple** `(1, 234.56)`, not as a comma-formatted number. The
official `normalize()` `eval`s it into exactly that tuple, and `get_acc` then takes element `[0]` —
so **the official evaluator scores a model that answers `1,234.56` as if it had answered `1`**.

That is a bug, and it was tempting to fix. It is *not* fixed here, and the safe evaluator therefore
permits tuples solely to reproduce it. The reason: a native metric that disagrees with the official
one is not a native metric. Its numbers would not be comparable with any published FinanceReasoning
result — which is the entire point of implementing the official metric rather than inventing one.

In practice it rarely bites: the structured prompt asks for a bare `numeric_value`, and the metric
prefers that float over the model's prose, so `normalize()` is only a fallback.

### Upstream data quality, recorded rather than smoothed over

6 of the 1,000 `easy` records ship an **empty context** — the question cannot be answered because
there is nothing to answer it from. They are kept (dropping them would break count-parity with
published numbers, which include them) and flagged with `metadata.context_empty=true`. So ~0.6 % of
the easy split is a floor no model can clear, and that is the benchmark's doing, not the model's.

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
