# Manual validity review — v0.1.0-rc1

**38 cases**, read individually from real run artifacts, and classified by hand against the automatic
label. The point of this exercise is not to confirm the evaluator. It is to find the cases where the
evaluator is confidently wrong — which is the only failure mode that matters, because a benchmark that
crashes gets fixed and a benchmark that quietly reports a plausible number gets cited.

**Agreement: 34 / 38 = 89.5%.**

Of the four disagreements, **two were genuine evaluator defects and are fixed in this release**; two
are tolerance judgements that I deliberately did **not** change, for a reason given below.

---

## Summary

| category | n | agree | disagree |
|---|---|---|---|
| Correct answers | 10 | 8 | **2** — the 1% tolerance credits a $27M error |
| Wrong numeric answers | 6 | 6 | 0 (but see the taxonomy note) |
| Unsupported claims | 5 | 4 | **1** — a metric failed to fire on an inverted direction |
| Refusals | 6 | 6 | 0 |
| Retrieval-success / generation-failure | 5 | 5 | 0 |
| Tool traces | 6 | 5 | **1** — `arguments_valid` was recorded as True for malformed args |

---

## The four disagreements

### 1 & 2. The 1% tolerance credits an answer that is wrong by $27 million — NOT CHANGED

`financebench:open_source:financebench_id_10285`

| | |
|---|---|
| gold | **$12,645.00** |
| model | **12,672.0** |
| automatic | **PASS** |
| human | **wrong** |

`grounding.py:301` allows a 1% relative band: `|12672 − 12645| = 27`, and 1% of 12,645 is 126. The
same band credits `financebench_id_04672` (gold $8.70, model 8.73).

These are millions of dollars. A model that computes 12,672 where the filing says 12,645 has made an
arithmetic error, not a rounding error.

**And I have not changed it.** The 1% band was chosen deliberately, before any results existed, with a
stated rationale ("filings round, and so do analysts"). Tightening it *now* — having seen exactly which
answers it credits — would be selecting a metric rule by its effect on the score. That the effect would
be to *lower* the score does not make it acceptable: the rule is "never change a metric rule after
seeing what it does to a number", and it does not have an exception for changes that happen to look
humble.

It is recorded here as a **validity threat**, and as a v0.1.1 item to be decided on its merits and
applied to every run at once. Readers should treat `financebench_answer_accuracy` as *"within 1% of the
filing's figure"*, which is what it measures and what it is now named in the docs.

### 3. A metric returned NOT-APPLICABLE on the clearest inversion in the set — FIXED

`secque:test:q_Ra007`, from the live 7B run:

| | |
|---|---|
| gold | "EBIT 2018: **$4,379 million** / EBIT 2017: **$4,945 million**" — EBIT **fell** |
| model | "NIKE's EBIT **increased** from $5,192 million in 2017 to $5,525 million in 2018" |
| `secque_unsupported_numeric_claim` | **flagged** ✅ (both figures are invented) |
| `secque_comparison_direction` | **`None` — not applicable** ❌ |

The direction metric looked for a direction *word* in the expert's answer. The expert states the
direction of travel by listing two dated figures at least as often as by writing "decreased" — so the
metric declared itself inapplicable on precisely the case it exists for, and then reported **1.000**
for the 7B, computed over the twelve easy questions where the expert did use a direction word.

**A 1.000 computed only over the questions a metric finds easy is not a lenient score. It is an
artifact of the metric's own coverage.**

Fixed: the gold's direction is now derived from its own dated figures when no direction word is
present, conservatively (two distinct years, one unambiguous figure each, actually differing —
otherwise `None`). `secque_comparison_direction` v1 → v2; both SECQUE runs re-scored.

### 4. `arguments_valid` was read off English prose — FIXED

From the tool trace, verbatim:

```
percent_change({'old": 2.9, "new": 2.7}}': [[2]]})
  tool_exists=True  arguments_valid=True  executed=False
  error: 'old' is not a number: None
```

The model emitted malformed JSON, which parsed into a garbage dict with one nonsense key. The tool
rejected it. And the trace recorded **`arguments_valid=True`** — because validity was inferred by
substring-matching the error message (`"must be an object" not in message and "requires" not in
message`), and this error's phrasing matched neither.

A reported metric was being read off English prose. Fixed: errors now carry a `ToolErrorKind`, and
nothing infers anything. `tool_execution_success` v1 → v2; **every v1 argument-validity number is an
overstatement.**

---

## A taxonomy imprecision (no score affected)

Three of the six "wrong numeric" cases are **yes/no questions**:

- `id_00288` — "Was there any drop in Cash & Cash equivalents?" gold **Yes** (~42% decline), model **No**
- `id_00438` — model **Yes**, gold **No**
- `id_00956` — model **Yes**, gold **No**

They are labelled `wrong_number` although no number is involved. The *metric* is correct in every case
(`passed=False`), so **no score is wrong** — but `failure_distribution` reports `wrong_number: 35` when
some of those are wrong booleans, which overstates the arithmetic problem and understates the reasoning
one. Cosmetic, recorded, not urgent.

---

## Cases where the automatic label was right, and worth reading anyway

### The refusals are genuinely unnecessary — but the PDF extraction is a confound

`financebench_id_05915` — "FY2018 fixed asset turnover for CVS Health". The model refused:
*"The provided financial statements do not contain the necessary information."*

They do. The context contains `Propertyandequipment,net 11,349 / 10,292` and `Totalrevenues 194,579`,
and 194,579 / ((11,349 + 10,292)/2) = **17.98**, exactly the gold. The label `unnecessary_refusal` is
**correct**.

But note the extracted text: **`Propertyandequipment,net`** — the PDF-to-text pass has stripped the
spaces out of every line item. The evidence is present and the model could not find it. That is a real
property of the corpus, it depresses every FinanceBench number in this repo, and it is a *pipeline*
limitation rather than a model one.

### "Generation failed after retrieval" is a JSON-envelope failure, not a reasoning one

All five cases are the model returning valid JSON in **its own shape** rather than the requested one:

```
{"financial_metric": "Retention Ratio", "value": 0.31}      # asked for {"answer": ..., "numeric_value": ...}
{"error": "Unsupported query, expected a value for field `value`"}
{}
```

The retriever found the page. The model computed something. The envelope was wrong, so nothing could be
read out of it. This is `invalid_structured_response` wearing a retrieval label — the distinction
matters, because the fix is a parser or a prompt, not an index.

### The tools: the model does not use them

Five of six samples: **zero tool calls**. The model did the arithmetic in its head while a calculator
sat unused in its context. The sixth called `percent_change` with malformed arguments and it failed.

**Zero successful tool executions in the entire run**, so `tool_result_utilization` is `None` — not
applicable, because nothing ever executed and there was therefore no result to ignore. A `0.0` there
would read as *"the model ignored its tools"*, which would be false and is a different, less damning
failure than the true one.

---

## What this review changed

| finding | action |
|---|---|
| `secque_comparison_direction` missed an inverted direction | **Metric fixed**, v1→v2, runs re-scored |
| `arguments_valid` inferred from prose | **Fixed**, v1→v2, `ToolErrorKind` added |
| 1% tolerance credits a $27M error | **Documented, NOT changed** — changing it after seeing the score is the manipulation this project exists to refuse |
| `wrong_number` label on yes/no questions | Documented; no score affected |
| PDF extraction strips spaces from line items | Documented as a corpus limitation |
