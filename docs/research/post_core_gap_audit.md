# Post-core gap audit

*What this platform could and could not honestly claim once the numerical core was frozen — and what
has been built since.*

The core (FinQA, TAT-QA, FinanceReasoning) is real, parity-tested against the upstream evaluators,
and live-verified on local models. That was never in question. What was in question is what it
**licenses you to say**, and the answer at the freeze point was: *far less than the name of the
project implies.*

This document exists because a benchmark platform's most dangerous failure is not a wrong number. It
is a **correct number answering a question nobody asked**, presented as though it answered the one
they did.

---

## The five questions

| # | Question | At core freeze | Now |
|---|---|---|---|
| 1 | Can the model calculate financial answers correctly? | **Yes** — FinQA, TAT-QA, FinanceReasoning, parity-tested | Yes |
| 2 | Can it reason over tables, text, **and conversations**? | Tables and text only. **No conversational benchmark existed.** | Yes — ConvFinQA, both protocols |
| 3 | Can it **retrieve and cite** evidence from real documents? | **No.** Every benchmark handed the model its evidence. | Yes — FinanceBench over 12,013 real PDF pages |
| 4 | Can it give grounded **CFO-style analysis for a small business**? | **No.** Nothing in the platform resembled a small business. | Yes — SMB-CFO, 562 cases, Python oracles |
| 5 | Can it recognise missing information and **avoid dangerous hallucination**? | **Measured, and measured wrongly.** See below. | Yes — and the metric that was inverted is fixed |

Three of the five could not be answered at all. The fourth — hallucination — was worse than
unanswered: it was answered **backwards**, which is the subject of the next section.

---

## What the core could not see

### 1. Every benchmark handed the model its evidence

FinQA, TAT-QA and FinanceReasoning all supply the relevant table and text in the prompt. A model
that scores well on them has demonstrated *reasoning over given evidence*. It has demonstrated
nothing whatsoever about finding that evidence — and in a real filing the evidence is one paragraph
in a 300-page 10-K.

This is not a small omission. A model with perfect FinQA accuracy and no retrieval ability is
**useless** on a real document, and the core could not distinguish it from one that works.

*Now measured.* FinanceBench runs in two modes over a real corpus (84 SEC filings, 12,013 pages,
extracted and content-hashed). The gap between them **is** the retrieval loss, and the failure
attribution separates `retrieval_miss` (the retriever never found the page) from
`generation_error_after_retrieval` (it found the page and the model still got it wrong) — because
those two have opposite fixes, and a single RAG-accuracy number sends you to repair the wrong
component.

The retrieval numbers are unflattering and they are honest: BM25 page recall is **2.7 %** at k=5
over the open corpus, **10.7 %** document-scoped, **27.3 %** at k=20. A hand-rolled BM25 over
12,013 pages of financial prose is simply not a good retriever, and the platform says so rather
than quietly reporting the mode that flatters it.

### 2. There was no small business anywhere in the platform

Every benchmark in the core is built on **public-company filings**. The mission is a CFO tool for
*small businesses*, and an SMB does not have a 10-K. It has a ledger, unpaid invoices, a payroll
date, and a cash balance that has to survive until the customer pays.

A model can be excellent at FinQA and unable to tell you when you run out of money.

*Now measured.* SMB-CFO: 562 cases from seeded synthetic businesses (ledgers, invoices with due
dates, FX, budgets, taxes), graded against **~24 deterministic Python oracles**. No LLM ever writes
a gold answer — the gold is a `Decimal` computed from the books, so the benchmark cannot inherit a
judge's mistakes, and it is the only benchmark here that is **provably uncontaminated** by anyone's
pretraining data.

### 3. Turn 1 of a conversation is *"and what was it in 2005?"*

A single-turn benchmark cannot see whether a model can hold a conversation. And "hold a
conversation" is not one capability but two, which almost every published evaluation conflates:

- can it reason about a follow-up turn, given a correct history? (**gold_history**)
- can it survive *its own* earlier mistake? (**model_history**)

The second is what a user actually experiences, and it is invisible to any evaluation that feeds the
model a gold history — which is the standard setup, because it is the easy one.

*Now measured.* ConvFinQA, 421 dev conversations / 1,490 turns, run under **both** protocols, whose
scores are **never mixed** — averaging them cancels exactly the effect they exist to isolate. The
engine now runs turns sequentially within a conversation and conversations in parallel, because
under `model_history` turn 3's prompt cannot be *built* until turn 2 has answered.

The dependency between turns had to be **recovered from the data**: ConvFinQA's turn programs are
self-contained (turn 4's is `subtract(60.94, 25.14), divide(#0, 25.14)` — it recomputes rather than
referring back), so there is no explicit cross-turn link anywhere in the dataset. But `60.94` and
`25.14` *are* the answers to turns 0 and 1, and that is recoverable. Error propagation is therefore
measured against the turns a turn **actually consumes**, not against mere adjacency — otherwise the
"propagation rate" would just be measuring how hard the conversation was.

---

## The one the core got backwards

The core *did* have a refusal metric. It was inverted, and it took a live run to see it.

qwen2.5:3b was asked what revenue would be in December 2027 — a question a ledger cannot answer. It
replied:

```json
{"data": [], "error": "The provided tables do not contain any entries for December 2027
 or subsequent months to calculate revenue."}
```

That is a **correct refusal**. It is precisely the behaviour the benchmark exists to reward. But the
model expressed it in its own JSON shape instead of setting the `insufficient_information` boolean
the prompt asked for — and the metric only read that flag. So the safest possible answer was recorded
as a **FAILED REFUSAL**: the most severe failure in the taxonomy, the one that means *"invented a
number for a question with no answer."*

The metric was measuring **format compliance** and reporting it as **dangerous hallucination**.

On identical cached responses, with no new inference:

| | before | after |
|---|---|---|
| `smb_cfo_refusal_correctness` | 0.667 | **1.000** |
| failed refusals | 10 / 10 | **0 / 10** |

The model had been right all along. What it actually cannot do is hold the requested output schema
under a ~6k-token ledger context — a *formatting* failure, now recorded as
`invalid_structured_response` and bounded by its own gate, rather than as a reasoning failure.

Two lessons went into the code:

1. Refusal is read from the **substance** of an answer (the flag if set, the text otherwise, in
   English and Russian — a Russian refusal is still a refusal, and an English-only detector would
   have reported the entire RU half of SMB-CFO as hallucinating).
2. *"I cannot determine this, however I estimate roughly 42,000"* is **not** a refusal. That is a
   number with a disclaimer attached, and a reader will use the number. Counting it as a refusal
   would let the most dangerous behaviour of all — confident invention, softly worded — score as the
   safest.

---

## The fingerprint had a hole in it

The evaluator fingerprint exists so a change to *our* code cannot silently move a score and have the
new number sit next to the old one as if they were comparable. It only works if it is complete.

It was not. `financebench` and `smb_cfo` were both missing from `DATASET_ADAPTER_VERSIONS`, and every
one of their metrics was missing from `METRIC_VERSIONS`. Nothing failed. The digest was computed,
written into `environment.json`, and compared between runs — while being **blind to two entire
benchmarks**. Regenerating the SMB-CFO oracles would have moved every SMB-CFO score in the repo and
left the fingerprint identical: the exact failure the fingerprint was built to prevent, wearing the
fingerprint's own badge.

A fingerprint with a hole in it is worse than no fingerprint, because it is *trusted*. The registries
are now the source of truth, and `tests/unit/test_fingerprint_coverage.py` fails if a registered
adapter or metric is not versioned.

---

## What the numbers actually say (qwen2.5:3b, live, `structured_financial_v1`)

Reported because they are real, not because they are good. **No metric rule was changed to improve
any of them.**

**FinanceBench, `context_given`, 150 questions:**

| metric | value | what it means |
|---|---|---|
| `financebench_answer_accuracy` | **0.247** | over the 89 questions with a checkable answer (52 numeric + 37 boolean). The other 61 are analytical: `passed=None`, never a fake zero. |
| `financebench_unsupported_numeric_claim` | **0.727** | 73 % of answers contained only numbers that appear in the evidence. **27 % stated a figure that appears nowhere in it.** |
| `financebench_citation_accuracy` | **0.000** | it never produced a usable citation. Not once in 150. |

The last row is the one to sit with. A model that is right a quarter of the time, invents a figure in
a quarter of its answers, and can never tell you where anything came from is not a tool a CFO can
check. **The verdict is `NOT_FINANCE_READY`, and it is correct.**

**SMB-CFO, adversarial split, 30 cases:**

| metric | value |
|---|---|
| `smb_cfo_injection_resistance` | **1.000** — it obeyed no instruction hidden in its own ledger |
| `smb_cfo_refusal_correctness` | **1.000** — it declined every unanswerable question |
| `smb_cfo_accuracy` | **0.000** — it could not hold the output schema under a 6k-token ledger |

A genuinely interesting profile, and one the broken refusal metric had completely obscured: the model
is **safe** and **unusable**, which is a different problem from being **unsafe**, and it needs a
different fix.

---

## Still not built

Stated plainly, because a gap audit that omits its own gaps is decoration:

- **SECQUE** — not implemented. When it is, the analytical score will be reported as
  `not_evaluated` where no trusted judge is configured, never as a fake zero.
- **Tool-assisted evaluation** — the `tool_assisted` eval mode exists in the schema; the sandbox
  does not.
- **OpenAI / Anthropic / Gemini / OpenRouter providers** — not implemented. When they are, and no
  API key exists on this machine, they will be labelled `implemented_not_live_verified`. A provider
  that has never made a successful call is not a working provider, and will not be described as one.
- **The full matched eval matrix** — qwen2.5:3b has the broadest coverage; qwen2.5:7b has a matched
  subset. Per the mission's own rule, **model count bends before benchmark coverage does.**
- **The self-contained HTML report** — per-run `report.html` exists; the cross-run one does not.
