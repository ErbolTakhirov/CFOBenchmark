# Benchmark source map

Every fact below was obtained by **cloning the official repository and reading its code and data**
in this session — not from the papers, and not from memory. Reference clones live in
`/tmp/financebench-references/` and are deliberately **not vendored** into this project.

## FinQA

| | |
|---|---|
| Source | `https://github.com/czyssrs/FinQA` |
| Pinned commit (data) | `0f16e2867befa6840783e58be38c9efb9229d742` |
| Dataset files | `dataset/{train,dev,test}.json` |
| Real counts | train **6,251** · dev **883** · test **1,147** |
| Evaluator | `code/evaluate/evaluate.py` |
| Native metrics | execution accuracy, program accuracy |
| Licence | MIT (code); data derives from FinTabNet → CC BY 4.0 / CDLA-Permissive |
| Public availability | Full gold, all three splits, in-repo |
| Tests | model, `context_given` |
| Contamination risk | **High** — public since 2021, in every major pretraining scrape |
| Implemented now | Adapter, execution accuracy, program accuracy |

There is also a `private_test` split used for the leaderboard; its gold is **not** public. Not
supported, and not faked.

## TAT-QA

| | |
|---|---|
| Source | `https://github.com/NExTplusplus/TAT-QA` |
| Pinned commit | `870accc41953dcde885aabeb963d94aabdc0fbc3` |
| Dataset files | `dataset_raw/tatqa_dataset_{train,dev,test,test_gold}.json` |
| Real counts | train / dev / test — **`tatqa_dataset_test_gold.json` is present in the repo**, so test gold is public |
| Evaluator | `tatqa_eval.py`, `tatqa_metric.py`, `tatqa_utils.py` |
| Native metrics | Exact Match, numeracy-aware F1, scale accuracy |
| Prediction contract | `{uid: [answer, scale]}` — a JSON dict |
| Licence | MIT (Copyright 2021 Fengbin Zhu) |
| Tests | model, `context_given` (table + paragraphs supplied) |
| Contamination risk | High (public since 2021) |
| Implemented now | Adapter, EM, numeracy F1, scale accuracy, parity suite |

Answer types: `span`, `multi-span`, `arithmetic`, `count`. Scales: `""`, `thousand`, `million`,
`billion`, `percent` — note TAT-QA conflates *scale* and *unit* (`percent` is a "scale"); the
adapter separates them into this platform's `Scale` + `unit`.

## FinanceReasoning

| | |
|---|---|
| Source | `https://github.com/BUPT-Reasoning-Lab/FinanceReasoning` (the bare `BUPT-Reasoning` org is a dead stub) |
| Pinned commit | `b0fe6455396f831955e4eb988472b4a563403bc5` |
| Dataset files | `data/FinanceReasoning/{easy,medium,hard}.json` |
| Real counts | easy **1,000** · medium **1,000** · hard **238** → **2,238** |
| Evaluator | `evaluation.py`, `utils/evaluation_utils.py` (itself adapted from Yale-NLP's FinanceMath) |
| Native metric | numeric accuracy at **0.2 % relative tolerance** (`eps = abs(gt) * 0.002`) |
| Fields | `question`, `context` (a JSON-encoded dataframe), `python_solution`, `ground_truth`, `difficulty` (float), `source`, `question_id`, `level` |
| Licence | ⚠️ **No LICENSE file, and no licence statement in the README** |
| Tests | model, `context_given`; supports CoT and PoT |
| Contamination risk | Moderate (2025 release) |
| Implemented now | Adapter (runtime download only), numeric accuracy, per-level slices |

**Licence blocker.** With no declared licence, the default is "all rights reserved". This project
therefore **downloads the data at runtime and never redistributes it** — no vendored fixtures beyond
short quotations used for parity testing. Recorded in `docs/licenses.md`.

## FinanceBench

| | |
|---|---|
| Source | `https://github.com/patronus-ai/financebench` |
| Pinned commit | `cc39aeb4afdf33909ee1412188bf89035950c2eb` |
| Dataset files | `data/financebench_open_source.jsonl`, `data/financebench_document_information.jsonl` |
| Real counts | **150** open-source questions; **368** source PDFs in `pdfs/` (~1.2 GB) |
| Evaluator | ⚠️ **None.** The repo ships only `evaluation_playground.ipynb` |
| Native metric | **Does not exist** |
| Fields | `financebench_id`, `company`, `doc_name`, `question_type` (`domain-relevant` / `novel-generated` / `metrics-generated`), `question_reasoning`, `question`, `answer`, `justification`, `evidence[{evidence_text, doc_name, evidence_page_num}]` |
| Licence | CC BY-NC-4.0 (declared on HuggingFace; **no LICENSE file in the repo**) |
| Tests | model **and** retrieval system — this is the one benchmark where the distinction bites |
| Full dataset | 10,231 questions, **gated** behind a direct request to Patronus AI → `user_supplied_required`, never claimed as supported |
| Implemented now | `supported_public_subset` (150), `context_given` + `retrieval_required` |

**Because FinanceBench ships no evaluator, every metric this project reports on it is *ours*.** They
are named accordingly (`financebench_*`) and are never described as "official".

## ConvFinQA

| | |
|---|---|
| Source | `https://github.com/czyssrs/ConvFinQA` |
| Pinned commit | `cf3eed2d5984960bf06bb8145bcea5e80b0222a6` |
| Dataset files | `data.zip` → `data/{train,dev,test_private}.json` (+ `*_turn.json` flattened variants) |
| Real counts | train **3,037** conversations · dev **421** conversations = **1,490** turns · test **434** conversations, **gold withheld** |
| Evaluator | `code/finqanet_generator/utils.py` — shares FinQA's `eval_program` / `equal_program` |
| Native metrics | execution accuracy, program accuracy |
| Licence | MIT (Copyright 2022 Zhiyu Chen) |
| Tests | model, multi-turn `context_given` |
| Implemented now | dev split; turn-level and conversation-level |

Record shape: `annotation.dialogue_break` (the per-turn questions), `annotation.turn_program` (the
per-turn programs — **cumulative and self-contained**, e.g. turn 5 is
`subtract(60.94, 25.14), divide(#0, 25.14)` where `#0` refers to *its own* first step, not to a
previous turn), `annotation.exe_ans_list` (per-turn gold answers).

Test gold was never publicly released (CodaLab submission only) → the test split is
`user_supplied_required` and is not evaluated here.

## SECQUE

| | |
|---|---|
| Source | **HuggingFace `nogabenyoash/SecQue`** |
| Not the source | `https://github.com/EnvCommons/SECQUE` — verified this session: it contains `server.py`, `Dockerfile` and sample agents. It is an **agent-harness / OpenReward environment**, and ships **no dataset** |
| Real count | **565** questions |
| Fields | `QID`, `Question`, `ground_truth_answer` (long expert prose), `question_type`, `page_number`, `accession_number`, `item`, `context_markdown_with_headers` / `_without_headers`, `context_html_*` |
| Evaluator | None public; the paper uses an LLM judge |
| Licence | ⚠️ No licence file |
| Tests | model, `context_given` (SEC filing context **is** supplied in the record) |
| Implemented now | Adapter + deterministic subset + optional local LLM judge, results marked **provisional** |

Because gold answers are multi-sentence expert analyses, exact match is meaningless. Only the
subset whose answers contain an extractable ratio/number can be scored deterministically; the rest
require a judge, and judged numbers are labelled provisional and never fold into a gate.

## SMB-CFO (this project's own benchmark)

Not third-party. Deterministic synthetic businesses; **gold answers computed by Python oracles,
never by an LLM**. It is the platform's **only** source of Russian-language coverage — no
third-party financial benchmark found in this research tests Russian at all (every "bilingual"
claim in the literature is English/Chinese).

## Explicitly not implemented

`FinBen` (42+ separately-licensed HF datasets, not one bundle — a taxonomy reference only),
`SEC-QA` (**no public artifact exists anywhere**; Kensho/S&P keep it commercial — recorded as
`unavailable`, not wrapped in a fake adapter), `XFinBench`, `FinMME`, `FinTextQA` (only 19 % is
legally redistributable), `BizFinBench`, `FinMTM`, `FinToolBench` (too immature: 7 commits,
training scripts withheld).

Five working adapters beat fifteen stubs.
