# Validity threats

Reasons a number produced by this platform might not mean what it appears to mean. Each threat
names the mitigation actually implemented, or states plainly that none exists yet.

## 1. Gold-answer leakage into the prompt — *was real, now structurally impossible*

**Threat.** If the answer key reaches the model, the benchmark measures nothing.

**It was real here.** `ModelRequest.simulation_context` carried `gold_answer` and
`gold_numeric_value`. That object was serialized into `predictions.jsonl` *and hashed into the
response-cache key*. Only a `provider == "mock"` branch kept it empty for real providers — a
convention, not a guarantee, and one line away from a silent catastrophe.

**Mitigation.** The field is **deleted**. `ModelRequest` has no channel through which gold can
travel; the mock receives its oracle by constructor injection instead. This is a structural
guarantee, not a check that can be forgotten.

`tests/security/test_gold_answer_leakage.py` proves it two ways:

1. **Structural** — no field on `ModelRequest` can carry gold.
2. **Scrub-equivalence** — take a real sample, replace its entire `gold`, its `evaluation`
   (tolerances), and its `acceptable_answers` with sentinel values, re-render the request, and
   assert it is **byte-identical** to the original. *If the prompt cannot change when the answer
   changes, the answer cannot be in the prompt.* Run across every benchmark × prompt profile ×
   eval mode.

**Deliberate, documented exception.** ConvFinQA's official protocol feeds *prior* turns' gold
answers as conversation history. That is permitted only under an explicit
`conversation_context_policy = gold_history` flag, only for turns strictly before the current one,
and never for the current turn. The default is `model_history` — the model's *own* prior answers —
because that is what actually exposes error propagation instead of hiding it.

## 2. Mock results masquerading as model results

**Threat.** The mock reads the answer key. A mock run scoring 100 % says nothing about any model,
yet writes byte-identical artifacts to a real run.

**Mitigation.** `--allow-mock` is required or the run fails; `run_type = "mock_test"`;
`eligible_for_leaderboard = false`; a `MOCK — NOT A MODEL RESULT` watermark in every report; excluded
from the leaderboard and from the Finance Capability Index. Mock runs cannot satisfy any
live-verification claim in this repository.

## 3. Benchmark contamination

**Threat.** FinQA (2021), TAT-QA (2021) and ConvFinQA (2022) predate every current model's training
cutoff and are in the standard scrapes. A high score may be recall, not reasoning.

**Mitigation — partial, and honestly so.** There is no way to decontaminate a public benchmark.
What is done instead:

- Contamination risk is recorded per benchmark in `benchmark_source_map.md` and surfaced in reports.
- **SMB-CFO** exists precisely as the uncontaminated counterweight: freshly generated, deterministic,
  with a **secret-seeded private variant** whose seed is never committed (only its hash is recorded
  in run metadata). A large gap between a model's public-benchmark score and its SMB-CFO score is
  itself the contamination signal.

**Residual risk: high.** Do not read a strong FinQA score as evidence of reasoning ability.

## 4. Measuring prompt-format compliance instead of financial ability

**Threat.** A model that computes the right number but emits malformed JSON scores identically to
one that computes the wrong number. That is a benchmark measuring formatting.

**Mitigation.** Answer correctness, format compliance, evidence correctness and explanation quality
are scored as **separate** dimensions. The numeric parser recovers a number from messy prose so a
formatting slip does not masquerade as a reasoning failure — and `invalid_output_rate` is tracked
separately as its own gate, so formatting failure is *visible* rather than *conflated*.

## 5. Conflating model ability, retrieval ability, and agent ability

**Threat.** A single "financial score" mixing a RAG pipeline's retrieval quality with the model's
reasoning tells you nothing about either.

**Mitigation.** Three named evaluation modes (`context_given`, `retrieval_required`,
`tool_assisted`), recorded in the run config, folded into the run id and the cache key, and reported
as **separate top-level scores** (Financial Core / RAG / Agent). They are never averaged into one
number without the mode-level breakdown alongside. On FinanceBench, retrieval-miss and
generation-error-after-successful-retrieval are attributed separately, so a failure can be blamed on
the right component.

## 6. Incomparable runs compared anyway

**Threat.** Two runs on "finqa test" are not comparable if they used different prompt profiles,
different eval modes, different sample subsets, or different evaluator versions.

**Mitigation.** A comparability fingerprint — `(benchmark_version, split_fingerprint,
prompt_profile, eval_mode, evaluator_version, sample_id_set_hash)`. `compare` and `leaderboard`
refuse to aggregate across differing fingerprints, or mark the comparison explicitly non-comparable.

## 7. Small-sample noise presented as a ranking

**Threat.** 40 samples on a 4 GB GPU is a small sample. The difference between two models at 45 %
and 50 % on 40 samples is very likely noise.

**Mitigation.** Bootstrap 95 % confidence intervals on every reported score; **paired** comparison on
identical sample IDs (not independent two-sample tests); minimum-sample warnings; per-category CIs.
No claim that one model is better is made when the intervals and the paired analysis do not support
it.

**Residual risk: significant.** Live runs in this environment are deliberately small. Every headline
number carries its interval, and the coverage statement says exactly how many samples it rests on.

## 8. LLM-as-judge non-determinism

**Threat.** SECQUE's answers are multi-sentence expert analyses. Grading them requires a judge, and a
judge is itself a model with its own biases and its own failure modes. Binary "correct/incorrect"
judging of nuanced analytical answers is known to be unreliable.

**Mitigation.** Judged results are marked **provisional**, are excluded from critical gates, and
record the judge identity, judge prompt version, judge cost, and inter-judge disagreement when more
than one judge is configured. The deterministic subset (answers containing an extractable ratio) is
scored without a judge and reported separately.

## 9. The tested model is small and local

**Threat.** `qwen2.5:7b` on a 4 GB GPU is not a frontier model. Its scores say little about what
GPT-5-class models can do.

**Mitigation.** None needed — this is a *scope* limit, not a validity flaw, and it is stated plainly.
The purpose of the live run is to prove the benchmark **correctly evaluates real model output**, not
to rank the frontier. The model is allowed to score badly, and evaluator rules are never tuned to
flatter it.

## 10. Benchmark success ≠ deployment safety

**Threat.** The most dangerous possible misreading of this repository.

**Mitigation.** The finance-readiness verdict tops out at
`EXCEPTIONAL_BUT_STILL_REQUIRES_CONTROLS`. There is no "safe for autonomous financial decisions"
label, by design. Critical gates can override a strong average — a model with a good mean score but
a meaningful catastrophic-numeric-error rate cannot receive a strong readiness label. A benchmark
measures a benchmark.
