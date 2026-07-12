# BLOCKERS — v0.1.0-rc1

This release candidate **was not tagged**. Every gate below must pass first.

Note what is *not* here: ruff, mypy, the 1,056 primary tests, the 411 security tests, and the 17 parity tests (zero skips) all **pass**. The code is healthy. What is missing is *evidence* — runs that have not finished. Tagging on a green test suite while the headline experiment is still executing is precisely the dishonesty this project exists to prevent.

## one evaluator fingerprint across all runs

```
6 distinct: ['5135fb2b8b97a8d8', '5e86dc70b21b2966', '80ca8a678b1c4fa1', 'e3af8137742105dc', 'ec3447250b3cd087
```

**What clears it**

Re-score the stale runs onto the current fingerprint:
`financebench resume --run-id <id> --model-config <cfg>` replays cached responses and costs nothing.

**Except for tool_assisted runs.** Those are deliberately uncached (an agent run is a conversation, not one hashable request), so re-scoring one means re-running every turn against the model.

## paired direct-vs-tools run complete

```
missing: ['7B direct', '7B tools']
```

**What clears it**

Run the missing variants against the frozen manifest:
`financebench eval --manifest configs/manifests/tool_paired_v1.json --model-config <cfg> --mode {context_given|tool_assisted}`

All four must exist, on the SAME 150 sample ids, or the paired comparison does not exist. Do not substitute an unrelated direct run — the previous tool run was on `tatqa:train:` ids while both direct runs used `tatqa:dev:`, so it could not be paired with anything at all.

## release-group run complete (both models)

```
0 of 2 present
```

**What clears it**

`financebench eval --manifest configs/manifests/release_v0_1.json --model-config <cfg>`

This is the **only** run that can produce a Finance Capability Index: the index is withheld unless ONE run covered SMB-CFO *and* a grounding benchmark *and* refusal together. Without it, the report's headline is `INSUFFICIENT_COVERAGE`.

## retrieval ablation complete

```
18 cells, 1 generated arm(s)
```

**What clears it**

Retrieval metrics for all 18 cells are cheap (no model runs):
`financebench retrieval-eval --retrievers bm25,dense,hybrid --top-k 1,3,5,10,20`

The **generated** arms are not cheap — ~4.6 GPU-hours each at 109.5 s/sample. At least two are needed to say anything about whether better retrieval produces better answers.

