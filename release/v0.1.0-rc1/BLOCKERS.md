# BLOCKERS — v0.1.0-rc1

This release candidate **was not tagged**. Every gate below must pass first.

Note what is *not* here: ruff, mypy, the 1,056 primary tests, the 411 security tests, and the 17 parity tests (zero skips) all **pass**. The code is healthy. What is missing is *evidence* — runs that have not finished. Tagging on a green test suite while the headline experiment is still executing is precisely the dishonesty this project exists to prevent.

## release built from a clean commit

```
dirty tree — this release cannot be reproduced from its commit
```

## release-group run complete (both models)

```
1 of 2 present
```

**What clears it**

`financebench eval --manifest configs/manifests/release_v0_1.json --model-config <cfg>`

This is the **only** run that can produce a Finance Capability Index: the index is withheld unless ONE run covered SMB-CFO *and* a grounding benchmark *and* refusal together. Without it, the report's headline is `INSUFFICIENT_COVERAGE`.

## retrieval ablation complete

```
18 cells, 0 generated arm(s)
```

**What clears it**

Retrieval metrics for all 18 cells are cheap (no model runs):
`financebench retrieval-eval --retrievers bm25,dense,hybrid --top-k 1,3,5,10,20`

The **generated** arms are not cheap — ~4.6 GPU-hours each at 109.5 s/sample. At least two are needed to say anything about whether better retrieval produces better answers.

