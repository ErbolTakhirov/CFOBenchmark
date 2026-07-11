# TAT-QA fixture

A curated slice of the **official TAT-QA dev split**, taken verbatim from
`NExTplusplus/TAT-QA` at commit `870accc41953dcde885aabeb963d94aabdc0fbc3`
(`dataset_raw/tatqa_dataset_dev.json`).

Licence: **MIT** (Copyright (c) 2021 Fengbin Zhu) — redistribution is permitted, which is why this
one *can* be vendored (FinanceReasoning's cannot; it ships no licence).

Curated, not sampled: it contains at least one question of every
`(answer_type, scale)` combination present in the real dev split — `span`, `multi-span`,
`arithmetic` and `count` crossed with `""`, `thousand`, `million`, `billion` and `percent`.
A random sample would have missed the rare combinations, and those are exactly where the official
metric's special cases live (`f1 := em` for arithmetic/count; `percent -> x0.01`).

The filename is `dev.json` so this directory can be handed straight to `TatQAAdapter(data_dir=...)`
as a drop-in `data_dir`.
