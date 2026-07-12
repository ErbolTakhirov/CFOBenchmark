# Local models

**Ollama is the only provider in this repository that has ever produced a real number.** Every result
in `runs/` came from it.

## Ollama — live-verified

```bash
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
financebench eval --benchmark finqa --split test --max-samples 40 \
  --model-config configs/models/ollama-qwen2.5-3b.yaml
```

Shipped configs: `configs/models/ollama-qwen2.5-3b.yaml`, `configs/models/ollama-qwen2.5-7b.yaml`.

**Their runtime settings are matched on purpose**, and this is not cosmetic. `concurrency` and
`timeout_seconds` are *not* part of the run id, the cache key, or the evaluator fingerprint — so none
of the machinery that guards comparability will catch a difference in them. They still move the
results: the first SECQUE 3B run used a 180 s timeout at concurrency 4 and recorded three
`ProviderTimeoutError`s, while the 7B ran at 300 s / concurrency 2 and recorded none. Those three
timeouts were then scored as the 3B getting three financial questions wrong.

## Know what your hardware is doing

On the machine these results came from — a **GTX 1650, 4 GB** — `qwen2.5:7b` is 4.7 GB of weights and
**spills to CPU**. Measured, from real run artifacts: TAT-QA 16.8 s/sample (3B) vs 29.3 s (7B); FinQA
35.8 s vs 70.7 s.

**Those ratios say as much about the 4 GB as about the models.** Every latency comparison in this repo
is labelled as a measurement of *this machine*, never as a general claim about 3B-vs-7B inference cost.
Correctness comparisons are unaffected.

Concurrency does not help a model that is already spilling to CPU. It makes it time out.

## vLLM / llama.cpp / LM Studio — implemented, unreachable

Any OpenAI-compatible server works through the `openai_compatible` provider:

```yaml
provider: openai_compatible
model: qwen2.5-7b-instruct
base_url: http://localhost:8000/v1
```

It is **implemented and unit-tested, but has never been run against a live server here** — no local
server was running when it was last probed, so `financebench verify-providers` labels it
`unreachable`. It is not claimed as working.

The base install and the test suite require **neither a GPU nor any of these backends**. See the
`local-transformers` and `vllm` extras in `pyproject.toml`, and [`providers.md`](providers.md) for what
is live-verified versus what merely compiles.
