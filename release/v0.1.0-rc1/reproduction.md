# Reproducing v0.1.0-rc1

Everything needed is in `release_manifest.json`: the commit, the evaluator fingerprint, the frozen
sample ids, the model digests and quantization, the runtime versions, the seeds, and the hardware.

## 1. The code

```bash
git clone https://github.com/ErbolTakhirov/CFOBenchmark.git
cd CFOBenchmark
git checkout <repository_commit from release_manifest.json>
python -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
```

Confirm you have the same evaluator:

```bash
./.venv/bin/python -c "from financebench.evaluation.fingerprint import current_fingerprint; print(current_fingerprint().digest)"
```

**If this digest differs from `evaluator_fingerprint.digest` in the manifest, stop.** Your numbers
will not be comparable to the published ones, and no amount of matching the model will fix that.
Fixing the answer parser once moved a FinQA score from 5% to 15% **on identical cached model
responses**.

## 2. The data

```bash
financebench prepare finqa && financebench prepare tatqa && financebench prepare finance_reasoning
financebench prepare financebench && financebench prepare convfinqa && financebench prepare secque
financebench validate-dataset --all-core
```

Each adapter is pinned to an upstream commit (`evaluator_fingerprint.dataset_adapters`) and verifies a
sha256 on download. SMB-CFO is generated in-repo from Python oracles.

## 3. The model

```bash
ollama pull qwen2.5:3b        # digest + quantization are in release_manifest.json -> models
ollama pull qwen2.5:7b
ollama show qwen2.5:3b        # confirm quantization == Q4_K_M
```

**Quantization matters.** A `Q8_0` build of the same model is a different model for these purposes.

## 4. The runs

Every run in the release used a **frozen sample manifest** — never `--max-samples`.

```bash
# The paired direct-vs-tools experiment (150 identical sample ids, four ways)
financebench eval --manifest configs/manifests/tool_paired_v1.json \
  --model-config configs/models/ollama-qwen2.5-3b.yaml --mode context_given
financebench eval --manifest configs/manifests/tool_paired_v1.json \
  --model-config configs/models/ollama-qwen2.5-3b.yaml --mode tool_assisted
financebench eval --manifest configs/manifests/tool_paired_v1.json \
  --model-config configs/models/ollama-qwen2.5-7b.yaml --mode context_given
financebench eval --manifest configs/manifests/tool_paired_v1.json \
  --model-config configs/models/ollama-qwen2.5-7b.yaml --mode tool_assisted

# The release group (the only run that can produce an FCI: it covers SMB-CFO + grounding + refusal)
financebench eval --manifest configs/manifests/release_v0_1.json \
  --model-config configs/models/ollama-qwen2.5-3b.yaml
financebench eval --manifest configs/manifests/release_v0_1.json \
  --model-config configs/models/ollama-qwen2.5-7b.yaml

# The retrieval ablation (no model in the loop — cheap)
financebench retrieval-eval --retrievers bm25,dense,hybrid --top-k 1,3,5,10,20
```

The manifest's `id_hash` is printed at the start of every run. **If it does not match the manifest in
the release, you are asking different questions.**

## 5. What will NOT reproduce exactly

Be honest about this rather than promising bit-identity:

- **Latency.** The published numbers are from a **GTX 1650 with 4 GB VRAM**, on which qwen2.5:7b
  spills to CPU. Your latencies will differ, possibly by a lot, and that says nothing about the models.
- **Ollama's sampling.** Temperature is 0 everywhere, but Ollama does not guarantee bit-identical
  output across versions or hardware. Expect small drift in individual answers; the aggregate metrics
  are stable.
- **The dense/hybrid embedding index.** Rebuilding it takes ~2 hours for 11,948 pages. It is
  checkpointed and resumable, and its corpus fingerprint (`75271c35298bac37`) is recorded — if yours
  differs, your corpus differs.

## 6. Verifying you got the same answer

```bash
financebench compare --run-id <yours> --run-id <ours> --metric financebench_answer_accuracy
```

`compare` refuses to compare runs with different evaluator fingerprints, and reports a **paired**
bootstrap CI over the samples both runs answered — not a difference of means, which can manufacture a
difference out of nothing.

## 7. The checksums

```bash
cd release/v0.1.0-rc1 && sha256sum -c checksums.txt
```
