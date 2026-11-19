# vLLM backend status

The paper serves generation and editor replies with vLLM. An optional adapter now
implements that path. Select it explicitly with `--backend vllm` for `session`,
`evaluate`, or `run-plan`; there is no automatic fallback to Transformers.

The adapter uses the local vLLM source's `LLM.collective_rpc`, model `load_weights`,
and `reset_prefix_cache` interfaces. A compatible vLLM package was not available
through the current environment's package lookup, so **real vLLM execution is
unverified**. Unit tests exercise an engine double, not an installed vLLM engine.
Do not interpret these tests as paper-serving equivalence.

Implementation decisions:

- One local synchronous GPU engine; bf16, no quantization, eager execution,
  prefix caching disabled. No distributed/shared-filesystem assumptions.
- Requests are batched with one explicit sample seed per problem/sample pair.
- A temporary safetensors overlay is loaded through a trusted harness RPC.
  All affected tensors are restored from the clean checkpoint after each request,
  including failed loads. Failed restoration permanently disables that adapter.
- Cache reset follows every weight load, including restoration. Generation and
  editing never share cached states computed under different weights.
- Editor replies run on the restored clean model. Repair experiments load damage
  for every requested original/edited generation.
- Hidden reads currently use a lazy bf16 CPU Transformers copy. This needs RAM
  for an additional model and is slower than a GPU read. The read compute path
  therefore remains a documented implementation difference.
- Default maximum context accommodates the session's context-reset threshold and
  editor reply allowance. A GPU must have sufficient weight/cache capacity; the
  adapter does not silently reduce these limits to fit available hardware.

Use a separate environment with a compatible Qwen3.5-capable vLLM build, then run:

```bash
python scripts/smoke_vllm.py
```

That script verifies generation before/with/after a weight edit using a 4096-token
context on the downloaded 2B checkpoint. Full editor/read/vision validation and
paper-budget runs must follow before claiming the vLLM backend is reproduced.
Do not reuse a saved Transformers session with vLLM: resume checks include the
backend identity. Configuration/source hashes and completed answers remain in
session/evaluation artifacts.
