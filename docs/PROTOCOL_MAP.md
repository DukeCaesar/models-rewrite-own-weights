# Paper-to-code map

This repository implements the procedures described in arXiv:2611.21885v2.
It is an independent implementation, not the authors' unreleased source code.
Numerical replication is a separate validation step and has not been completed.

| Paper procedure | Implementation | Verification |
| --- | --- | --- |
| Immutable baseline, clean editor, temporary edited inference | `checkpoint.py`, `backend.py`, `vllm_backend.py` | Actual tiny transformer; Qwen3.5-2B GPU edit/restoration smoke; vLLM contract double only |
| Session loop, budgets, diagnosis, context reset, final submission | `session.py`, `config.py` | Scripted-editor integration; reduced-budget real Qwen editor smoke |
| Persistent CPU Python worker and SaveEdit restrictions | `worker.py`, `wire.py`, `isolation.py`, `edits.py` | Direct Linux syscall denial, typed IPC, immutable baseline and bf16 scalar-rescaling tests |
| Generate/read forward passes, paired seeds, stored artifacts | `forward.py`, `backend.py` | Tiny-model generation and hidden reads; real Qwen hidden/vision smoke |
| Thinking-aware correctness, truncation, looping, paired uncertainty | `evaluation.py`, `vision.py` | Math-Verify, known paired outcomes, image reference fixtures |
| Practice/held-out preparation and benchmark exclusions | `data.py`, `adapters.py`, `overlap.py` | Deterministic splits, original source imports, explicit lexical/manual overlap audit |
| Repair corruption, reported clipping/projection/time-bias edits | `edits.py`, `replay.py` | Mathematical invariants and named tensor-selection tests |
| Nine first-protocol and eight second-protocol sessions | `experiments.py`, `configs/paper-sessions.json` | Schedule/independence and submission-history tests |
| Reserved practice check and immutable candidate comparisons | `session.py`, `forward.py` | Real tiny-model orchestration and reserved-answer isolation |
| Held-out comparisons and shared baseline samples | `heldout.py`, `baseline.py` | Atomic resume, changed-plan rejection, exact baseline reuse |
| E.9 screening, case retests, confirmation, optional second look | `staged.py` | Controlled outcomes exercise gates, both looks, and resume |
| Six image benchmarks with fixed sample selection | `adapters.py`, `vision.py`, `heldout.py` | Source-specific conversion and grading tests; deterministic selection and image hashes |

## Explicit reconstruction choices

The paper does not provide its complete source, exact split IDs, image draw seed,
original damage RNG seed, seed-hash implementation, complete model trajectories,
or full image graders. This code records its choices rather than treating them
as the original values. Prompts and tool interfaces are reconstructed from the
published protocol. Measured direction vectors must be supplied or estimated
from practice data; a reported edit replay does not demonstrate rediscovery.

Transformers is the verified execution backend. It generates serially and its
sampling stream and throughput differ from the paper's vLLM setup. The optional
vLLM implementation needs a compatible runtime and hardware integration check.

## Work needed for numerical replication

Obtain authorized GPQA-Diamond data, finish semantic overlap review, prepare the
recorded splits, and run the scheduled sessions with the intended checkpoints.
Then precommit and execute the appropriate held-out evaluations. The available
16 GB GPU cannot hold the unquantized 27B checkpoint, and no paper-budget session
or published accuracy/significance result has been reproduced here. All stored
smoke reports identify their reduced scope.
