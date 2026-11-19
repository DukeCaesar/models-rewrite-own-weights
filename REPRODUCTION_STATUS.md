# Reproduction status

Target: https://arxiv.org/html/2611.21885 (v2, 18 November 2026).
Independent implementation; no author source repository linked from the paper.

## Implemented
- CPU direction projection, mean-difference direction, block damage, clipping.
- Validated bf16 edit storage with baseline immutability and protected tensors.
- Thinking-aware grading interface, looping detection, paired statistics,
  problem-level permutation test, stable paired seeds.

- Lazy checkpoint reader, standalone edited checkpoint export, experiment presets.
- Transformers generation, clean/damaged/edited weight switching with restoration,
  hidden-state, intermediate MLP and token-log-probability reads.
- Validated forward request execution, paired seed reuse, answer-count limits,
  JSONL generation logs and tensor read artifacts.

## Verified
- `.venv/bin/python -m pytest -q`: 47 passing tests (CPU).
- Integration tests use an actual randomly initialized tiny Llama transformer,
  not a simulated generation backend. They do not validate Qwen accuracy.
- Actual Math-Verify checks: fractional equivalence succeeds; unequal answers fail.
- Installed Transformers 5.23.0, torch 2.8.0+cu129, safetensors 0.8.0,
  Math-Verify 0.9.0 in a workspace virtual environment using system torch.
- Available GPU: RTX 4060 Ti, 16 GB. Unquantized 27B weights alone exceed this.

## Newly implemented
- Persistent numerical CPU worker with host-mediated weights/artifacts, code policy,
  timeouts, bounded output, notes, and process restart. Linux seccomp sealing is now implemented.
- Model-driven session controller, diagnosis, action parsing, budget accounting,
  context reset, event logs, checkpoint/resume, and CLI for protocol 1.
- Protocol 2 CLI, reserved-practice decision flow, immutable candidate comparisons,
  and fail-closed Linux seccomp sealing with data-only worker messages.
- Integration test executes a scripted editor session with real transformer forward
  passes and worker-produced edits. It is an orchestration test, not a model result.

## Required for numerical replication
- Run paper-budget Qwen experiments; the full 17-session schedule and independent-job driver are implemented.
- Obtain authorized GPQA-Diamond data and audit semantic overlaps; six other exclusion sources are downloaded and converted.
- Obtain and audit original benchmark exports, complete evaluation orchestration across all experiments, and run actual benchmarks.
- Complete paper-budget evaluations. Reduced-budget real Qwen model-driven smoke verification is already recorded below.

## Known limits on exact replication
The article does not supply exact practice/held-out IDs, original random damage
seed, hash function, complete prompts or trajectories. These must be recorded
as reconstruction choices; statistical equality with published results cannot
be promised. Reported model-written edits can be replayed separately from newly
sampled self-rewriting sessions. A replay is not evidence that a session discovers
an edit. Full benchmark results have not been reproduced.

## Execution fidelity
The default backend uses verified serial Transformers generation. An optional
vLLM adapter now implements batched generation/editor replies and transactional
weight loading; real vLLM runtime execution remains unverified. Its hidden reads
currently use a lazy CPU Transformers copy. See docs/VLLM.md for fidelity limits. The forward
interface supports original/edited/both and second-protocol saved candidates. A real Qwen editor smoke session completed under explicitly reduced limits, with no edit; no paper-budget editor session has completed.

## Data and evaluation implementation
- Canonical JSONL conversion, seeded 3000/1000 split reconstruction, 300 reserved
  practice items, provenance hashes, explicit normalized exact overlap checks,
  manual exclusion support, and confirmatory positions 500–999 filtering.
- Held-out CLI precommits model/data/settings hashes, resumes complete paired
  artifacts, and analyzes only after all requested answers finish.
- Reported repair replay with named/early ablations, measured-direction recipe,
  and protocol-2 session-8 time-step bias replay. Figure 12's MLP list skips layer 4.
- Tests cover changed-plan rejection, changed-edit rejection, paired resume,
  deterministic data preparation, and replay tensor selection.
- Full benchmark decontamination, published split matching, and numerical
  accuracy reproduction remain unverified.

## Protocol 2 verification
- Direct OS-isolation probe (without AST policy) verifies denied filesystem opens,
  socket creation and subprocess execution. No pickle decoding of worker messages.
- Candidate snapshots, shared baseline seeds, total answer accounting, restoration
  of active edits, and inaccessible reserved answers covered by integration tests.
- Scripted-editor second-protocol session exercises real tiny-transformer forward
  calls, one reserved check, and submission. This does not reproduce model discovery.
- Residual isolation limit: no claim of protection against all native/kernel exploits.

## Actual Qwen execution and images
- Official Qwen3.5-2B downloaded at revision
  15852e8c16360a2fea060d615a32b45270f8a8fc; provenance stored with checkpoint.
- `scripts/smoke_qwen.py` completed on GPU: text generation, image generation,
  post-layer reads at layers 0/3, edit application, and exact clean-weight restoration.
- Report: `runs/qwen2b-smoke/report.json`; peak allocated GPU memory 4,477,304,320 bytes.
  Both answers hit the 128-token cap. This is execution evidence, not accuracy reproduction.
- Actual execution exposed auxiliary MTP tensors in layer helpers; fixed selection
  and added regression coverage. MTP tensors are excluded from replay selection too.
- Six-benchmark canonical image manifests, deterministic sample selection, image
  hashes, explicit graders, and pooled/per-benchmark evaluation reports implemented.
- All six original image exports are converted and validated. Original image
  grading and split equivalence remain unverified.

## Latest verification
- Real Qwen editor smoke: `runs/qwen2b-editor-smoke/smoke_report.json`.
  Completed three replies (two allowed action turns plus final reply), no valid
  tool actions, no saved edits. Reduced limits and synthetic practice only.
- Added 17-session experiment plan/driver and protocol-2 replacement history.
- Resumes validate source model/data hashes and reconcile committed forward logs;
  interrupted actions are recorded rather than silently rerun.
- Qwen template thinking markers are preserved to prevent truncated draft code
  execution; end-of-message tokens cannot invalidate SUBMIT decisions.
- Tokenizer encode/decode is available through the sealed worker capability API.
- Original-format JSON/CSV/Parquet text/image adapters and provenance manifests
  are tested with representative fixtures, including embedded image bytes.
- Optional datasets installation failed with DNS resolution errors; current
  local adapters use installed PyArrow/Pillow. Subsequent source downloads are recorded below.

## vLLM serving implementation
- Explicit backend factory and CLI selection; resume identity includes backend.
- Batched requests with explicit per-sample seeds, local safe-tensor overlays,
  clean restoration on failures, and cache reset after every load.
- Contract test uses an engine double; it does NOT verify actual vLLM/Qwen execution.
- Package lookup found no compatible available vLLM distribution in this environment.
  No runtime was installed; a separate `scripts/smoke_vllm.py` awaits execution.

## Staged evaluation and shared baselines
- Shared baseline-answer cache keyed by checkpoint/data/sampling identity, with
  checksums and file locks; edited comparisons reuse the same baseline samples.
- Second-protocol screening on two seed pools, success/reverse case selection,
  16-seed case retests, confirmation, and predeclared session-8 second-look rule.
- Rules and seed pools persist before generation; changed plans are rejected.
- Controlled tests cover failed gates, 16-sample pooling, and no-resampling resume;
  actual tiny-model tests cover baseline reuse across different edits.
- Added explicit reported-overlap mode for the paper's audited repeated-question
  comparisons. Practice IDs cannot recur; default remains strict; confirmatory
  subsets remain disjoint. Original semantic-overlap audits are unavailable.
- No staged paper-model benchmark has yet been run.

## Actual source audit and constraint fixes
- Downloaded pinned MATH-500, AIME 2024/2025, HMMT February 2025, AMC source,
  and all 18 OlympiadBench exports; file hashes and source metadata recorded.
- Converted exclusion rows: 500, 30, 30, 30, 40 (AMC 2023 only), and 8,476.
- Actual AMC source contains 2022 and 2023; importer now explicitly selects 2023.
- DeepScaleR has six empty reference answers. Conversion records those source
  rows as rejected; 40,309 usable rows remain. Earlier unfiltered copy retained.
- Exact normalized overlap audit found three MATH-500 matches and zero against
  the other five sources. This is NOT a completed semantic-decontamination audit.
- GPQA-Diamond download returned HTTP 401; authorized local export/access requested.
  No complete exclusion split has been generated without it.
- Fixed verified bf16 global-rescaling bypass: intersect rounding-bin scalar
  intervals rather than relying on a rounded least-squares estimate. Tests cover
  non-power-of-two, negative, and zero scales and legitimate non-scalar edits.
- Exclusion-only rows are labeled explicitly and cannot be used as generation
  references; proof/multiple-answer sources do not need fabricated answers.

## Image-source validation and overlap review
- All six image-source downloads completed at recorded revisions with hashes.
  Conversion and full canonical-record validation are complete.
- Source inspection verified ScreenSpot-v2 pixel xywh boxes. Import now requires
  explicit coordinate-space and box-format flags and converts to normalized xyxy.
- MathVista pid 781 repeats correct value 18 in two options. Import retains both
  acceptable labels instead of rejecting or arbitrarily relabeling the record.
- Numeric precision/unit instructions are preserved for MathVista.
- Six-source lexical overlap audit completed: 206 candidate pairs (148 MATH-500,
  58 OlympiadBench), with 15 explicitly reviewed so far. Candidate absence and
  cosine thresholds are not semantic-cleanliness guarantees.
- GPQA authorization is still pending; no complete decontaminated split claimed.

## Final implementation verification
- 47 tests pass, including real tiny-transformer integration and Linux isolation.
- The installed CLI exposes session, repair/export/replay, dataset preparation,
  overlap review, paired/staged evaluation, and the full 17-session schedule.
- `configs/verified-environment.json` records the library versions used for checks.
- Canonical image counts: MathVista 1,000; MathVerse vision-only 788; MMStar 1,500;
  HallusionBench 951; POPE 9,000; ScreenSpot-v2 1,272.
- POPE retains all questions and stores its 500 distinct images once, with exact
  decoded-pixel hashing. The repeatable importer verifies downloaded file hashes.
- MathVerse supports value-valued references and all-required multi-choice sets.
  ScreenSpot's single one-pixel out-of-bounds annotation is explicitly recorded.
- `data/image-evaluation/images.jsonl` contains the fixed 600-question selection
  (100 per benchmark from each seeded 300-item draw), with image hashes and IDs
  in its manifest. These images have not been run as an accuracy benchmark.
- Implementation coverage and reconstruction choices: `docs/PROTOCOL_MAP.md`.
