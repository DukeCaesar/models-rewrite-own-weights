# Evaluation workflows

`evaluate` precommits a single comparison. `evaluate-staged` implements the
second-protocol workflow from Appendix E.9. Neither command selects the best edit
from held-out scores; inputs must be edits already submitted by a session.

```bash
self-rewrite evaluate-staged --checkpoint checkpoints/Qwen3.5-27B \
  --edits runs/paper/protocol2-27B-session2/edited \
  --initial data/splits/initial.jsonl \
  --confirmatory data/splits/confirmatory.jsonl \
  --practice data/splits/practice.jsonl \
  --baseline-cache runs/shared-baseline --output runs/eval-session2
```

Use the same baseline-cache directory across sessions to reuse exact baseline
answers, as the paper does. Keys include base/damage checkpoint identity, full
problem content, image hash if any, sampling settings, sample count and seeds.
They exclude the edited tensor set. File locks prevent competing processes from
independently sampling the same baseline item. Cache checksums detect corruption.
This cache is not a model prefix/KV cache.

Stages fixed in `staged_plan.json` before generation:

1. Two independent one-sample runs on the initial subset (normally 200 items).
   Continue to confirmation only when pooled edited accuracy exceeds baseline.
2. Identify successes/reversals that occur in both screening runs; retest each
   selected problem on 16 new paired samples. Report every case, not only gains.
3. On a disjoint confirmatory subset, generate eight paired samples per problem
   and use a one-sided problem-level sign-flip test with 100,000 draws.
4. If `--two-looks` was fixed before generation, use alpha 0.0294 at the first
   look. If it fails, add eight samples and test all sixteen at alpha 0.0294.
   This reproduces session 8's predeclared optional second look. Other sessions
   use one look at alpha 0.05. Changing the look rule on resume is rejected.

All seed bases are explicit in the plan; they are reconstruction choices because
the original values were not published. Completed paired artifacts are resumed,
not regenerated. Component reports are files for audit; do not inspect them to
change a running plan. Final reporting occurs after all applicable stages finish.

The original first-200 evaluation included some questions repeated in practice.
Default execution refuses these. To reproduce that *audited* comparison without
silently replacing questions, use `--report-initial-overlap` for staged evaluation
or `--report-practice-overlap` for a single comparison. Repeated normalized
question IDs are recorded in the component plans. Reused practice identifiers
are always rejected. The confirmatory subset must remain disjoint, and these
options do not detect semantic paraphrases. A separate manually audited clean
subset is needed for the paper's no-repeat sensitivity analysis.

`prepare` now emits `initial.jsonl` as the first 200 held-out rows. Its
`confirmatory.jsonl` selects held-out positions 500–999 after removing normalized
exact repeats against practice/earlier held-out rows. Reconstructed counts may
differ from the paper's 490. Dataset identity, counts and selection must be
reported rather than relabeled as the original split.

Verification uses actual tiny-model comparisons for baseline reuse, and a
controlled answer generator to exercise gates, case retests, both looks and
resume. This validates control flow, not published accuracy or significance.
