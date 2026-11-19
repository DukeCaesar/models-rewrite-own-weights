# Source data and reconstruction limits

The article does not publish its actual split identifiers, image draw seed, or
complete grading implementation. Converters preserve source-row order and record
input file hashes. These do not establish equivalence to the article's samples.

Verified source pages:

- [DeepScaleR](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset): already downloaded at a recorded revision.
- [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) and [AIME 2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024).
- [MathVista](https://huggingface.co/datasets/AI4Math/MathVista): use `testmini`; converter maps reference choice text to a letter and retains choices in the prompt.
- [MathVerse](https://huggingface.co/datasets/AI4Math/MathVerse): converter keeps `Vision Only` rows and uses `query_cot`, since their `question` may be empty.
- [MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar): `val`; choices already occur in the source question.
- [HallusionBench mirror](https://huggingface.co/datasets/lmms-lab-encoder/HallusionBench): converter excludes rows explicitly marked without visual input and maps binary reference labels.
- [POPE mirror](https://huggingface.co/datasets/lmms-lab-encoder/POPE): yes/no labels; source variants must be recorded by the user.
- [ScreenSpot-v2](https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2): original JSON/image exports; choose the bounding-box coordinate convention explicitly.

Pinned AIME 2025, HMMT February 2025, AMC 2023 and OlympiadBench exclusion
sources have now been obtained and converted. GPQA-Diamond remains gated and
requires authorized access or a local export. Semantic overlaps remain unaudited. No substitute is silently used for
a missing source or access restriction. The converter supports common original
text field names, but unfamiliar schemas fail and require explicit inspection.

Import downloaded original exports (JSON array, JSONL, CSV, or Parquet):

```bash
self-rewrite import-benchmark --benchmark math500 --source raw/math500.parquet \
  --output benchmarks/math500 --repository HuggingFaceH4/MATH-500 --revision COMMIT
self-rewrite import-benchmark --benchmark mathverse --source raw/mathverse.parquet \
  --output benchmarks/mathverse --repository AI4Math/MathVerse --revision COMMIT
self-rewrite import-benchmark --benchmark screenspot-v2 --source raw/screenspot.json \
  --image-root raw/images --bbox-space pixels --bbox-format xywh --output benchmarks/screenspot-v2
```

Repeat `--source` for multiple shards. Embedded Parquet image bytes and local image
paths are supported. Images are saved losslessly as RGB PNG. Each output has
`problems.jsonl` and a provenance `manifest.json`; pass the former to `prepare`
or `prepare-images`. Repository/revision labels are user-supplied provenance,
while file hashes are computed. Exported images and metadata must be reviewed
before treating a new source schema as validated.

Hugging Face download access recovered after the earlier DNS failure. Six of seven
exclusion sources are now local. Recorded revisions and file hashes are in
`data/source-metadata/` and `data/exclusion-raw/*/download.json`; normalized
exclusion records are in `data/exclusions/*/problems.jsonl`. GPQA returned HTTP 401.
`data/exclusion-audit.json` records exact matches only, not semantic equivalence.

Use `--exclusion-only` for benchmark question extraction without evaluation
references (needed for proof and multi-answer OlympiadBench rows). These records
cannot be passed to model generation. AMC imports require a year field or dated
source URL, and select only 2023. The DeepScaleR import rejects six empty
reference-answer rows with an explicit source-row audit; 40,309 usable rows remain.


Actual image exports have now been downloaded at recorded revisions. Raw file
hashes are in `data/image-raw/*/download.json`; acquisition selects MathVista
`testmini`, MathVerse `testmini` (filtering Vision Only), MMStar `val`,
HallusionBench's image split, all three POPE test shards, and the original
ScreenSpot-v2 JSON/image archive. Source conversion is tracked separately from
successful download.

Verified source-specific corrections:
- ScreenSpot-v2's original boxes are pixel xywh, requiring both explicit flags.
  One source box extends one pixel past the image. Pixel annotations may be
  clipped by at most one pixel; larger violations fail. Corrections retain the
  original pixel box in the reference and are listed in the import manifest.
- MathVista record pid 781 has identical correct values in options C and D.
  Conversion retains the item and permits either label (`answer: "C,D"`).
- MathVista numeric precision and units are retained as prompt instructions.
- MathVerse has value-valued choice references, parenthesized labels, and two
  questions requiring multiple letters. The importer maps exact option values
  using evaluation-only source text without adding that text to the prompt.
  `choice-set` requires every referenced letter; it differs from interchangeable
  MathVista options accepted by `choice`.
- MMStar's original image column stores raw bytes; both raw bytes and Hugging
  Face image dictionaries are supported.

Near-duplicate review:

```bash
self-rewrite audit-overlap --pool data/deepscaler.jsonl \
  --exclude math500=data/exclusions/math500/problems.jsonl \
  --output data/candidates.jsonl
```

Include each available benchmark with another `--exclude`. This retrieves
TF-IDF bigram/trigram candidates, not guaranteed semantic matches. Review both
full questions, set `decision` to `overlap` or `distinct`, and add a `review_note`.
`reviewed-exclusions` requires a decision for every candidate before writing a
manual-exclusion ID list. A no-match result never proves absence of paraphrases.
The real six-source audit retrieved 206 candidates at threshold 0.65; its first
15 pairs have explicit semantic reviews. Original candidate output is retained.


Completed local conversion counts are 1,000 MathVista, 788 MathVerse, 1,500 MMStar,
951 HallusionBench, 9,000 POPE, and 1,272 ScreenSpot-v2 questions. POPE's 500
unique decoded images are stored once while retaining all 9,000 question rows.
`python scripts/import_images.py` verifies downloaded hashes, converts missing
imports, and checks completed imports without replacing them. Partial outputs
must be moved aside explicitly before retrying.

Create a new reproducible image selection after importing:

```bash
self-rewrite prepare-images \
  --source mathvista=data/image-benchmarks/mathvista/problems.jsonl \
  --source mathverse=data/image-benchmarks/mathverse/problems.jsonl \
  --source mmstar=data/image-benchmarks/mmstar/problems.jsonl \
  --source hallusionbench=data/image-benchmarks/hallusionbench/problems.jsonl \
  --source pope=data/image-benchmarks/pope/problems.jsonl \
  --source screenspot-v2=data/image-benchmarks/screenspot-v2/problems.jsonl \
  --output data/image-evaluation --seed 0
```

The local `data/image-evaluation` selection already exists: 100 questions per
benchmark, 600 total. The manifest records the 300-item draw, 100-item selection,
and selected image hashes. Seed zero is a reconstruction choice, not a claim to
match the paper's unpublished draw. No accuracy evaluation has been run on it.
