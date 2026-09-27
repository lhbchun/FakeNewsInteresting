# Multimedia Appendix 5 — Analysis Code

**Purpose.** This appendix contains the source code used to produce the reported
analyses, tables and figures, released so that the work can be inspected, re-run
and re-used independently.

Every reported result was computed from raw text by the scripts in this appendix.
No reported result is taken from a stored intermediate output file.

---

## Contents

- `src/corpus/build_corpus.py` — Reconstructs the labelled analysis corpus, re-derives per-item veracity labels from the source releases, drops items that cannot be labelled unambiguously, and writes `corpus_labelled.csv` (52,917 items).
- `src/features/recompute_features.py` — Recomputes affective and lexical-frequency features from raw text, including valence, arousal, dominance, word-frequency ranks, and five-class transformer sentiment.
- `src/features/recompute_transformers.py` — Recomputes model-based concreteness with the published regression model.
- `src/features/embed_minilm.py` — Produces sentence-transformer embeddings for the reference baseline.
- `src/analysis/reanalysis.py` — Runs the classification analyses, validation protocols, leakage diagnostic, baselines, transfer and temporal validation, feature importance, calibration, and model comparison. Writes `results_values.json` and `table_*.csv` files.
- `src/reporting/make_figures.py` — Regenerates Figures 1-4 at 300 dpi from the saved result tables.

---

## Data inputs

Run the scripts from the project root. They expect these source-data folders there:

```
repo/gispy/result.csv             cohesion and concreteness features: official, CONSTRAINT, COVID-Rumor
repo/gispy/result2.csv            cohesion and concreteness features: CoAID
repo/gispy/resultTruthseeker.csv  cohesion and concreteness features: TruthSeeker
labels/constraint_train.csv       CONSTRAINT/AAAI labels
labels/truthseeker.csv            TruthSeeker labels
labels/rumor_en_dup.csv           COVID-Rumor rumour statements
labels/rumor_news.csv             COVID-Rumor news statements
```

The four source misinformation datasets (CONSTRAINT/AAAI, CoAID, COVID-Rumor,
TruthSeeker) are public and are not redistributed here. Official public-health posts were collected under the
platform's terms of service and only post identifiers can be shared.

## Reproducing the results

```bash
pip install -r requirements.txt
python src/corpus/build_corpus.py          # -> corpus_labelled.csv
python src/features/recompute_features.py  # -> features_rank_vad.csv
python src/features/recompute_transformers.py
python src/features/embed_minilm.py        # baseline only; needs ~2 GB of model downloads
python src/analysis/reanalysis.py          # -> results_values.json, table_*.csv
python src/reporting/make_figures.py       # -> figure1..4 .png
```

Scripts read inputs and write outputs relative to the current working directory.
The random seed is fixed at 928 throughout.
Runtime is dominated by the transformer feature passes in `features`; the
classification analyses in `analysis` are comparatively inexpensive.

## Notes and known limitations

- **No hyperparameter search was performed.** Library defaults are used except
  where settings are specified in the analysis scripts. Reported values are
  therefore not upper bounds on achievable performance. This applies equally to all
  protocols and baselines and so does not account for the differences between
  them.
- **Two transformer regressors are no longer publicly retrievable.** The
  analytic feature set is consequently 43 features rather than 45.
- **Label recovery is not exhaustive.** Items whose labels cannot be
  unambiguously re-derived are dropped rather than guessed.
- `analysis/reanalysis.py` includes a deliberately leaky Protocol A to quantify
  leakage; its estimates should not be read as valid performance estimates.

## Licence

A licence for this code is to be confirmed by the authors before deposit (a
permissive licence such as MIT or BSD-3-Clause is suggested). The source datasets
retain their own licences.

This repository was developed with assistance from Claude Opus 5.5 for code refactoring and quality maintenance to improve manuscript quality.
