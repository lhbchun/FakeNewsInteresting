# Multimedia Appendix 5 — Analysis Code

**Purpose.** This appendix contains the source code used to produce the reported
analyses, tables and figures, released so that the work can be inspected, re-run
and re-used independently.

Every reported result was computed from raw text by the scripts in this appendix.
No reported result is taken from a stored intermediate output file.

---

## Contents

| Folder | File | What it does |
|---|---|---|
| `corpus` | `build_corpus.py` | Reconstructs the labelled analysis corpus. Re-derives per-item veracity labels for each source dataset from the original releases, drops items that cannot be labelled unambiguously, and writes `corpus_labelled.csv` (52,917 items). |
| `features` | `recompute_features.py` | Recomputes the affective and lexical-frequency features from raw text: valence, arousal and dominance (Warriner lexicon); word-frequency rank statistics; five-class transformer sentiment. |
| | `recompute_transformers.py` | Recomputes model-based concreteness with the published regression model. |
| | `embed_minilm.py` | Produces sentence-transformer embeddings used only for the reference baseline, not for the interpretable-feature models. |
| `analysis` | `reanalysis.py` | All classification analyses: the three validation protocols, the leakage diagnostic, reference baselines, the surface/metadata probe, leave-one-dataset-out transfer, the temporal holdout, single-feature stumps, permutation importance, calibration, confusion tables and model comparison tests. Writes `results_values.json` and all `table_*.csv` files. |
| `reporting` | `make_figures.py` | Regenerates Figures 1-4 at 300 dpi from the saved result tables. |

---

## Data inputs

The scripts expect the source-data folders alongside them:

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
python corpus/build_corpus.py          # -> corpus_labelled.csv
python features/recompute_features.py  # -> features_rank_vad.csv
python features/recompute_transformers.py
python features/embed_minilm.py        # baseline only; needs ~2 GB of model downloads
python analysis/reanalysis.py          # -> results_values.json, table_*.csv
python reporting/make_figures.py       # -> figure1..4 .png
```

Run each script from the directory holding the data inputs, since all paths are
relative to the working directory. The random seed is fixed at 928 throughout.
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
