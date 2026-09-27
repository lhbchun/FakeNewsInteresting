"""
build_corpus.py — reconstruct the labelled analysis corpus from the source releases.

The GisPy feature extractor writes one row per document but does not carry the
class label, so per-item veracity has to be re-derived from the original dataset
releases. This script does that and writes `corpus_labelled.csv`, the single input
to every downstream analysis.

Label derivation, by source:
  offComm         official     — official public-health accounts, by construction
  coaid           fake         — only the fake-labelled CoAID subset was extracted
  constraintAAAI  fake/nonfake — positional join on the CONSTRAINT training file,
                                 which preserves the extraction order (label column)
  truthseeker     fake/nonfake — text join on a punctuation/case/URL-stripped key;
                                 BinaryNumTarget 0 -> fake, 1 -> nonfake
  covidRumor      fake/nonfake — text join against the rumour and news release files;
                                 label F -> fake, T -> nonfake, U -> dropped.
                                 The extractor had replaced the token "that" with
                                 "qwertyu" in this subset, so the join key reverses
                                 that substitution (see key2); items still unmatched
                                 are joined on the first 45 key characters.

Items that cannot be labelled unambiguously, and COVID-Rumor items annotated
"unverified", are dropped rather than guessed.

Inputs (paths relative to the project archive):
    repo/gispy/result.csv             GisPy features, offComm + constraintAAAI + covidRumor
    repo/gispy/result2.csv            GisPy features, coaid
    repo/gispy/resultTruthseeker.csv  GisPy features, truthseeker
    labels/constraint_train.csv       CONSTRAINT/AAAI training labels
    labels/truthseeker.csv            TruthSeeker labels
    labels/rumor_en_dup.csv           COVID-Rumor rumour statements
    labels/rumor_news.csv             COVID-Rumor news statements

Output:
    corpus_labelled.csv   d_id, source, veracity, text, + 34 GisPy features

Usage:  python build_corpus.py
"""

import html
import re

import numpy as np
import pandas as pd

GISPY = [
    "DESPC",
    "DESSC",
    "CoREF",
    "PCREF_1",
    "PCREF_a",
    "PCREF_1p",
    "PCREF_ap",
    "PCDC",
    "SMCAUSe_1",
    "SMCAUSe_a",
    "SMCAUSe_1p",
    "SMCAUSe_ap",
    "SMCAUSwn_1p_path",
    "SMCAUSwn_1p_lch",
    "SMCAUSwn_1p_wup",
    "SMCAUSwn_ap_path",
    "SMCAUSwn_ap_lch",
    "SMCAUSwn_ap_wup",
    "SMCAUSwn_1_path",
    "SMCAUSwn_1_lch",
    "SMCAUSwn_1_wup",
    "SMCAUSwn_a_path",
    "SMCAUSwn_a_lch",
    "SMCAUSwn_a_wup",
    "SMCAUSwn_1p_binary",
    "SMCAUSwn_ap_binary",
    "SMCAUSwn_1_binary",
    "SMCAUSwn_a_binary",
    "PCCNC_megahr",
    "WRDIMGc_megahr",
    "PCCNC_mrc",
    "WRDIMGc_mrc",
    "WRDHYPnv",
    "gis",
]

GISPY_DIR = "repo/gispy"
LABEL_DIR = "labels"


def key(s):
    """Join key: unescape HTML, lowercase, drop URLs, keep alphanumerics only."""
    s = html.unescape(str(s)).lower()
    s = re.sub(r"https?://\S+", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def key2(s):
    """As key(), but reversing the 'that' -> 'qwertyu' substitution."""
    s = html.unescape(str(s)).lower().replace("qwertyu", "that")
    s = re.sub(r"https?://\S+", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


# ───────────────── features: three GisPy result files ─────────────────
cols = ["d_id", "text"] + GISPY
r1 = pd.read_csv(f"{GISPY_DIR}/result.csv", usecols=cols)
r2 = pd.read_csv(f"{GISPY_DIR}/result2.csv", usecols=cols)
r4 = pd.read_csv(f"{GISPY_DIR}/resultTruthseeker.csv", usecols=cols)

r1["source"] = r1.d_id.str.replace(r"_\d+\.txt", "", regex=True)
r2["source"] = "coaid"
r4["source"] = "truthseeker"

corp = pd.concat([r1, r2, r4], ignore_index=True)
corp["i"] = corp.d_id.str.extract(r"_(\d+)\.txt")[0].astype(int)

# ───────────────────────────── labels ─────────────────────────────
corp["veracity"] = None
corp.loc[corp.source == "offComm", "veracity"] = "official"
corp.loc[corp.source == "coaid", "veracity"] = "fake"

# CONSTRAINT/AAAI: positional join on the training file
ctr = pd.read_csv(f"{LABEL_DIR}/constraint_train.csv")
m = corp.source == "constraintAAAI"
corp.loc[m, "veracity"] = np.where(
    ctr.label.values[corp.loc[m, "i"].values] == "fake", "fake", "nonfake"
)

# TruthSeeker: text join
tsb = pd.read_csv(f"{LABEL_DIR}/truthseeker.csv", usecols=["tweet", "BinaryNumTarget"])
tsb["K"] = tsb.tweet.map(key)
m = corp.source == "truthseeker"
corp.loc[m, "veracity"] = (
    corp.loc[m, "text"]
    .map(key)
    .map(tsb.drop_duplicates("K").set_index("K")["BinaryNumTarget"])
    .map({0.0: "fake", 1.0: "nonfake"})
)

# COVID-Rumor: text join against both release files, then a prefix fallback
ru2 = pd.read_csv(f"{LABEL_DIR}/rumor_en_dup.csv")
ru2["K2"] = ru2.content.map(key2)
nw = pd.read_csv(
    f"{LABEL_DIR}/rumor_news.csv", header=None, names=["id", "label", "content", "x"]
)
nw["K2"] = nw.content.map(key2)
comb = (
    pd.concat([ru2[["K2", "label"]], nw[["K2", "label"]]])
    .drop_duplicates("K2")
    .set_index("K2")["label"]
)

m = corp.source == "covidRumor"
corp.loc[m, "veracity"] = (
    corp.loc[m, "text"]
    .map(key2)
    .map(comb)
    .map({"F": "fake", "T": "nonfake", "U": "unlabelled"})
)

un = corp[(corp.source == "covidRumor") & corp.veracity.isna()].copy()
src = (
    pd.concat([ru2[["K2", "label"]], nw[["K2", "label"]]])
    .dropna()
    .drop_duplicates("K2")
)
best = src.assign(P=src.K2.str[:45]).drop_duplicates("P").set_index("P")["label"]
fill = (
    un.text.map(key2)
    .str[:45]
    .map(best)
    .map({"F": "fake", "T": "nonfake", "U": "unlabelled"})
)
corp.loc[fill.dropna().index, "veracity"] = fill.dropna()

# ───────────────────────────── output ─────────────────────────────
out = corp[corp.veracity.notna() & (corp.veracity != "unlabelled")].copy()
out = out[["d_id", "source", "veracity", "text"] + GISPY].reset_index(drop=True)
out.to_csv("corpus_labelled.csv", index=False)

print("rows:", len(out))
print(out.groupby(["veracity", "source"]).size().to_string())
