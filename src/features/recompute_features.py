"""Faithful recomputation of the affective / lexical-frequency features.

Mirrors FakeNewsInteresting/preprocess.py exactly:
  tk()      : strip punctuation, lowercase, word_tokenize
  VAD       : mean Warriner V/A/D.Mean.Sum over lemmatised non-stopword tokens; 5 if none
  ranks     : 1-based rank in unigram_freq.csv (count-desc); sentinels -1 / 999999
  sentiment : softmax over the 5 logits of the 5-class BERT sentiment model
  concrete  : logits[0][0] of j-hartmann/concreteness-english-distilroberta-base
"""

import re, sys, math
import numpy as np
import pandas as pd
from statistics import median, StatisticsError

import os
import nltk

nltk.data.path.insert(0, os.path.abspath("nltk_data"))
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

wnl = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

corp = pd.read_csv("corpus_labelled.csv")
texts = corp["text"].astype(str).tolist()
print("texts", len(texts), flush=True)


def tk(text):
    return word_tokenize(re.sub(r"[^\w\s]", "", str(text).lower()))


def stopword(tokens):
    return [w for w in tokens if w not in stop_words]


def lemma(tokens):
    return [wnl.lemmatize(t) for t in tokens]


# ---------- Warriner VAD ----------
vad = pd.read_csv("ps928/labels/warriner.csv", index_col=1)
V, A, D = (vad[c].to_dict() for c in ["V.Mean.Sum", "A.Mean.Sum", "D.Mean.Sum"])


def vad_mean(tokens, table):
    vals = [table[t] for t in tokens if t in table]
    return sum(vals) / len(vals) if vals else 5


# ---------- unigram ranks ----------
uf = pd.read_parquet("ps928/labels/train-00000-of-00001.parquet")
uf.columns = [c.lower() for c in uf.columns]
uf = uf.reset_index(drop=True)
rankDict = {w: i + 1 for i, w in enumerate(uf["word"].astype(str).tolist())}
print("unigram vocab", len(rankDict), "| head", uf["word"].head(3).tolist(), flush=True)

rows = []
for t in texts:
    toks = tk(t)
    cont = stopword(toks)
    rk = [rankDict[j] for j in toks if j in rankDict]
    rkc = [rankDict[j] for j in cont if j in rankDict]
    lem = lemma(cont)
    rows.append(
        (
            max(rk) if rk else -1,  # MaxRank
            median(rk) if rk else -1,  # MedianRank
            median(rkc) if rkc else -1,  # MedianContentRank
            min(rkc) if rkc else 999999,  # MinContentRank
            vad_mean(lem, V),
            vad_mean(lem, A),
            vad_mean(lem, D),
        )
    )
feat = pd.DataFrame(
    rows,
    columns=[
        "MaxRank",
        "MedianRank",
        "MedianContentRank",
        "MinContentRank",
        "valence",
        "arousal",
        "dominance",
    ],
)
feat.to_csv("features_rank_vad.csv", index=False)
print("rank+VAD done", feat.shape, flush=True)
print(feat.describe().round(3).to_string(), flush=True)
