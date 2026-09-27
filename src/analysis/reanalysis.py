"""Leakage-free re-analysis of the two classification experiments.

Protocol A  is a deliberately leaky reference condition.
Protocol A' repeats A but measures how many test rows have a duplicate in train.
Protocol B  deduplicates first, splits before any resampling, handles imbalance with
            class weights inside training folds only, and uses repeated stratified CV.
Protocol C  keeps all rows but uses GroupKFold on duplicate clusters.
LODO        leave-one-source-dataset-out.
TEMPORAL    official communication split by posting year.
"""

import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    GroupKFold,
)
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from lightgbm import LGBMClassifier

RS = 928
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
RANKS = ["MaxRank", "MedianRank", "MedianContentRank", "MinContentRank"]
OTHER = ["Sentiment", "valence", "arousal", "dominance", "roberta_concreteness"]
FEATURES = OTHER + RANKS + GISPY


def load():
    c = pd.read_csv("corpus_labelled.csv")
    rv = pd.read_csv("features_rank_vad.csv")
    tf = pd.read_csv("features_transformer.csv")
    d = pd.concat([c.reset_index(drop=True), rv, tf], axis=1)
    d = d.merge(
        pd.read_csv("corpus_dupgroups.csv")[["d_id", "dupgroup"]], on="d_id", how="left"
    )
    # Log rank features and collapse the 5-point sentiment.
    d[RANKS] = np.log(d[RANKS])
    d["Sentiment"] = (
        0.25 * d.sentiment_2
        + 0.5 * d.sentiment_3
        + 0.75 * d.sentiment_4
        + 1.0 * d.sentiment_5
    )
    d = d.drop(columns=[f"sentiment_{i + 1}" for i in range(5)])
    return d.dropna(subset=FEATURES).reset_index(drop=True)


def models(balanced):
    cw = "balanced" if balanced else None
    return {
        "Decision tree": DecisionTreeClassifier(
            criterion="gini",
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=4,
            random_state=RS,
            class_weight=cw,
        ),
        "AdaBoost": AdaBoostClassifier(n_estimators=1000, random_state=RS),
        "LightGBM": LGBMClassifier(
            n_estimators=1000,
            importance_type="gain",
            random_state=RS,
            verbose=-1,
            class_weight=cw,
        ),
    }


def metrics(y, yh, yp):
    return dict(
        accuracy=accuracy_score(y, yh),
        balanced_accuracy=balanced_accuracy_score(y, yh),
        precision=precision_score(y, yh, zero_division=0),
        recall=recall_score(y, yh, zero_division=0),
        f1=f1_score(y, yh, zero_division=0),
        f1_macro=f1_score(y, yh, average="macro", zero_division=0),
        roc_auc=roc_auc_score(y, yp),
        pr_auc=average_precision_score(y, yp),
    )


def ci(v):
    v = np.asarray(v, dtype=float)
    m, s = v.mean(), v.std(ddof=1)
    h = 1.96 * s / np.sqrt(len(v))
    return m, m - h, m + h


def experiment(d, which):
    """Return (positive-class rows, negative-class rows) for an experiment."""
    fake = d[d.veracity == "fake"]
    if which == 1:
        return fake, d[d.veracity == "official"]
    return fake, d[d.veracity == "nonfake"]


# ---------------------------------------------------------------- Protocol A
def protocol_a(d, which):
    """Deliberately leaky reference condition."""
    pos, neg = experiment(d, which)
    if which == 1:
        xy1 = resample(pos, replace=True, n_samples=len(neg), random_state=RS)
        xy2 = neg
    else:
        xy1 = pos
        xy2 = resample(neg, replace=True, n_samples=len(pos), random_state=RS)
    xy = pd.concat([xy1, xy2])
    y = (xy.veracity == "fake").astype(int).values
    X = xy[FEATURES].values
    idx = np.arange(len(xy))
    Xtr, Xte, ytr, yte, itr, ite = train_test_split(
        X, y, idx, test_size=0.2, random_state=RS
    )

    # how much of the test set is a duplicate of something in training?
    did = xy.d_id.values
    dup_row = np.isin(did[ite], did[itr]).mean()
    grp = xy.dupgroup.values
    dup_grp = np.isin(grp[ite], grp[itr]).mean()

    rows = []
    for name, m in models(balanced=False).items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        r = metrics(yte, m.predict(Xte), p)
        r.update(model=name, train_accuracy=accuracy_score(ytr, m.predict(Xtr)))
        rows.append(r)
    return pd.DataFrame(rows), dup_row, dup_grp


# ---------------------------------------------------------------- Protocol B
def protocol_b(d, which, n_splits=5, n_repeats=2):
    """Deduplicate, then repeated stratified CV with class weights inside folds."""
    pos, neg = experiment(d, which)
    xy = pd.concat([pos, neg]).drop_duplicates("dupgroup").reset_index(drop=True)
    y = (xy.veracity == "fake").astype(int).values
    X = xy[FEATURES].values
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RS
    )
    per_fold = {k: [] for k in models(True)}
    for tr, te in cv.split(X, y):
        for name, m in models(balanced=True).items():
            if name == "AdaBoost":
                w = np.where(
                    y[tr] == 1,
                    len(y[tr]) / (2 * (y[tr] == 1).sum()),
                    len(y[tr]) / (2 * (y[tr] == 0).sum()),
                )
                m.fit(X[tr], y[tr], sample_weight=w)
            else:
                m.fit(X[tr], y[tr])
            per_fold[name].append(
                metrics(y[te], m.predict(X[te]), m.predict_proba(X[te])[:, 1])
            )
    rows = []
    for name, folds in per_fold.items():
        f = pd.DataFrame(folds)
        r = {"model": name, "n": len(xy), "n_pos": int(y.sum())}
        for k in f.columns:
            m, lo, hi = ci(f[k])
            r[k] = m
            r[k + "_lo"], r[k + "_hi"] = lo, hi
        rows.append(r)
    return pd.DataFrame(rows), xy


# ---------------------------------------------------------------- Protocol C
def protocol_c(d, which, n_splits=5):
    """All rows retained, but duplicate clusters never span the train/test boundary."""
    pos, neg = experiment(d, which)
    xy = pd.concat([pos, neg]).reset_index(drop=True)
    y = (xy.veracity == "fake").astype(int).values
    X = xy[FEATURES].values
    g = xy.dupgroup.values
    per_fold = {k: [] for k in models(True)}
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups=g):
        for name, m in models(balanced=True).items():
            if name == "AdaBoost":
                w = np.where(
                    y[tr] == 1,
                    len(y[tr]) / (2 * (y[tr] == 1).sum()),
                    len(y[tr]) / (2 * (y[tr] == 0).sum()),
                )
                m.fit(X[tr], y[tr], sample_weight=w)
            else:
                m.fit(X[tr], y[tr])
            per_fold[name].append(
                metrics(y[te], m.predict(X[te]), m.predict_proba(X[te])[:, 1])
            )
    rows = []
    for name, folds in per_fold.items():
        f = pd.DataFrame(folds)
        r = {"model": name}
        for k in f.columns:
            m, lo, hi = ci(f[k])
            r[k] = m
            r[k + "_lo"], r[k + "_hi"] = lo, hi
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- LODO
def lodo(d, which):
    """Train on all but one source dataset, test on the held-out one."""
    pos, neg = experiment(d, which)
    xy = pd.concat([pos, neg]).drop_duplicates("dupgroup").reset_index(drop=True)
    y = (xy.veracity == "fake").astype(int).values
    X = xy[FEATURES].values
    rows = []
    for src in sorted(xy.loc[xy.veracity == "fake", "source"].unique()):
        te = (xy.source == src).values
        tr = ~te
        # the held-out fold needs both classes: add the negative-class rows back in
        te = te | ((xy.veracity != "fake") & (np.arange(len(xy)) % 5 == 0)).values
        tr = tr & ~te
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        for name, m in models(balanced=True).items():
            if name == "AdaBoost":
                w = np.where(
                    y[tr] == 1,
                    len(y[tr]) / (2 * (y[tr] == 1).sum()),
                    len(y[tr]) / (2 * (y[tr] == 0).sum()),
                )
                m.fit(X[tr], y[tr], sample_weight=w)
            else:
                m.fit(X[tr], y[tr])
            r = metrics(y[te], m.predict(X[te]), m.predict_proba(X[te])[:, 1])
            r.update(held_out=src, model=name, n_test=int(te.sum()))
            rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- temporal
def temporal(d):
    """Experiment 1 with the official class split by posting year (2022 held out)."""
    meta = pd.read_csv("offcomm_metadata.csv", parse_dates=["dt"])
    pos, neg = experiment(d, 1)
    xy = pd.concat([pos, neg]).drop_duplicates("dupgroup").reset_index(drop=True)
    xy = xy.merge(meta[["d_id", "dt"]], on="d_id", how="left")
    xy["year"] = xy.dt.dt.year
    off_late = (xy.veracity == "official") & (xy.year >= 2022)
    off_early = (xy.veracity == "official") & (xy.year < 2022)
    fk = xy.veracity == "fake"
    rng = np.random.default_rng(RS)
    fk_test = fk & pd.Series(rng.random(len(xy)) < 0.2, index=xy.index)
    te = (off_late | fk_test).values
    tr = (off_early | (fk & ~fk_test)).values
    y = (xy.veracity == "fake").astype(int).values
    X = xy[FEATURES].values
    rows = []
    for name, m in models(balanced=True).items():
        if name == "AdaBoost":
            w = np.where(
                y[tr] == 1,
                len(y[tr]) / (2 * (y[tr] == 1).sum()),
                len(y[tr]) / (2 * (y[tr] == 0).sum()),
            )
            m.fit(X[tr], y[tr], sample_weight=w)
        else:
            m.fit(X[tr], y[tr])
        r = metrics(y[te], m.predict(X[te]), m.predict_proba(X[te])[:, 1])
        r.update(model=name, n_train=int(tr.sum()), n_test=int(te.sum()))
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- baselines
def baselines(xy, which):
    """Trivial and text-only reference points for the same deduplicated split."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = (xy.veracity == "fake").astype(int).values
    nw = xy.text.str.split().str.len().values.reshape(-1, 1)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=RS)
    specs = {
        "Majority class": (DummyClassifier(strategy="most_frequent"), nw),
        "Tweet length only": (
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
            nw,
        ),
        "Interpretable features (LightGBM)": (
            LGBMClassifier(
                n_estimators=1000, random_state=RS, verbose=-1, class_weight="balanced"
            ),
            xy[FEATURES].values,
        ),
        "TF-IDF word 1-2gram (logistic)": (
            make_pipeline(
                TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True),
                LogisticRegression(max_iter=2000, class_weight="balanced"),
            ),
            xy.text.values,
        ),
    }
    if EMB is not None:
        e = EMB.reindex(xy.d_id.values)
        if e.notna().all(axis=None):
            specs["Sentence-transformer embeddings (logistic)"] = (
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=3000, class_weight="balanced"),
                ),
                e.values,
            )
    rows = []
    for name, (mdl, Xb) in specs.items():
        folds = []
        for tr, te in cv.split(np.zeros(len(y)), y):
            from sklearn.base import clone

            m = clone(mdl)
            Xtr = Xb[tr] if not isinstance(Xb, np.ndarray) or Xb.ndim > 0 else Xb
            m.fit(Xtr, y[tr])
            p = (
                m.predict_proba(Xb[te])[:, 1]
                if hasattr(m, "predict_proba")
                else m.predict(Xb[te]).astype(float)
            )
            folds.append(metrics(y[te], m.predict(Xb[te]), p))
        f = pd.DataFrame(folds)
        r = {"baseline": name}
        for k in f.columns:
            mm, lo, hi = ci(f[k])
            r[k] = mm
            r[k + "_lo"], r[k + "_hi"] = lo, hi
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------- single-feature stumps
def stumps(xy):
    """Out-of-fold decision stump per feature: replaces the in-sample ranking."""
    y = (xy.veracity == "fake").astype(int).values
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RS)
    splits = list(cv.split(np.zeros(len(y)), y))
    rows = []
    for f in FEATURES:
        x = xy[f].values.reshape(-1, 1)
        acc, bal, auc = [], [], []
        for tr, te in splits:
            s = DecisionTreeClassifier(
                criterion="gini", max_depth=1, random_state=RS, class_weight="balanced"
            )
            s.fit(x[tr], y[tr])
            yh = s.predict(x[te])
            acc.append(accuracy_score(y[te], yh))
            bal.append(balanced_accuracy_score(y[te], yh))
            auc.append(roc_auc_score(y[te], s.predict_proba(x[te])[:, 1]))
        a_m, a_lo, a_hi = ci(acc)
        b_m, b_lo, b_hi = ci(bal)
        u_m, u_lo, u_hi = ci(auc)
        rows.append(
            dict(
                feature=f,
                accuracy=a_m,
                accuracy_lo=a_lo,
                accuracy_hi=a_hi,
                balanced_accuracy=b_m,
                balanced_accuracy_lo=b_lo,
                balanced_accuracy_hi=b_hi,
                roc_auc=u_m,
                roc_auc_lo=u_lo,
                roc_auc_hi=u_hi,
            )
        )
    return pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)


def perm_importance(xy):
    from sklearn.inspection import permutation_importance

    y = (xy.veracity == "fake").astype(int).values
    X = xy[FEATURES].values
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=RS, stratify=y
    )
    m = LGBMClassifier(
        n_estimators=1000, random_state=RS, verbose=-1, class_weight="balanced"
    )
    m.fit(Xtr, ytr)
    pi = permutation_importance(
        m,
        Xte,
        yte,
        n_repeats=10,
        random_state=RS,
        scoring="balanced_accuracy",
        n_jobs=1,
    )
    return pd.DataFrame(
        dict(feature=FEATURES, importance=pi.importances_mean, sd=pi.importances_std)
    ).sort_values("importance", ascending=False)


def oof_diagnostics(xy, tag):
    """Out-of-fold predictions -> confusion matrices, calibration, model comparison."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    from scipy import stats

    y = (xy.veracity == "fake").astype(int).values
    X = xy[FEATURES].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    oof = {k: np.zeros(len(y)) for k in models(True)}
    fold_bal = {k: [] for k in models(True)}
    for tr, te in cv.split(X, y):
        for name, m in models(balanced=True).items():
            if name == "AdaBoost":
                w = np.where(
                    y[tr] == 1,
                    len(y[tr]) / (2 * (y[tr] == 1).sum()),
                    len(y[tr]) / (2 * (y[tr] == 0).sum()),
                )
                m.fit(X[tr], y[tr], sample_weight=w)
            else:
                m.fit(X[tr], y[tr])
            oof[name][te] = m.predict_proba(X[te])[:, 1]
            fold_bal[name].append(balanced_accuracy_score(y[te], m.predict(X[te])))

    cm_rows, cal_rows = [], []
    for name, p in oof.items():
        yh = (p >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yh).ravel()
        cm_rows.append(
            dict(
                model=name,
                tn=tn,
                fp=fp,
                fn=fn,
                tp=tp,
                specificity=tn / (tn + fp),
                sensitivity=tp / (tp + fn),
                brier=brier_score_loss(y, p),
                roc_auc=roc_auc_score(y, p),
                pr_auc=average_precision_score(y, p),
            )
        )
        ft, mp = calibration_curve(y, p, n_bins=10, strategy="quantile")
        for a, b in zip(mp, ft):
            cal_rows.append(dict(model=name, mean_predicted=a, observed_fraction=b))
    pd.DataFrame(cm_rows).to_csv(f"table_{tag}_confusion.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(f"table_{tag}_calibration.csv", index=False)

    names = list(fold_bal)
    comp = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = np.array(fold_bal[names[i]]), np.array(fold_bal[names[j]])
            t = stats.ttest_rel(a, b)
            comp.append(
                dict(
                    model_a=names[i],
                    model_b=names[j],
                    mean_diff=(a - b).mean(),
                    t=t.statistic,
                    p=t.pvalue,
                )
            )
    pd.DataFrame(comp).to_csv(f"table_{tag}_model_comparison.csv", index=False)
    np.save(f"oof_{tag}.npy", np.vstack([oof[n] for n in names]))
    return pd.DataFrame(cm_rows), pd.DataFrame(comp)


def surface_artifacts(xy, tag):
    """Do URLs, hashtags, mentions and retweet markers alone separate the classes?"""
    t = xy.text.astype(str)
    S = pd.DataFrame(
        {
            "has_url": t.str.contains(r"https?://", na=False).astype(int),
            "n_hashtags": t.str.count(r"#\w+"),
            "n_mentions": t.str.count(r"@\w+"),
            "is_retweet": t.str.match(r"^\s*RT\s+@", na=False).astype(int),
            "n_words": t.str.split().str.len(),
            "n_chars": t.str.len(),
            "n_upper": t.str.count(r"[A-Z]"),
            "n_exclaim": t.str.count(r"!"),
        }
    )
    prev = S.assign(veracity=xy.veracity.values).groupby("veracity").mean().round(3)
    prev.to_csv(f"table_{tag}_surface_prevalence.csv")
    y = (xy.veracity == "fake").astype(int).values
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=RS)
    folds = []
    for tr, te in cv.split(S.values, y):
        m = LGBMClassifier(
            n_estimators=300, random_state=RS, verbose=-1, class_weight="balanced"
        )
        m.fit(S.values[tr], y[tr])
        folds.append(
            metrics(y[te], m.predict(S.values[te]), m.predict_proba(S.values[te])[:, 1])
        )
    f = pd.DataFrame(folds)
    r = {"features": "surface/metadata only (8)"}
    for k in f.columns:
        mm, lo, hi = ci(f[k])
        r[k], r[k + "_lo"], r[k + "_hi"] = mm, lo, hi
    out = pd.DataFrame([r])
    out.to_csv(f"table_{tag}_surface_model.csv", index=False)
    return prev, out


def topics(d, n_topics=12):
    """NMF topics over the pooled corpus, then the topic mix of each source."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF

    v = TfidfVectorizer(max_features=20000, min_df=5, stop_words="english")
    M = v.fit_transform(d.text.astype(str))
    nmf = NMF(n_components=n_topics, random_state=RS, init="nndsvd", max_iter=400)
    W = nmf.fit_transform(M)
    vocab = np.array(v.get_feature_names_out())
    terms = [
        ", ".join(vocab[np.argsort(-nmf.components_[k])[:8]]) for k in range(n_topics)
    ]
    share = W / np.clip(W.sum(1, keepdims=True), 1e-9, None)
    td = pd.DataFrame(share, columns=[f"T{k + 1}" for k in range(n_topics)])
    td["group"] = (d.veracity + " / " + d.source).values
    prof = td.groupby("group").mean().round(4).T
    prof.insert(0, "top_terms", terms)
    prof.to_csv("table_topic_distribution.csv")
    return prof


EMB = None
try:
    _e = pd.read_parquet("embeddings_minilm.parquet").set_index("d_id")
    EMB = _e
except Exception:
    pass


def main():
    d = load()
    print(
        f"analytic rows {len(d)} | features {len(FEATURES)} | "
        f"{d.veracity.value_counts().to_dict()}",
        flush=True,
    )
    log = {}
    for which, tag in [(1, "exp1_fake_vs_official"), (2, "exp2_fake_vs_nonfake")]:
        A, dup_row, dup_grp = protocol_a(d, which)
        A.insert(0, "protocol", "A leaky reference")
        print(
            f"\n[{tag}] Protocol A  test rows duplicated in train: "
            f"{100 * dup_row:.2f}% exact, {100 * dup_grp:.2f}% incl. near-duplicates",
            flush=True,
        )
        print(
            A[["model", "train_accuracy", "accuracy", "balanced_accuracy", "roc_auc"]]
            .round(4)
            .to_string(index=False),
            flush=True,
        )
        log[tag + "_dup_leak"] = dict(exact=float(dup_row), cluster=float(dup_grp))

        B, xy = protocol_b(d, which)
        B.insert(0, "protocol", "B corrected (dedup + CV)")
        print(f"[{tag}] Protocol B  n={len(xy)}", flush=True)
        print(
            B[
                [
                    "model",
                    "accuracy",
                    "balanced_accuracy",
                    "f1_macro",
                    "roc_auc",
                    "pr_auc",
                ]
            ]
            .round(4)
            .to_string(index=False),
            flush=True,
        )

        C = protocol_c(d, which)
        C.insert(0, "protocol", "C grouped CV (all rows)")
        pd.concat([A, B, C], ignore_index=True).to_csv(
            f"table_{tag}_protocols.csv", index=False
        )

        L = lodo(d, which)
        L.to_csv(f"table_{tag}_lodo.csv", index=False)
        print(f"[{tag}] LODO balanced accuracy by held-out source:", flush=True)
        print(
            L.pivot_table(index="held_out", columns="model", values="balanced_accuracy")
            .round(4)
            .to_string(),
            flush=True,
        )

        S = stumps(xy)
        S.to_csv(f"table_{tag}_stumps.csv", index=False)
        print(
            f"[{tag}] top 6 single features (out-of-fold balanced accuracy):",
            flush=True,
        )
        print(
            S.head(6)[["feature", "accuracy", "balanced_accuracy", "roc_auc"]]
            .round(4)
            .to_string(index=False),
            flush=True,
        )

        Bl = baselines(xy, which)
        Bl.to_csv(f"table_{tag}_baselines.csv", index=False)
        print(f"[{tag}] baselines:", flush=True)
        print(
            Bl[["baseline", "accuracy", "balanced_accuracy", "roc_auc"]]
            .round(4)
            .to_string(index=False),
            flush=True,
        )

        P = perm_importance(xy)
        P.to_csv(f"table_{tag}_permutation.csv", index=False)
        print(f"[{tag}] top 8 by permutation importance:", flush=True)
        print(P.head(8).round(4).to_string(index=False), flush=True)

        CM, CP = oof_diagnostics(xy, tag)
        print(f"[{tag}] out-of-fold confusion / calibration:", flush=True)
        print(CM.round(4).to_string(index=False), flush=True)
        print(
            f"[{tag}] paired model comparison (fold-level balanced accuracy):",
            flush=True,
        )
        print(CP.round(4).to_string(index=False), flush=True)

        PV, SM = surface_artifacts(xy, tag)
        print(f"[{tag}] surface/metadata prevalence by class:", flush=True)
        print(PV.to_string(), flush=True)
        print(f"[{tag}] surface/metadata-only model:", flush=True)
        print(
            SM[["features", "accuracy", "balanced_accuracy", "roc_auc"]]
            .round(4)
            .to_string(index=False),
            flush=True,
        )

    T = temporal(d)
    T.to_csv("table_exp1_temporal.csv", index=False)
    print("\n[exp1] temporal holdout (official 2022 held out):", flush=True)
    print(
        T[["model", "n_train", "n_test", "accuracy", "balanced_accuracy", "roc_auc"]]
        .round(4)
        .to_string(index=False),
        flush=True,
    )

    TP = topics(d)
    print("\ntopic mix by source (NMF, 12 topics):", flush=True)
    print(TP.to_string(), flush=True)

    json.dump(log, open("reanalysis_log.json", "w"), indent=1)


if __name__ == "__main__":
    main()
