"""
make_figures.py — regenerate Figures 1-4 from saved analysis results.

Reads:
    results_values.json                                   (all point estimates)
    table_exp1_fake_vs_official_protocols.csv             (Figure 1, left panel)
    table_exp2_fake_vs_nonfake_protocols.csv              (Figure 1, right panel)

Writes:
    figure1_protocols.png   figure2_lodo.png
    figure3_baselines.png   figure4_importance.png

All figures are 300 dpi. Legends are placed outside the plotting area so that no
legend element overlaps a data element.

Usage:  python make_figures.py
"""

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ───────────────────────────── style ─────────────────────────────
mpl.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.titlelocation": "left",
        "axes.titlepad": 6,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": False,
        "axes.grid": False,
    }
)

MODELS = ["Decision tree", "AdaBoost", "LightGBM"]
MODEL_C = {"Decision tree": "#9aa3ad", "AdaBoost": "#4e79b6", "LightGBM": "#c2410c"}
META_GREY = "#8b929b"
PANELS = [
    ("a1", "Analysis 1: fake news vs official communication"),
    ("a2", "Analysis 2: fake news vs non-fake news"),
]

V = json.load(open("results_values.json"))


def finish(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(path)


# ═══════════════════ Figure 1 — validation protocols ═══════════════════
PROTO_FILES = {
    "a1": "table_exp1_fake_vs_official_protocols.csv",
    "a2": "table_exp2_fake_vs_nonfake_protocols.csv",
}
PROTO_LABELS = {
    "A leaky reference": "Leaky reference\n(oversample, then split)",
    "B corrected (dedup + CV)": "Deduplicated\n5x4-fold CV",
    "C grouped CV (all rows)": "Grouped CV\n(all rows)",
}


def figure1():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    order = list(PROTO_LABELS)
    w = 0.26
    for ax, (tag, title) in zip(axes, PANELS):
        df = pd.read_csv(PROTO_FILES[tag])
        x = np.arange(len(order))
        for j, m in enumerate(MODELS):
            sub = df[df.model == m].set_index("protocol").reindex(order)
            v = sub.balanced_accuracy.values
            lo = sub.balanced_accuracy_lo.values
            hi = sub.balanced_accuracy_hi.values
            err = np.vstack(
                [np.where(np.isnan(lo), 0, v - lo), np.where(np.isnan(hi), 0, hi - v)]
            )
            ax.bar(x + (j - 1) * w, v, w, color=MODEL_C[m], zorder=2)
            ax.errorbar(
                x + (j - 1) * w,
                v,
                yerr=err,
                fmt="none",
                ecolor="#1f2937",
                elinewidth=0.8,
                capsize=1.6,
                zorder=3,
            )
            if m == "LightGBM":
                for xx, vv in zip(x[:2], v[:2]):
                    ax.annotate(
                        f"{vv:.3f}",
                        (xx + w, vv),
                        xytext=(0, 9),
                        textcoords="offset points",
                        ha="center",
                        fontsize=7.5,
                        color=MODEL_C[m],
                    )
        ax.axhline(0.5, color=META_GREY, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(list(PROTO_LABELS.values()))
        ax.set_ylim(0.46, 1.03)
        ax.set_title(title)
    axes[0].set_ylabel("Balanced accuracy")
    handles = [Patch(facecolor=MODEL_C[m], label=m) for m in MODELS] + [
        Line2D([], [], color=META_GREY, lw=0.8, ls=(0, (4, 3)), label="chance (0.5)")
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.10),
        columnspacing=1.6,
        handlelength=1.6,
    )
    fig.text(
        0.995,
        -0.055,
        "higher = better",
        ha="right",
        va="top",
        fontsize=7,
        color=META_GREY,
    )
    fig.tight_layout()
    finish(fig, "figure1_protocols.png")


# ═══════════════ Figure 2 — leave-one-dataset-out transfer ═══════════════
DS_LABEL = {
    "coaid": "CoAID",
    "constraintAAAI": "CONSTRAINT",
    "covidRumor": "COVID-Rumor",
    "truthseeker": "TruthSeeker",
}
WITHIN = {"a1": 0.943693, "a2": 0.803969}


def figure2():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    order = list(DS_LABEL)
    w = 0.26
    for ax, (tag, title) in zip(axes, PANELS):
        lodo, ns = V[tag + "_lodo"], V[tag + "_lodo_n"]
        x = np.arange(len(order))
        for j, m in enumerate(MODELS):
            v = [lodo[m][d] for d in order]
            ax.bar(x + (j - 1) * w, v, w, color=MODEL_C[m], zorder=2)
        ax.axhline(
            WITHIN[tag], color=MODEL_C["LightGBM"], lw=1.0, ls=(0, (5, 2)), zorder=1
        )
        ax.axhline(0.5, color=META_GREY, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{DS_LABEL[d]}\n(n={ns[d]:,})" for d in order])
        ax.set_ylim(0.46, 1.03)
        ax.set_title(title)
    axes[0].set_ylabel("Balanced accuracy\non held-out dataset")
    handles = [Patch(facecolor=MODEL_C[m], label=m) for m in MODELS] + [
        Line2D(
            [],
            [],
            color=MODEL_C["LightGBM"],
            lw=1.0,
            ls=(0, (5, 2)),
            label="within-corpus LightGBM",
        ),
        Line2D([], [], color=META_GREY, lw=0.8, ls=(0, (4, 3)), label="chance (0.5)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.10),
        columnspacing=1.4,
        handlelength=1.6,
    )
    fig.text(
        0.995,
        -0.055,
        "higher = better",
        ha="right",
        va="top",
        fontsize=7,
        color=META_GREY,
    )
    fig.tight_layout()
    finish(fig, "figure2_lodo.png")


# ═══════════════════ Figure 3 — reference baselines ═══════════════════
ROWS = [
    ("Majority class", "Majority class", META_GREY),
    ("Tweet length only", "Tweet length only", META_GREY),
    ("__surface__", "Surface + platform metadata (8)", "#0f766e"),
    (
        "Interpretable features (LightGBM)",
        "Interpretable linguistic features (43)",
        "#c2410c",
    ),
    (
        "Sentence-transformer embeddings (logistic)",
        "Sentence-transformer embeddings",
        "#6b8ec4",
    ),
    ("TF-IDF word 1-2gram (logistic)", "Word/bigram TF-IDF", "#6b8ec4"),
]


def figure3():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, (tag, title) in zip(axes, PANELS):
        B = V[tag + "_baselines"]
        y = np.arange(len(ROWS))[::-1]
        for yy, (k, lab, c) in zip(y, ROWS):
            if k == "__surface__":
                s = V[tag + "_surface"]
                v = s["balanced_accuracy"]
                e = [[v - s["balanced_accuracy_lo"]], [s["balanced_accuracy_hi"] - v]]
            else:
                v, e = B[k]["balanced_accuracy"], None
            ax.hlines(yy, 0.5, v, color=c, lw=1.0, alpha=0.55)
            ax.plot([v], [yy], "o", ms=5.5, color=c, zorder=3)
            if e:
                ax.errorbar(
                    [v], [yy], xerr=e, fmt="none", ecolor=c, elinewidth=0.9, capsize=1.8
                )
            ax.annotate(
                f"{v:.3f}",
                (v, yy),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                fontsize=6.5,
                color=c,
            )
        ax.axvline(0.5, color=META_GREY, lw=0.8, ls=(0, (4, 3)), zorder=0)
        ax.set_xlim(0.45, 1.06)
        ax.set_yticks(y)
        ax.set_title(title)
        ax.set_xlabel("Balanced accuracy")
    axes[0].set_yticklabels([lab for _, lab, _ in ROWS])
    for t, (_, _, c) in zip(axes[0].get_yticklabels(), ROWS):
        t.set_color(c if c != META_GREY else "#374151")
    fig.text(
        0.995,
        -0.02,
        "higher = better; dashed line = chance (0.5)",
        ha="right",
        va="top",
        fontsize=7,
        color=META_GREY,
    )
    fig.tight_layout()
    finish(fig, "figure3_baselines.png")


# ═══════════════════ Figure 4 — permutation importance ═══════════════════
NICE = {
    "gis": "Gist inference score",
    "DESPC": "Paragraph count",
    "DESSC": "Sentence count",
    "CoREF": "Coreference",
    "PCDC": "Deep cohesion",
    "WRDHYPnv": "Noun/verb hypernymy",
    "PCREF_1": "Referential cohesion (adjacent)",
    "PCREF_a": "Referential cohesion (all pairs)",
    "PCREF_1p": "Referential cohesion (adjacent, para.)",
    "PCREF_ap": "Referential cohesion (all pairs, para.)",
    "PCCNC_megahr": "Word concreteness (MegaHR)",
    "PCCNC_mrc": "Word concreteness (MRC)",
    "WRDIMGc_megahr": "Word imageability (MegaHR)",
    "WRDIMGc_mrc": "Word imageability (MRC)",
    "SMCAUSe_1": "Semantic verb overlap (adjacent)",
    "SMCAUSe_a": "Semantic verb overlap (all pairs)",
    "SMCAUSe_1p": "Semantic verb overlap (adjacent, para.)",
    "SMCAUSe_ap": "Semantic verb overlap (all pairs, para.)",
    "roberta_concreteness": "Model-based concreteness",
    "Sentiment": "Transformer sentiment",
    "valence": "Valence (lexicon)",
    "arousal": "Arousal (lexicon)",
    "dominance": "Dominance (lexicon)",
    "MaxRank": "Maximum word rank",
    "MedianRank": "Median word rank",
    "MedianContentRank": "Median content-word rank",
    "MinContentRank": "Minimum content-word rank",
}


def nice(f):
    if f in NICE:
        return NICE[f]
    if f.startswith("SMCAUSwn"):
        return "WordNet verb overlap (" + f.replace("SMCAUSwn_", "") + ")"
    return f


def figure4(top=12):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, (tag, title) in zip(axes, PANELS):
        d = pd.DataFrame(V[tag + "_perm"]).nlargest(top, "importance")[::-1]
        y = np.arange(len(d))
        ax.barh(y, d.importance, color="#4e79b6", height=0.68, zorder=2)
        ax.errorbar(
            d.importance,
            y,
            xerr=d.sd,
            fmt="none",
            ecolor="#1f2937",
            elinewidth=0.8,
            capsize=1.6,
            zorder=3,
        )
        ax.set_yticks(y)
        ax.set_yticklabels([nice(f) for f in d.feature])
        ax.set_xlabel("Drop in balanced accuracy when permuted")
        ax.set_title(title)
        ax.set_xlim(0, max(d.importance + d.sd) * 1.18)
    fig.text(
        0.995,
        -0.02,
        "bars = mean over 10 permutations; whiskers = SD",
        ha="right",
        va="top",
        fontsize=7,
        color=META_GREY,
    )
    fig.tight_layout()
    finish(fig, "figure4_importance.png")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
