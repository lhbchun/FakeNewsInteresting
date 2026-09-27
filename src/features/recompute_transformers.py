"""Recompute the transformer-derived features.

sentiment_1..5 : softmax over the 5 logits of the 5-class BERT sentiment model
                 (nlptown/bert-base-multilingual-uncased-sentiment -- the 5-class
                 model the pipeline's local 'bert-senti' directory corresponds to)
roberta_concreteness : logits[0][0] of j-hartmann/concreteness-english-distilroberta-base

roberta_arousal / roberta_dominance cannot be recomputed: the model repositories
j-hartmann/arousal-english-distilbroberta-base and
j-hartmann/dominance-english-distilroberta-base are no longer publicly available.
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

torch.set_num_threads(10)
corp = pd.read_csv("corpus_labelled.csv")
texts = corp["text"].astype(str).tolist()
order = np.argsort([len(t) for t in texts])  # length-sorted batching


def score(model_id, n_out, batch=64):
    tok = AutoTokenizer.from_pretrained(model_id)
    mod = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
    print(model_id, "labels:", mod.config.num_labels, flush=True)
    out = np.zeros((len(texts), n_out), dtype=np.float32)
    with torch.inference_mode():
        for s in range(0, len(order), batch):
            idx = order[s : s + batch]
            enc = tok(
                [texts[i] for i in idx],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            lg = mod(**enc).logits
            if n_out == 5:
                lg = torch.softmax(lg, dim=-1)
                out[idx] = lg.numpy()
            else:
                out[idx, 0] = lg[:, 0].numpy()
            if s % (batch * 100) == 0:
                print(f"  {s}/{len(order)}", flush=True)
    return out


senti = score("nlptown/bert-base-multilingual-uncased-sentiment", 5)
conc = score("j-hartmann/concreteness-english-distilroberta-base", 1)

df = pd.DataFrame(senti, columns=[f"sentiment_{i + 1}" for i in range(5)])
df["roberta_concreteness"] = conc[:, 0]
df.to_csv("features_transformer.csv", index=False)
print("saved", df.shape, flush=True)
print(df.describe().round(4).to_string(), flush=True)
