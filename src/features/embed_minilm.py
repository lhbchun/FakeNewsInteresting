"""Frozen sentence-transformer embeddings for the text-only baseline.

Not a fine-tuned model: mean-pooled all-MiniLM-L6-v2 representations fed to a
logistic regression, which gives a reference point for how much of the signal is
recoverable from surface text alone.
"""

import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel

M = "sentence-transformers/all-MiniLM-L6-v2"
corp = pd.read_csv("corpus_labelled.csv", usecols=["d_id", "text"])
tok = AutoTokenizer.from_pretrained(M)
mdl = AutoModel.from_pretrained(M).eval()
torch.set_num_threads(8)

out = []
B = 256
with torch.no_grad():
    for i in range(0, len(corp), B):
        t = corp.text.iloc[i : i + B].astype(str).tolist()
        enc = tok(t, padding=True, truncation=True, max_length=128, return_tensors="pt")
        h = mdl(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        emb = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        out.append(torch.nn.functional.normalize(emb, dim=1).numpy())
        if i % 5120 == 0:
            print(f"{i}/{len(corp)}", flush=True)

E = np.vstack(out).astype(np.float32)
df = pd.DataFrame(E, columns=[f"e{j}" for j in range(E.shape[1])])
df.insert(0, "d_id", corp.d_id.values)
df.to_parquet("embeddings_minilm.parquet", index=False)
print("saved", df.shape, flush=True)
