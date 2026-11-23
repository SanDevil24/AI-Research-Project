"""
Method 2: Perplexity Curvature Detection
Project: Robust Detection for AI-Generated Social Media Text

Runs three zero-shot detectors on the same human/AI posts and reports
AUROC, Accuracy, Precision, Recall, Specificity and F1:
  - Log-likelihood (perplexity) baseline
  - DetectGPT (Mitchell et al., 2023): score post -> T5 rewordings -> curvature
  - Fast-DetectGPT (Bao et al., 2024): analytic curvature, one forward pass
Higher score = more likely AI. Threshold is chosen on the train split and
all metrics are reported on the held-out test split.

Setup:
    pip install torch transformers accelerate scikit-learn pandas matplotlib tqdm sentencepiece
Run (a CUDA GPU is strongly recommended):
    python method2_perplexity_curvature.py
Outputs: method2_results.csv, method2_scores.csv, method2_roc.png, method2_hist.png
"""
import matplotlib
matplotlib.use("Agg")   # save figures to files, no display needed
# ======================= CONFIG =======================
DATA_SOURCE    = "csv"           # "csv" = our team's file | "mage" = download MAGE test split
CSV_PATH       = "multisocial_english_10000_clean.csv"
TEXT_COL       = "text"
LABEL_COL      = "label"
AI_LABEL_VALUE = 1               # value in LABEL_COL that means AI-generated (can be a string, e.g. "ai")
                                 # NOTE: MAGE uses 0 = machine, 1 = human (handled automatically below)
LANG_COL       = None            # file is already English-only; set to "language" for raw MultiSocial
LANG_VALUE     = "en"            # keep only this language (scoring LMs are English-only)

N_PER_CLASS    = 5000            # 5000 human + 5000 AI, as on our slide
MIN_WORDS      = 1               # keep everything (shortest post is 3 words) so the test set matches the other methods
MAX_WORDS      = 200             # cut long texts (keeps T5 + LM inputs in range)
SEED           = 42
USE_EXISTING_SPLIT = True        # use the file's "split" column: train -> calibration, test -> test
                                 # (same 2,000 test posts the other methods are evaluated on)

# Scoring language model (used by all three detectors)
SCORING_MODEL  = "EleutherAI/gpt-neo-2.7B"   # if you run out of GPU memory: "gpt2-large" or "gpt2-medium"
MAX_TOKENS     = 256
BATCH_SIZE     = 8

# DetectGPT (perturbation) settings
RUN_DETECTGPT       = True
DETECTGPT_PER_CLASS = 250        # posts per class for DetectGPT (500 total); it's ~20x slower
N_PERTURBATIONS     = 20         # reworded versions per post (paper uses up to 100)
MASK_MODEL          = "t5-large" # T5 mask-filler that makes the rewordings
SPAN_LENGTH         = 2          # words per masked span
MASK_PCT            = 0.3        # fraction of words to reword
# ======================================================

import os, re, math, random, time
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device, "| GPU:", torch.cuda.get_device_name(0) if device == "cuda" else "none (this will be slow)")

# ---------- 1. Data collection & processing ----------
if DATA_SOURCE == "mage":
    raw = pd.read_csv("https://huggingface.co/datasets/yaful/MAGE/resolve/main/test.csv")
    TEXT_COL, LABEL_COL, AI_LABEL_VALUE = "text", "label", 0   # MAGE: 0 = machine-generated
else:
    raw = pd.read_csv(CSV_PATH)

print("Columns:", list(raw.columns))
print("Label values:\n", raw[LABEL_COL].value_counts().head(10))

df = raw.copy()
if LANG_COL:
    df = df[df[LANG_COL].astype(str).str.lower() == LANG_VALUE.lower()]
df = df.dropna(subset=[TEXT_COL, LABEL_COL])
df["text"] = df[TEXT_COL].astype(str).str.strip()
df = df[df["text"].str.split().str.len() >= MIN_WORDS]
df["text"] = df["text"].apply(lambda t: " ".join(t.split(" ")[:MAX_WORDS]))
df["y"] = (df[LABEL_COL].astype(str).str.strip().str.lower()
           == str(AI_LABEL_VALUE).strip().lower()).astype(int)   # 1 = AI, 0 = human
df = df.drop_duplicates("text")

assert df["y"].nunique() == 2, "Only one class found - check LABEL_COL / AI_LABEL_VALUE."
n = min(N_PER_CLASS, df["y"].value_counts().min())
df = (df.groupby("y", group_keys=False).sample(n=n, random_state=SEED)
        .reset_index(drop=True))
keep = ["text", "y"] + [c for c in ["split", "multi_label", "source", "src", LANG_COL] if c and c in df.columns]
df = df[keep].rename(columns={"split": "orig_split"})

if USE_EXISTING_SPLIT and "orig_split" in df.columns:
    df["split"] = np.where(df["orig_split"].astype(str).str.lower() == "test", "test", "cal")
    print("Using the file's own train/test split")
else:
    cal_idx, test_idx = train_test_split(df.index, test_size=0.5, stratify=df["y"], random_state=SEED)
    df["split"] = "test"; df.loc[cal_idx, "split"] = "cal"

print(f"\nUsing {len(df)} posts ({n} human + {n} AI)")
print(df.groupby(["split", "y"]).size().rename("count"))
print(df.sample(5, random_state=SEED)[["text", "y", "split"]].to_string())

# ---------- 2. Model scoring (log-likelihood + Fast-DetectGPT curvature) ----------
tok = AutoTokenizer.from_pretrained(SCORING_MODEL)
tok.pad_token = tok.eos_token
tok.padding_side = "right"
lm = AutoModelForCausalLM.from_pretrained(
    SCORING_MODEL, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()

@torch.no_grad()
def _score_batch(texts):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS).to(device)
    logits = lm(**enc).logits[:, :-1].float()
    labels = enc.input_ids[:, 1:]
    mask = enc.attention_mask[:, 1:].float()
    lprobs = torch.log_softmax(logits, dim=-1)
    probs = lprobs.exp()
    ll_tok = lprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)          # log p(actual token)
    mean_tok = (probs * lprobs).sum(-1)                                   # E[log p] over the model's own samples
    var_tok = (probs * lprobs.pow(2)).sum(-1) - mean_tok.pow(2)
    n_tok = mask.sum(1).clamp(min=1)
    ll_sum, mean_sum, var_sum = (ll_tok * mask).sum(1), (mean_tok * mask).sum(1), (var_tok * mask).sum(1)
    avg_ll = ll_sum / n_tok
    fast = (ll_sum - mean_sum) / var_sum.clamp(min=1e-6).sqrt()           # Fast-DetectGPT curvature
    return avg_ll.cpu().numpy(), fast.cpu().numpy()

def score_texts(texts, desc="Scoring"):
    # sort by length to minimise padding, then restore original order
    order = np.argsort([len(t) for t in texts])
    avg_ll, fast = np.zeros(len(texts)), np.zeros(len(texts))
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=desc):
        idx = order[i:i + BATCH_SIZE]
        a, f = _score_batch([texts[j] for j in idx])
        avg_ll[idx], fast[idx] = a, f
    return avg_ll, fast

t0 = time.time()
df["loglik"], df["fast_detectgpt"] = score_texts(df["text"].tolist(), "Fast-DetectGPT")
df["perplexity"] = np.exp(-df["loglik"])
df.to_csv("method2_scores.csv", index=False)
print(f"Done in {(time.time()-t0)/60:.1f} min")
print(df.groupby("y")[["perplexity", "fast_detectgpt"]].median().rename(index={0: "human", 1: "AI"}))

# ---------- 3. Text perturbation helpers (DetectGPT) ----------
FILL_PATTERN = re.compile(r"<extra_id_\d+>")

def mask_text(text, span=SPAN_LENGTH, pct=MASK_PCT):
    words = text.split(" ")
    n_spans = max(1, int(pct * len(words) / (span + 2)))
    n_masked, tries = 0, 0
    while n_masked < n_spans and tries < 1000:
        start = random.randint(0, max(0, len(words) - span))
        end = start + span
        if "<<<mask>>>" not in words[max(0, start - 1):min(len(words), end + 1)]:
            words[start:end] = ["<<<mask>>>"]
            n_masked += 1
        tries += 1
    k = 0
    for i, w in enumerate(words):
        if w == "<<<mask>>>":
            words[i] = f"<extra_id_{k}>"; k += 1
    return " ".join(words)

def extract_fills(decoded):
    decoded = [d.replace("<pad>", "").replace("</s>", "").strip() for d in decoded]
    return [[f.strip() for f in FILL_PATTERN.split(d)[1:-1]] for d in decoded]

def apply_fills(masked_texts, fills):
    out = []
    for m, f in zip(masked_texts, fills):
        words = m.split(" ")
        n = len(FILL_PATTERN.findall(m))
        if len(f) < n:
            out.append(None); continue          # T5 didn't fill every gap -> retry
        for i in range(n):
            words[words.index(f"<extra_id_{i}>")] = f[i]
        out.append(" ".join(words))
    return out

# ---------- 4. DetectGPT: perturb, re-score, compare curvature ----------
if RUN_DETECTGPT:
    t0 = time.time()
    sub = df.groupby(["split", "y"], group_keys=False).sample(
        n=min(DETECTGPT_PER_CLASS // 2, df.groupby(["split", "y"]).size().min()), random_state=SEED)

    t5_tok = AutoTokenizer.from_pretrained(MASK_MODEL, model_max_length=512)
    t5 = AutoModelForSeq2SeqLM.from_pretrained(MASK_MODEL).to(device).eval()   # fp32: T5 is unstable in fp16

    @torch.no_grad()
    def t5_fill(masked):
        n_max = max(len(FILL_PATTERN.findall(m)) for m in masked)
        stop_id = t5_tok.convert_tokens_to_ids(f"<extra_id_{n_max}>")
        enc = t5_tok(masked, return_tensors="pt", padding=True, truncation=True).to(device)
        out = t5.generate(**enc, max_length=150, do_sample=True, top_p=1.0, eos_token_id=stop_id)
        return extract_fills(t5_tok.batch_decode(out, skip_special_tokens=False))

    def perturb(texts, t5_batch=32, max_retries=5):
        results = [None] * len(texts)
        todo = list(range(len(texts)))
        for attempt in range(max_retries + 1):
            if not todo: break
            masked = {i: mask_text(texts[i]) for i in todo}
            for b in range(0, len(todo), t5_batch):
                ids = todo[b:b + t5_batch]
                m = [masked[i] for i in ids]
                for i, r in zip(ids, apply_fills(m, t5_fill(m))):
                    results[i] = r
            todo = [i for i in todo if results[i] is None]
        for i in todo:                       # rare: give up and keep the original
            results[i] = texts[i]
        return results

    sub_texts = sub["text"].tolist()
    expanded = [t for t in sub_texts for _ in range(N_PERTURBATIONS)]
    perturbed = []
    for b in tqdm(range(0, len(expanded), 256), desc="Perturbing with T5"):
        perturbed += perturb(expanded[b:b + 256])

    del t5; torch.cuda.empty_cache()
    pert_ll, _ = score_texts(perturbed, "Scoring perturbations")
    pert_ll = pert_ll.reshape(len(sub_texts), N_PERTURBATIONS)
    mu, sd = pert_ll.mean(1), pert_ll.std(1)
    sub["detectgpt"] = (sub["loglik"].values - mu) / np.where(sd > 1e-6, sd, 1e-6)   # normalized curvature

    df["detectgpt"] = np.nan
    df.loc[sub.index, "detectgpt"] = sub["detectgpt"]
    df.to_csv("method2_scores.csv", index=False)
    print(f"DetectGPT done on {len(sub)} posts in {(time.time()-t0)/60:.1f} min")
    print("Example rewording:\n  ORIGINAL :", sub_texts[0][:200], "\n  PERTURBED:", perturbed[0][:200])

# ---------- 5. Evaluation: threshold on calibration half, metrics on test half ----------
def evaluate(d, col, name):
    d = d.dropna(subset=[col])
    cal, te = d[d["split"] == "cal"], d[d["split"] == "test"]
    fpr, tpr, thr = roc_curve(cal["y"], cal[col])
    t = thr[np.argmax(tpr - fpr)]                    # Youden's J: best TPR - FPR trade-off
    pred = (te[col] >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(te["y"], pred, labels=[0, 1]).ravel()
    return {"Method": name, "Test posts": len(te),
            "AUROC": roc_auc_score(te["y"], te[col]),
            "Accuracy": accuracy_score(te["y"], pred),
            "Precision": precision_score(te["y"], pred, zero_division=0),
            "Recall": recall_score(te["y"], pred, zero_division=0),
            "Specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
            "F1": f1_score(te["y"], pred, zero_division=0),
            "Threshold": t}

rows = [evaluate(df, "loglik", "Log-likelihood baseline"),
        evaluate(df, "fast_detectgpt", "Fast-DetectGPT")]
if RUN_DETECTGPT:
    same = df[df["detectgpt"].notna()]
    rows += [evaluate(same, "detectgpt", "DetectGPT (subset)"),
             evaluate(same, "fast_detectgpt", "Fast-DetectGPT (same subset)"),
             evaluate(same, "loglik", "Log-likelihood (same subset)")]

results = pd.DataFrame(rows)
results.to_csv("method2_results.csv", index=False)
print(f"Scoring model: {SCORING_MODEL}")
print(results.round(3).to_string(index=False))

# ---------- 6. Figures for the slides ----------
methods = [("loglik", "Log-likelihood"), ("fast_detectgpt", "Fast-DetectGPT")]
if RUN_DETECTGPT: methods.append(("detectgpt", "DetectGPT (subset)"))

fig, ax = plt.subplots(figsize=(5.5, 5))
for col, name in methods:
    te = df[(df["split"] == "test") & df[col].notna()]
    fpr, tpr, _ = roc_curve(te["y"], te[col])
    ax.plot(fpr, tpr, label=f"{name} (AUROC = {roc_auc_score(te['y'], te[col]):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
ax.set_xlabel("False positive rate (human flagged as AI)"); ax.set_ylabel("True positive rate (AI caught)")
ax.set_title("Method 2: ROC on unseen test posts"); ax.legend(loc="lower right", fontsize=8)
plt.tight_layout(); plt.savefig("method2_roc.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(6.5, 4))
for y, name in [(0, "Human"), (1, "AI")]:
    ax.hist(df.loc[df["y"] == y, "fast_detectgpt"], bins=60, alpha=0.6, label=name, density=True)
t = results.loc[results["Method"] == "Fast-DetectGPT", "Threshold"].iloc[0]
ax.axvline(t, color="k", ls="--", lw=1, label=f"Threshold = {t:.2f}")
ax.set_xlabel("Fast-DetectGPT curvature score (higher = more AI-like)"); ax.set_ylabel("Density")
ax.set_title("Curvature scores: human vs AI posts"); ax.legend()
plt.tight_layout(); plt.savefig("method2_hist.png", dpi=200); plt.close()

print("\nSaved: method2_results.csv, method2_scores.csv, method2_roc.png, method2_hist.png")