# 6‑Prompt Hate‑Speech Detection (LLM) — Reproducibility Guide

This README explains **what the 6-prompts experiment does** and **how to fully reproduce** the results for **India (hi)** and **China (zh)** using the provided scripts.

---

## Overview

We evaluate GPT‑4o predictions for meme hate‑speech detection using two evaluation flows:

1. **`eval.py`** — compares per‑country prompt columns (e.g., `IN`) to `hate_prediction`. 
2. **`evaluation_resutls.py`** — computes standard classification metrics against the human ground truth for each experiment.

Inference is done with:

- **`6-prompt-experiment\inference\gpt_4o.py`** — runs GPT‑4o caption inference with a language flag.

> **Important:** The evaluation that aggregates per meme (`evaluation_resutls.py`) **requires** the ground‑truth file `data/raw_annotations.csv` and a processed predictions file like `processed_responses_<lang>.csv`, where `<lang>` is `hi` or `zh`.

---

---

## Data & Repo Structure

Expected key files/folders (paths shown in Windows style to match the commands below):

```
data\
  captions\
  memes\
  raw_annotations.csv                # human annotations; many annotators per meme

6-prompt-experiment\
  inference\
    gpt_4o.py                        # inference entrypoint
  results\
    gpt_4o_caption_india_original\
      responses_hi_original.csv
      processed_responses_hi.csv
    gpt_4o_caption_india_ours\
      processed_responses_hi.csv
    gpt_4o_caption_china_original\
      responses_zh_original.csv
      processed_responses_zh.csv
    gpt_4o_caption_china_ours\
      processed_responses_zh.csv
  eval\
    eval.py                          # per-country prompt vs prediction
    evaluation_resutls.py            # language-filtered, GT vs aggregated preds
```

> Filenames/folders for China (`zh`) follow the same pattern as India (`hi`). Adjust to match your actual output locations.

---

## 1) Inference

Run GPT‑4o caption inference with the `--language` switch:

### India (Hindi)
```bash
python 6-prompt-experiment\inference\gpt_4o.py --caption --language hi
```

### China (Chinese)
```bash
python 6-prompt-experiment\inference\gpt_4o.py --caption --language zh
```

This should produce CSVs under `6-prompt-experiment\results\...` (e.g., `responses_<lang>_original.csv`).

---

## 2) Evaluation Path A — `eval.py` (per‑country prompt columns)

This script should create `6-prompt-experiment\results\` (e.g., `processed_responses_<lang>.csv`). it will convert the prediction from a and b choices to 0 and 1 so that we can run `evaluation_resutls.py` to get the Accuracy and Precision and Recall

### Example: India (Hindi)

```bash
python 6-prompt-experiment\eval\eval.py --language hi --input_file 6-prompt-experiment\results\gpt_4o_caption_india_original\responses_hi_original.csv
```


---

## 3) Evaluation Path B — `evaluation_resutls.py` (language‑filtered, GT vs aggregated preds)

This script:
- Filters predictions by `--lang` using the path segment (e.g., `memes\hi\...\222.jpg` → `hi`).
- Aggregates multiple prompts per meme (majority vote).
- Joins with `data/raw_annotations.csv` (majority vote across annotators).
- Computes Accuracy, Precision, Recall, and shows the confusion matrix.

### India (Hindi) — Original prompts
```bash
python 6-prompt-experiment\eval\evaluation_resutls.py --lang hi --pred 6-prompt-experiment\results\gpt_4o_caption_india_original\processed_responses_hi.csv
```
**Observed output**
```
==================== Evaluation ====================
Language                 : hi
Memes in GT (post-filter): 300
Memes in Pred (post-agg) : 300
Memes evaluated (joined) : 300
----------------------------------------------------
Accuracy                 : 64.000%
Precision (positive=1)   : 90.476%
Recall    (positive=1)   : 43.182%

Confusion matrix [rows=true, cols=pred]
          pred 0   pred 1
true 0 |     116        8
true 1 |     100       76
```

### India (Hindi) — “Ours” prompts
```bash
python 6-prompt-experiment\eval\evaluation_resutls.py --lang hi --pred 6-prompt-experiment\results\gpt_4o_caption_india_ours\processed_responses_hi.csv
```
**Observed output**
```
==================== Evaluation ====================
Language                 : hi
Memes in GT (post-filter): 300
Memes in Pred (post-agg) : 300
Memes evaluated (joined) : 300
----------------------------------------------------
Accuracy                 : 68.667%
Precision (positive=1)   : 87.963%
Recall    (positive=1)   : 53.977%

Confusion matrix [rows=true, cols=pred]
          pred 0   pred 1
true 0 |     111       13
true 1 |      81       95
```

### China (Chinese)

Replace `hi` with `zh` and point `--pred` to the corresponding processed file:
```bash
python 6-prompt-experiment\eval\evaluation_resutls.py --lang zh --pred 6-prompt-experiment\results\gpt_4o_caption_china_original\processed_responses_zh.csv
# ...or for your “ours” prompts:
python 6-prompt-experiment\eval\evaluation_resutls.py --lang zh --pred 6-prompt-experiment\results\gpt_4o_caption_china_ours\processed_responses_zh.csv
```

