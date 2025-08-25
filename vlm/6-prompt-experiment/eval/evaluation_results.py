#!/usr/bin/env python3
"""
Evaluate LLM hate-speech predictions against human annotations, filtered by language.

Metrics
-------
* Accuracy  = (TP + TN) / (TP + FP + TN + FN)
* Precision =  TP / (TP + FP)
* Recall    =  TP / (TP + FN)

Usage
-----
python evaluate.py --lang zh \
  --raw data/raw_annotations.csv \
  --pred vlm/results/gpt_4o_caption/processed_responses_zh.csv \
  --save eval_zh.csv
"""

from pathlib import Path
import argparse
import sys
import re

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


# ----------------------------- helpers ---------------------------------------
def extract_id(path: str) -> int:
    """
    Extract numeric meme id from a path-like string.
    Examples:
      'memes/hi/.../222.jpg' -> 222
      '...\\memes\\zh\\...\\0156.png' -> 156
    """
    m = re.search(r"(\d+)(?:\.\w+)?$", str(path))
    if not m:
        raise ValueError(f"Could not extract numeric ID from: {path!r}")
    return int(m.group(1))


def extract_lang(path: str) -> str | None:
    """
    Extract language segment from a path like '.../memes/zh/...'.
    Returns lowercase language code or None if not found.
    """
    m = re.search(r"memes[\\/](\w+)[\\/]", str(path))
    return m.group(1).lower() if m else None


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ------------------------------- main ----------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM predictions against human annotations (language-filtered)."
    )
    parser.add_argument(
        "--lang", required=True, type=str,
        help="Language code to evaluate (e.g., en, hi, zh)"
    )
    parser.add_argument(
        "--raw", type=Path, default=Path("data/raw_annotations.csv"),
        help="Path to raw human annotations CSV"
    )
    parser.add_argument(
        "--pred", type=Path,
        default=Path("vlm/results/gpt_4o_caption/processed_responses_zh.csv"),
        help="Path to processed LLM predictions CSV"
    )
    parser.add_argument(
        "--save", type=Path, default=None,
        help="Optional path to save merged evaluation table as CSV"
    )
    args = parser.parse_args()
    expected_lang = args.lang.lower()

    # ---------------------- 1) Load CSVs -------------------------------------
    if not args.raw.exists():
        print(f"[ERROR] Ground-truth CSV not found: {args.raw}", file=sys.stderr)
        sys.exit(1)
    if not args.pred.exists():
        print(f"[ERROR] Predictions CSV not found: {args.pred}", file=sys.stderr)
        sys.exit(1)

    raw_df  = pd.read_csv(args.raw)
    pred_df = pd.read_csv(args.pred)

    # Sanity check required columns
    if "Meme ID" not in raw_df.columns or "hatespeech" not in raw_df.columns:
        print("[ERROR] raw_annotations.csv must contain columns: 'Meme ID', 'hatespeech'", file=sys.stderr)
        sys.exit(1)
    if "Meme ID" not in pred_df.columns or "hate_prediction" not in pred_df.columns:
        print("[ERROR] predictions CSV must contain columns: 'Meme ID', 'hate_prediction'", file=sys.stderr)
        sys.exit(1)

    # ---------------------- 1b) Optional GT language filter ------------------
    lang_col = first_existing_column(raw_df, ["language"])
    if lang_col:
        before = len(raw_df)
        raw_df = raw_df[raw_df[lang_col].astype(str).str.lower() == expected_lang]
        print(f"[GT] Filtered ground truth by {lang_col}='{expected_lang}': {before} -> {len(raw_df)} rows")
        if raw_df.empty:
            print("[ERROR] After language filter, ground-truth has 0 rows. "
                  "Check --lang or your GT CSV.", file=sys.stderr)
            sys.exit(1)
    else:
        print("[GT] No language column in ground-truth; using all GT rows.")

    # ---------------------- 2) Ground-truth one row per meme -----------------
    raw_df["Meme ID"] = raw_df["Meme ID"].astype(int)
    gt_df = (
        raw_df
        .groupby("Meme ID", as_index=False)["hatespeech"]
        .mean()       # average over annotators
        .round()      # majority vote
        .astype(int)
        .rename(columns={"hatespeech": "ground_truth"})
    )

    # ---------------------- 3) Predictions: enforce language -----------------
    pred_df = pred_df.copy()
    pred_df["meme_id"] = pred_df["Meme ID"].apply(extract_id)
    pred_df["lang"]    = pred_df["Meme ID"].apply(extract_lang)

    present_langs = pred_df["lang"].value_counts(dropna=False).to_dict()
    print(f"[PRED] Languages present in predictions: {present_langs}")

    before = len(pred_df)
    pred_df = pred_df[pred_df["lang"] == expected_lang]
    print(f"[PRED] Filtered predictions to lang='{expected_lang}': {before} -> {len(pred_df)} rows")
    if pred_df.empty:
        print("[ERROR] After language filter, predictions have 0 rows. "
              "Check --lang or your predictions CSV.", file=sys.stderr)
        sys.exit(1)

    pred_df["hate_prediction"] = pred_df["hate_prediction"].astype(int)

    # Majority vote across prompts -> one row per meme_id
    pred_df = (
        pred_df
        .groupby(["meme_id"], as_index=False)["hate_prediction"]
        .mean()
        .round()
        .astype(int)
    )

    # ---------------------- 4) Join on meme id -------------------------------
    eval_df = (
        gt_df
        .merge(pred_df, left_on="Meme ID", right_on="meme_id", how="inner")
        .drop(columns=["meme_id"])
    )

    if eval_df.empty:
        print("[ERROR] No overlapping Meme IDs between ground-truth and predictions "
              "after filtering. Check that both files refer to the same set.", file=sys.stderr)
        sys.exit(1)

    # ---------------------- 5) Metrics ---------------------------------------
    y_true = eval_df["ground_truth"]
    y_pred = eval_df["hate_prediction"]

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # ---------------------- 6) Report ----------------------------------------
    print("\n==================== Evaluation ====================")
    print(f"Language                 : {expected_lang}")
    print(f"Memes in GT (post-filter): {len(gt_df)}")
    print(f"Memes in Pred (post-agg) : {len(pred_df)}")
    print(f"Memes evaluated (joined) : {len(eval_df)}")
    print("----------------------------------------------------")
    print(f"Accuracy                 : {acc:.3%}")
    print(f"Precision (positive=1)   : {prec:.3%}")
    print(f"Recall    (positive=1)   : {rec:.3%}")
    print("\nConfusion matrix [rows=true, cols=pred]")
    print("          pred 0   pred 1")
    print(f"true 0 | {cm[0,0]:7d} {cm[0,1]:8d}")
    print(f"true 1 | {cm[1,0]:7d} {cm[1,1]:8d}")

    # ---------------------- 7) Optional save ---------------------------------
    if args.save:
        # include language column for traceability
        out = eval_df.copy()
        out.insert(1, "lang", expected_lang)
        out.to_csv(args.save, index=False)
        print(f"\n[INFO] Saved merged evaluation to: {args.save}")


if __name__ == "__main__":
    main()
