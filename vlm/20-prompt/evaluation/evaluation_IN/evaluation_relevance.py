# evaluation/evaluate_all.py

import sys
import csv
import json
import re
from pathlib import Path

# ─── allow inference/ imports ───────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent.resolve()
INFERENCE_DIR = PROJECT_ROOT / "inference"
sys.path.insert(0, str(INFERENCE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
# ──────────────────────────────────────────────────────────────────────────────

from llm_inference_service import (
    OpenAIClientAdapter,
    AsyncOpenAIClientAdapter,
    LLMInferenceService,
    AllowedModelId,
)
from message_utils import prepare_multimodal_message

import matplotlib.pyplot as plt

# ——— CONFIG ——————————————————————————————————————————————————————————————
BASE_DIR   = Path(__file__).parent
CSV_PATH   = BASE_DIR.parent / "results" / "llm_responses.csv"
PLOTS_DIR  = BASE_DIR / "relevance"
JSON_DIR   = BASE_DIR / "relevance_json"

PLOTS_DIR.mkdir(exist_ok=True)
JSON_DIR.mkdir(exist_ok=True)

MODEL_ID   = AllowedModelId.GPT_4O
# ——————————————————————————————————————————————————————————————————————

# Initialize LLM service\
service = LLMInferenceService(
    OpenAIClientAdapter(),
    AsyncOpenAIClientAdapter()
)

# Fully‐escaped JSON prompt template
EVAL_INSTRUCTIONS = """
You are an expert evaluation engine. You will receive a CSV snippet for one image,
columns: ImageID, ModelID, Category, PromptID, Response.

Produce JSON exactly like this (no markdown, no extra text):

{{
  "ImageID": "{image_id}",
  "CategoryRelevance": {{ "CoreCulturalUnderstanding": 20.0, "SensitivityModeration": 30.0, "RegionalLinguisticAdaptation": 10.0, "HumorViralDynamics": 25.0, "AdvancedInsightEmpathy": 15.0 }},
  "SubpromptRelevance": {{
    "CoreCulturalUnderstanding": {{ "general": 10.0, "india": 5.0, "china": 3.0, "regionalInterpretation": 1.0, "humorType": 1.0 }},
    "SensitivityModeration": {{ "tabooDetection": 10.0, "censorshipPrediction": 10.0, "moderationThreshold": 5.0, "communitySentiment": 5.0 }},
    "RegionalLinguisticAdaptation": {{ "iconography": 4.0, "dialectRendering": 3.0, "formatAppropriateness": 2.0, "localizationSuggestion": 1.0 }},
    "HumorViralDynamics": {{ "politicalAlignment": 10.0, "formatFit": 5.0, "emotionalImpact": 5.0, "crossCulturalRisk": 5.0 }},
    "AdvancedInsightEmpathy": {{ "audienceAppropriateness": 10.0, "biasMitigation": 3.0, "politicalAlignmentAdvanced": 2.0 }}
  }}
}}
"""

def load_csv_for_image(image_id: str) -> str:
    """Return header + all rows for the given image_id as CSV text."""
    lines = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        lines.append(",".join(header))
        for row in reader:
            if row and row[0] == image_id:
                escaped = [c.replace('"','""') for c in row]
                lines.append(",".join(f'"{c}"' for c in escaped))
    return "\n".join(lines)


def call_llm_eval(image_id: str, csv_snippet: str) -> dict:
    """Send the evaluation prompt + CSV to the LLM and parse its JSON output."""
    system_msg = prepare_multimodal_message(
        role="system",
        blocks=[{"text": "You are a JSON-only evaluation engine."}]
    )
    user_msg = prepare_multimodal_message(
        role="user",
        blocks=[
            {"text": EVAL_INSTRUCTIONS.format(image_id=image_id)},
            {"text": csv_snippet}
        ]
    )
    resp = service.get_responses([MODEL_ID], messages=[system_msg, user_msg])
    raw = (resp[MODEL_ID].content or "").strip()
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


def plot_all_categories(image_id: str, eval_json: dict):
    """Generate and save one combined pie-grid for the given eval JSON."""
    cat_rel = eval_json.get("CategoryRelevance", {})
    sub_rel = eval_json.get("SubpromptRelevance", {})
    cats = list(cat_rel.items())

    n = len(cats)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
    axes = axes.flatten()

    fig.suptitle(f"Relevance Breakdown: {image_id}", fontsize=18)
    fig.subplots_adjust(hspace=0.6, wspace=1.2)

    for ax, (cat, pct) in zip(axes, cats):
        subs = sub_rel.get(cat, {})
        labels = list(subs.keys())
        sizes  = list(subs.values())
        if sizes and sum(sizes) > 0:
            ax.pie(
                sizes,
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
                labeldistance=0.85,
                textprops={"fontsize":8}
            )
        else:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', fontsize=10, color='gray')
        ax.set_title(f"{cat} ({pct:.1f}%)", fontsize=14)
        ax.axis("equal")

    for ax in axes[n:]:
        ax.axis('off')

    out_file = PLOTS_DIR / f"{image_id}.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


def main():
    # Gather unique image IDs from the CSV
    image_ids = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row and row[0] not in image_ids:
                image_ids.append(row[0])

    # In-memory storage of all eval JSONs for overall mean
    evals = []

    # Process each image
    for img in image_ids:
        print("Processing", img)
        snippet = load_csv_for_image(img)
        if snippet.count("\n") <= 1:
            print("  no data, skipping")
            continue
        ej = call_llm_eval(img, snippet)
        evals.append(ej)

        # Save individual JSON for future use
        with open(JSON_DIR / f"{img}.json", "w", encoding="utf-8") as jf:
            json.dump(ej, jf, indent=2)

        plot_all_categories(img, ej)

    # Compute and plot overall mean
    if evals:
        print("\nComputing overall mean relevance across all images...")
        cats = evals[0]["CategoryRelevance"].keys()
        overall_cat = {
            cat: sum(e["CategoryRelevance"][cat] for e in evals) / len(evals)
            for cat in cats
        }
        overall_sub = {}
        for cat in cats:
            subs_keys = evals[0]["SubpromptRelevance"][cat].keys()
            overall_sub[cat] = {
                sub: sum(e["SubpromptRelevance"][cat].get(sub, 0) for e in evals) / len(evals)
                for sub in subs_keys
            }
        overall_json = {
            "ImageID": "overall_mean",
            "CategoryRelevance": overall_cat,
            "SubpromptRelevance": overall_sub
        }
        # Save overall mean JSON
        with open(JSON_DIR / "overall_mean.json", "w", encoding="utf-8") as jf:
            json.dump(overall_json, jf, indent=2)

        plot_all_categories("overall_mean", overall_json)
    else:
        print("No successful evaluations to aggregate.")

if __name__ == "__main__":
    main()
