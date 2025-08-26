# evaluation/evaluate_bias_country.py

import sys
import csv
import json
import re
from pathlib import Path

# ——— CONFIG ——————————————————————————————————————————————————————————————
BASE_DIR       = Path(__file__).parent
RESULTS_DIR    = BASE_DIR / "evaluation_IN" / "results"
CSV_PATH       = RESULTS_DIR / "llm_responses.csv"
OUTPUT_JSON    = RESULTS_DIR / "bias_country_scores.json"
PLOT_PATH      = RESULTS_DIR / "bias_us_india.png"
MODEL_ID       = AllowedModelId.GPT_4O
# ——————————————————————————————————————————————————————————————————————


from llm_inference_service import (
    OpenAIClientAdapter,
    AsyncOpenAIClientAdapter,
    LLMInferenceService,
    AllowedModelId,
)
from message_utils import prepare_multimodal_message

import matplotlib.pyplot as plt
import numpy as np

# ——— CONFIG ——————————————————————————————————————————————————————————————
BASE_DIR       = Path(__file__).parent
CSV_PATH       = BASE_DIR.parent / "results" / "llm_responses.csv"
OUTPUT_JSON    = BASE_DIR / "bias_country_scores.json"
PLOT_PATH      = BASE_DIR / "relevance" / "bias_us_india.png"
MODEL_ID       = AllowedModelId.GPT_4O
# ——————————————————————————————————————————————————————————————————————

# Initialize LLM service
service = LLMInferenceService(
    OpenAIClientAdapter(),
    AsyncOpenAIClientAdapter()
)

BIAS_COUNTRY_PROMPT = '''
You are a cultural bias evaluator. Given the combined LLM responses for a meme image, compute two bias scores:

1) "USBiasScore": A number from 0 to 100 indicating how strongly the analysis is biased toward a US-centric perspective.
2) "IndiaBiasScore": A number from 0 to 100 indicating how strongly the analysis is biased toward an India-centric perspective.

Return valid JSON exactly like:
{
  "ImageID": "<image_id>",
  "USBiasScore": <0-100>,
  "IndiaBiasScore": <0-100>
}

Do not include any additional text or markdown.
'''.strip()


def load_responses_by_image():
    """Aggregate all responses per ImageID into a single text blob."""
    blobs = {}
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row['ImageID']
            text = row['Response'].replace('"', '\\"')
            blobs.setdefault(img, []).append(text)
    return {img: '\n'.join(lines) for img, lines in blobs.items()}


def call_llm_bias_country(image_id: str, text_blob: str) -> dict:
    """Send the bias-country prompt + responses to the LLM and parse JSON output."""
    system_msg = prepare_multimodal_message(
        role="system",
        blocks=[{"text": "You are a JSON-only evaluation engine."}]
    )
    user_msg = prepare_multimodal_message(
        role="user",
        blocks=[
            {"text": BIAS_COUNTRY_PROMPT.replace('<image_id>', image_id)},
            {"text": text_blob}
        ]
    )
    resp = service.get_responses([MODEL_ID], messages=[system_msg, user_msg])
    raw = (resp[MODEL_ID].content or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def main():
    blobs = load_responses_by_image()
    results = []

    for img, blob in blobs.items():
        print(f"Evaluating US vs India bias for {img}...")
        try:
            score = call_llm_bias_country(img, blob)
            results.append(score)
        except Exception as e:
            print(f"  Error for {img}: {e}")

    # Save JSON results
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Saved bias-country scores -> {OUTPUT_JSON}")

    # Compute overall averages
    if results:
        overall_us = sum(r['USBiasScore'] for r in results) / len(results)
        overall_in = sum(r['IndiaBiasScore'] for r in results) / len(results)
        print(f"Overall US Bias: {overall_us:.1f}%")
        print(f"Overall India Bias: {overall_in:.1f}%")
    else:
        print("No bias scores to average.")

    # Plot grouped bar chart with annotations
    image_ids = [r['ImageID'] for r in results]
    us_scores = [r['USBiasScore'] for r in results]
    in_scores = [r['IndiaBiasScore'] for r in results]

    x = np.arange(len(image_ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(image_ids)*0.4), 5))
    bars_us = ax.bar(x - width/2, us_scores, width, label='US Bias')
    bars_in = ax.bar(x + width/2, in_scores, width, label='India Bias')

        # No per-image percentage annotations
    # Instead include overall averages in the chart title
    if results:
        overall_us = sum(r['USBiasScore'] for r in results) / len(results)
        overall_in = sum(r['IndiaBiasScore'] for r in results) / len(results)
        ax.set_title(f"US vs India Bias per Image(Overall: US {overall_us:.1f}%, India {overall_in:.1f}%)",fontsize=14)
    else:
        ax.set_title("US vs India Bias per Image", fontsize=14)
    ax.legend()

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {PLOT_PATH}")

if __name__ == '__main__':
    main()
