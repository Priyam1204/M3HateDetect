#!/usr/bin/env python
import sys
import csv
from pathlib import Path
import pandas as pd


# ─── allow inference/ imports ───────────────────────────────────────────────────
# Define project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Add inference/ to sys.path if you're using relative imports
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

# ─── 2. Initialize the LLM inference service ──────────────────────────────────────
service = LLMInferenceService(
    OpenAIClientAdapter(),
    AsyncOpenAIClientAdapter()
)

# ─── 3. Robust caption loader (handles commas in captions) ─────────────────────────
def load_captions(path: Path) -> dict[str,str]:
    caps = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            img_num = row[0].strip()
            group   = row[1].strip()
            # join the rest back into the caption
            caption = ",".join(cell.strip() for cell in row[2:])
            caps[f"en-{group}-{img_num}"] = caption
    return caps

# ─── 4. Pick one .jpg per group ───────────────────────────────────────────────────
def pick_one_per_group(root: Path, lang: str="en") -> list[Path]:
    picks = []
    base = root / lang
    for grp in sorted(base.iterdir()):
        if grp.is_dir():
            jpgs = sorted(grp.glob("*.jpg"))
            if jpgs:
                picks.append(jpgs[0])
    return picks

# ─── 5. Build ImageID from path ────────────────────────────────────────────────────
def gen_image_id(p: Path) -> str:
    parts = p.parts
    idx   = parts.index("memes")
    return f"{parts[idx+1]}-{parts[idx+2]}-{p.stem}"

# ─── 6. Define your two prompts ───────────────────────────────────────────────────
BASELINE_SYSTEM = [{"text":
    "You are a US-based content-moderation classifier. "
    "Classify this meme as exactly one of: Hate or Non-Hate. "
    "Respond with only that word."
}]
INDIAN_SYSTEM   = [{"text":
    "You are an Chinese internet user familiar with regional culture, politics, pop-culture, and internet slang. "
    "Look at the following meme (image and caption). Answer with exactly one word—either “Hate” or “Non-Hate”—"
    "based on whether this meme is offensive or hateful in the Chinese cultural context."
}]

# ─── 7. Classification helper ─────────────────────────────────────────────────────
def classify(img: Path, system_block) -> str:
    system_msg = prepare_multimodal_message(role="system", blocks=system_block)
    user_msg   = prepare_multimodal_message(
        role="user",
        blocks=[
            {"text": "Please classify this meme:"},
            {"image_path": str(img)}
        ]
    )
    resp = service.get_responses(
        [AllowedModelId.GPT_4O],
        messages=[system_msg, user_msg]
    )
    out = resp[AllowedModelId.GPT_4O].content.strip().lower()
    return "Hate" if out.startswith("hate") else "Non-Hate"

# ─── 8. Main entrypoint ───────────────────────────────────────────────────────────
def main():
    captions = load_captions(PROJECT_ROOT / "data" / "captions" / "en.csv")
    images   = pick_one_per_group(PROJECT_ROOT / "data" / "memes", "en")

    rows = []
    for img in images:
        img_id   = gen_image_id(img)
        caption  = captions.get(img_id, "<NO CAPTION>")
        base_lbl = classify(img, BASELINE_SYSTEM)
        ind_lbl  = classify(img, INDIAN_SYSTEM)
        rows.append((img_id, base_lbl, ind_lbl))
        print(f"{img_id:30} | {base_lbl:9} | {ind_lbl}")

    df = pd.DataFrame(rows, columns=["ImageID","US_Centric","Indian_Centric"])
    out = HERE / "evaluation_CN" / "results" / "baseline_vs_cultural.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved comparison CSV → {out}")

if __name__ == "__main__":
    main()
