import csv
import yaml
from pathlib import Path

from prompt_manager import PromptManager
from llm_inference_service import (
    AsyncOpenAIClientAdapter,
    LLMInferenceService,
    OpenAIClientAdapter,
    AllowedModelId,
)
from message_utils import prepare_multimodal_message

# ——— CONFIG ————————————————————————————————————————————————————————————
BASE_DIR     = Path(__file__).parent
DATA_ROOT    = BASE_DIR.parent / "data"
MEMES_ROOT   = DATA_ROOT / "memes"
CAPTIONS_CSV = DATA_ROOT / "captions" / "en.csv"       # change language file as needed
LANG         = CAPTIONS_CSV.stem                       # e.g. "en"
PROMPTS_DIR  = BASE_DIR / "prompts"
CFG_PATH     = PROMPTS_DIR / "config.yaml"
OUTPUT_CSV   = BASE_DIR / "llm_responses.csv"
# ——————————————————————————————————————————————————————————————————————

# Initialize services
pm      = PromptManager(str(PROMPTS_DIR), str(CFG_PATH))
service = LLMInferenceService(OpenAIClientAdapter(), AsyncOpenAIClientAdapter())

def load_captions(captions_file: Path) -> dict[str, str]:
    caps = {}
    with captions_file.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            img_num, group, caption = row[0].strip(), row[1].strip(), row[2].strip()
            image_id = f"{LANG}-{group}-{img_num}"
            caps[image_id] = caption
    return caps

def pick_one_per_group(memes_root: Path, lang: str=LANG) -> list[Path]:
    picks = []
    base = memes_root / lang
    if not base.exists():
        return picks
    for group_dir in sorted(base.iterdir()):
        if not group_dir.is_dir():
            continue
        for img in sorted(group_dir.iterdir()):
            if img.suffix.lower() in (".jpg", ".png"):
                picks.append(img)
                break
    return picks

def generate_image_id(image_path: Path) -> str:
    parts = list(image_path.parts)
    try:
        idx = parts.index("memes")
        lang, group = parts[idx+1], parts[idx+2]
    except ValueError:
        lang, group = parts[-3], parts[-2]
    return f"{lang}-{group}-{image_path.stem}"

def analyze_image(image_path: Path, caption: str, category: str) -> dict[str,str]:
    cfg = yaml.safe_load(CFG_PATH.read_text())
    if category not in cfg["prompts"]:
        raise KeyError(f"Category {category!r} not in config")
    out = {}
    for prompt_key in cfg["prompts"][category]:
        composite = f"{category}.{prompt_key}"
        system_prompt = pm.get(
            composite,
            country="China",
            dialect="Chinese",
            persona="a social activist",
        )

        # <-- FIXED: use keyword args here
        system_msg = prepare_multimodal_message(
            role="system",
            blocks=[{"text": system_prompt}],
        )
        user_msg = prepare_multimodal_message(
            role="user",
            blocks=[
                {"text": caption},
                {"image_path": str(image_path)}
            ],
        )

        resp = service.get_responses(
            [AllowedModelId.GPT_4O],
            messages=[system_msg, user_msg]
        )
        out[prompt_key] = resp[AllowedModelId.GPT_4O].content

    return out

def main():
    captions   = load_captions(CAPTIONS_CSV)
    images     = pick_one_per_group(MEMES_ROOT, LANG)
    cfg        = yaml.safe_load(CFG_PATH.read_text())
    categories = list(cfg["prompts"].keys())

    first = not OUTPUT_CSV.exists()
    with OUTPUT_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if first:
            writer.writerow(["ImageID","ModelID","Category","PromptID","Response"])

        for img in images:
            image_id = generate_image_id(img)
            caption  = captions.get(image_id, "<NO CAPTION>")
            model_id = AllowedModelId.GPT_4O.value

            for category in categories:
                responses = analyze_image(img, caption, category)
                for prompt_key, text in responses.items():
                    writer.writerow([
                        image_id,
                        model_id,
                        category,
                        prompt_key,
                        text
                    ])

    total = len(images) * len(categories)
    print(f"Done! Wrote ~{total} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
