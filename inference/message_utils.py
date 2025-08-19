import base64
import os
import json
from typing import Literal, List, Dict, Union
from openai.types.chat import ChatCompletionMessageParam

# --- Block TypedDicts ---
class TextBlock(Dict):
    text: str

class ImageBlock(Dict):
    image_path: str
    detail: Literal["low", "high", "auto"]

Block = Union[TextBlock, ImageBlock]


def make_text_block(text: str) -> Dict[str, str]:
    """Turn plain text into the required 'text' sub-block."""
    return {
        "type": "text",
        "text": text.strip()
    }


def make_image_file_block(
    image_path: str,
    detail: Literal["low", "high", "auto"] = "auto"
) -> Dict[str, Union[str, Dict[str, str]]]:
    """
    Turn a local image file into a base64 URI 'image_url' sub-block,
    including an optional detail hint.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"No such file: {image_path}")

    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    if ext not in {"png", "jpg", "jpeg", "gif"}:
        raise ValueError(f"Unsupported image format: .{ext}")

    raw = open(image_path, "rb").read()
    b64  = base64.b64encode(raw).decode("utf-8")
    uri  = f"data:image/{ext};base64,{b64}"

    return {
        "type": "image_url",
        "image_url": {
            "url": uri,
            "detail": detail
        }
    }


def prepare_multimodal_message(
    *,
    role: Literal["system", "user", "assistant"],
    blocks: List[Block]
) -> ChatCompletionMessageParam:
    """
    Build a ChatCompletionMessageParam from a role and an ordered list of blocks.
    Each block is either:
      - {"text": "..."}
      - {"image_path": "...", "detail": "low"|"high"|"auto"}
    """
    content_blocks = []
    for blk in blocks:
        if "text" in blk:
            content_blocks.append(make_text_block(blk["text"]))
        elif "image_path" in blk:
            # pull detail if provided, else default to "auto"
            detail = blk.get("detail", "auto")  # type: ignore[attr-defined]
            content_blocks.append(make_image_file_block(blk["image_path"], detail=detail))  # type: ignore[arg-type]
        else:
            raise ValueError(f"Block must have either 'text' or 'image_path': {blk!r}")

    return {
        "role": role,
        "content": content_blocks
    }


def save_message_to_file(
    message: Union[ChatCompletionMessageParam, List[ChatCompletionMessageParam]],
    file_path: str
) -> None:
    """
    Serialize a single message or list of messages to JSON and write to file.
    
    Args:
        message:   Either one ChatCompletionMessageParam dict, or a list of them.
        file_path: The path where the JSON will be written.
    """
    # Ensure the directory exists
    import os
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(
            message,
            f,
            ensure_ascii=False,
            indent=2
        )

# --- Usage Example ---
if __name__ == "__main__":
    blocks = [
        {"text": "Here's the first paragraph of description."},
        {"image_path": r"C:\Users\sayee\Documents\UTN_Sem2\Deep_learning_for_digital_humanities_and_social_sciences\Project_Multimodality\Code\Multi3Hate\data\memes\en\American-Pride-Eagle\8.jpg", "detail": "high"},
        {"text": "And here is some follow-up text."},
        {"image_path": r"C:\Users\sayee\Documents\UTN_Sem2\Deep_learning_for_digital_humanities_and_social_sciences\Project_Multimodality\Code\Multi3Hate\data\memes\en\Arabic-Meme\199.jpg"}  # detail defaults to "auto"
    ]

    msg = prepare_multimodal_message(role="user", blocks=blocks)
    save_message_to_file(msg, "test/message.json")
    
