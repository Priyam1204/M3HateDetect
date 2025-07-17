import base64
from openai import OpenAI
import sys
import os
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

current_script_dir = os.path.dirname(os.path.abspath(__file__))
two_dirs_up = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
sys.path.append(two_dirs_up)
from vlm.inference.utils import pipeline_inference
from vlm.inference.culture_prompts import set_culture_prompts, set_prompts


LANGUAGES = ["en", "de", "es", "hi", "zh"]
# LANGUAGES = ["de"]
API_KEY = os.getenv("OPENAI_API_KEY")


def input_creator(all_prompts, image_paths, model_path, df_captions, add_caption, unimodal):
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Input for model_inference()
    processor = None
    processed_prompts = []
    for image_path in image_paths:
        base64_image = encode_image(image_path)
        for index, raw_prompt in enumerate(all_prompts):
            # Correct way to get meme ID from a file path like 'memes/en/Advicejew/222.jpg'
            filename = os.path.basename(image_path)  # '222.jpg'
            meme_id = os.path.splitext(filename)[0]  # '222'
            id_image = meme_id
            prompt_1 = raw_prompt[0]
            prompt_2 = raw_prompt[1]
            if add_caption:
                id_image = meme_id
                caption = df_captions[df_captions["ID"]
                                      == id_image]["Translation"].iloc[0]
                text_prompt_1 = {"type": "text", "text": prompt_1.format(str(caption))}
                text_prompt_2 = {"type": "text", "text": prompt_2.format(str(caption))}
            else:
                text_prompt_1 = {"type": "text", "text": prompt_1}
                text_prompt_2 = {"type": "text", "text": prompt_2}

            if unimodal:
                text_prompt_1["text"] = text_prompt_1["text"][:-7]
                messages = [
                    {
                        "role": "user",
                        "content": [
                            text_prompt_1,
                            text_prompt_2,
                        ],
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            text_prompt_1,
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                            text_prompt_2,
                        ],
                    }
                ]
            processed_prompts.append({"prompt": [messages, id_image + "_" + str(index)]})

    return processor, processed_prompts


def model_creator(model_path):
    api_key = API_KEY
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
    client = OpenAI(api_key=api_key)
    return client


def model_inference(prompt, model, processor, unimodal):
    # Extract the messages from the prompt structure
    # prompt structure: [[messages], id]
    if isinstance(prompt, list) and len(prompt) >= 1:
        messages = prompt[0]  # The messages array is the first element
        prompt_id = prompt[1] if len(prompt) > 1 else "unknown"  # ID is second element
    else:
        messages = prompt
        prompt_id = "unknown"
    
    try:
        response = model.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=messages,
            max_tokens=40,
            temperature=0,
        )
        response_text = messages[0]["content"][0]["text"] + "\nAssistant:" + response.choices[0].message.content
        return response_text
    except Exception as e:
        print(f"ERROR in model_inference: {e}")
        print(f"Messages that caused error: {messages}")
        raise


if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Run pipeline inference with specified model path.')

    # Add an argument for MODEL_PATH
    parser.add_argument('--model_path', type=str, required=False, default='gpt_4o/dont/matter')
    parser.add_argument('--caption', action='store_true', help='Enable captioning')
    parser.add_argument('--multilingual', action='store_true', help='Enable captioning')
    parser.add_argument('--country_insertion', action='store_true', help='Enable captioning')
    parser.add_argument('--unimodal', action='store_true', help='Enable captioning')
    parser.add_argument('--use_culture_prompts', action='store_true', help='Use culture-specific prompts for hi and zh')
    parser.add_argument('--language', type=str, choices=LANGUAGES + ["all"], default="all", help='Language to run inference on (or "all" for all languages)')
    args = parser.parse_args()

    if args.language == "all":
        languages_to_run = LANGUAGES
    else:
        languages_to_run = [args.language]

    # Process each language individually
    for language in languages_to_run:
        if args.use_culture_prompts and language in ["hi", "zh"]:
            all_prompts, prompt_caption, prompt_prefix, prompt_postfix, prompt_image_prefix = set_culture_prompts(language)
        else:
            all_prompts, prompt_caption, prompt_prefix, prompt_postfix, prompt_image_prefix = set_prompts(language)
        
        # Run pipeline inference for this specific language
        pipeline_inference(args.model_path, [language], input_creator, model_creator, model_inference, 
                          add_caption=args.caption, multilingual=args.multilingual, country_insertion=args.country_insertion,
                          unimodal=args.unimodal)