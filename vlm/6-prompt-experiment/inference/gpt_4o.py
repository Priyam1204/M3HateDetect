import base64
from openai import OpenAI
import sys
import os
import argparse
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

current_script_dir = os.path.dirname(os.path.abspath(__file__))
two_dirs_up = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
sys.path.append(two_dirs_up)
from vlm.inference.utils import pipeline_inference
from vlm.inference.all_prompts import set_prompts


LANGUAGES = ["en", "de", "es", "hi", "zh"]
# LANGUAGES = ["de"]
API_KEY = os.getenv("PROXY_API_KEY")  # Use proxy key instead of OpenAI key
BASE_URL = os.getenv("OPENAI_BASE_URL")  # Add this line to get the custom base URL

# Cost estimation constants
PROMPT_NUMBER = 6  # From utils.py
GPT4O_INPUT_COST_PER_1M = 5.00  # $5.00 per 1M input tokens
GPT4O_OUTPUT_COST_PER_1M = 15.00  # $15.00 per 1M output tokens
ESTIMATED_INPUT_TOKENS_PER_CALL = 800  # Prompt + image description
ESTIMATED_OUTPUT_TOKENS_PER_CALL = 50  # Short response


def estimate_cost(language, add_caption=False):
    """Estimate the cost for processing a specific language"""
    try:
        # Load annotation data
        df = pd.read_csv('data/final_annotations.csv')
        
        # Map language to column
        lang_to_col = {'en': 'US', 'de': 'DE', 'es': 'MX', 'zh': 'CN', 'hi': 'IN'}
        col = lang_to_col.get(language)
        
        if not col:
            return None, "Language not found in annotation data"
        
        # Count images for this language
        if language in ['hi', 'zh']:
            # Hindi and Chinese have 300 images each
            image_count = 300
        else:
            image_count = df[col].sum()
        
        # Calculate API calls
        api_calls = image_count * PROMPT_NUMBER
        
        # Calculate tokens
        input_tokens = api_calls * ESTIMATED_INPUT_TOKENS_PER_CALL
        output_tokens = api_calls * ESTIMATED_OUTPUT_TOKENS_PER_CALL
        
        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * GPT4O_INPUT_COST_PER_1M
        output_cost = (output_tokens / 1_000_000) * GPT4O_OUTPUT_COST_PER_1M
        total_cost = input_cost + output_cost
        
        return {
            'language': language,
            'image_count': image_count,
            'api_calls': api_calls,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost
        }, None
        
    except Exception as e:
        return None, f"Error estimating cost: {str(e)}"


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
                caption = df_captions[df_captions["Meme ID"]
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
        raise ValueError("PROXY_API_KEY not found in environment variables. Please check your .env file.")
    # Use custom base_url if provided
    if BASE_URL:
        client = OpenAI(api_key=api_key, base_url=BASE_URL)
    else:
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
            model="openai/gpt-4o",
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
    parser.add_argument('--language', type=str, choices=LANGUAGES + ["all"], default="all", help='Language to run inference on (or "all" for all languages)')
    args = parser.parse_args()

    if args.language == "all":
        languages_to_run = LANGUAGES
    else:
        languages_to_run = [args.language]

    # Estimate costs for all languages to be processed
    print("=" * 60)
    print("COST ESTIMATION")
    print("=" * 60)
    
    total_cost = 0
    cost_details = []
    
    for language in languages_to_run:
        cost_info, error = estimate_cost(language, args.caption)
        if error:
            print(f"Error estimating cost for {language}: {error}")
            continue
            
        cost_details.append(cost_info)
        total_cost += cost_info['total_cost']
        
        print(f"\n{language.upper()} Language:")
        print(f"  Images: {cost_info['image_count']}")
        print(f"  API calls: {cost_info['api_calls']:,}")
        print(f"  Input tokens: {cost_info['input_tokens']:,}")
        print(f"  Output tokens: {cost_info['output_tokens']:,}")
        print(f"  Input cost: ${cost_info['input_cost']:.4f}")
        print(f"  Output cost: ${cost_info['output_cost']:.4f}")
        print(f"  Total cost: ${cost_info['total_cost']:.4f}")
    
    print(f"\n" + "=" * 60)
    print(f"TOTAL ESTIMATED COST: ${total_cost:.4f}")
    print("=" * 60)
    
    # Ask for confirmation
    print(f"\nDo you want to proceed with processing? (y/n): ", end="")
    response = input().lower().strip()
    
    if response not in ['y', 'yes']:
        print("Processing cancelled.")
        exit(0)
    
    print("\nStarting processing...")
    print("=" * 60)

    # Process each language individually
    for language in languages_to_run:
        all_prompts, prompt_caption, prompt_prefix, prompt_postfix, prompt_image_prefix, prompts_country_insertion = set_prompts(language)
        
        # Run pipeline inference for this specific language
        pipeline_inference(args.model_path, [language], input_creator, model_creator, model_inference, 
                          add_caption=args.caption, multilingual=args.multilingual, country_insertion=args.country_insertion,
                          unimodal=args.unimodal)