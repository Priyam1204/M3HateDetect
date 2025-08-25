import base64
from openai import OpenAI
import sys
import os
import argparse
import pandas as pd
import glob
from pathlib import Path

current_script_dir = os.path.dirname(os.path.abspath(__file__))
two_dirs_up = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
sys.path.append(two_dirs_up)

# LLM Configuration - Hindi images with English prompts
API_KEY = "your api key here"
BASE_URL = "https://llm.tensor.rocks/v1"

def get_hindi_images():
    """Get all Hindi images from the data directory"""
    hindi_image_dir = "/home/utn/omul36yx/git/Multi3Hate/data/memes/hi"
    
    # Find all jpg files in the directory and subdirectories
    image_paths = []
    for root, dirs, files in os.walk(hindi_image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
    
    return sorted(image_paths)

def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def create_single_prompt(image_path, add_caption=False, unimodal=False):
    """Create single India country prompt for an image"""
    
    # Get image ID from filename
    id_image = os.path.basename(image_path).split(".jpg")[0].split(".jpeg")[0].split(".png")[0]
    
    # English prefix prompt (keep intact)
    prefix_prompt = """Definition of Hate Speech:

Any kind of communication [...], that attacks or uses pejorative or discriminatory language with reference to a person or a group on the basis of who they are, in other words, based on their religion, ethnicity, nationality, race, colour, descent, gender or other identity factor.

"""
    
    #country  cultural cues insertion prompt for India
    country_prompt = """Answer if this meme is hate or not in India, keeping in mind Indian sentiments and cultural sensitivities. Consider that Indians are generally offended by:
    - Content that mocks or disrespects religious beliefs, deities, or religious practices (Hinduism, Islam, Sikhism, Christianity, Buddhism, Jainism)
    - Derogatory remarks about caste, regional identities, or linguistic communities
    - Content that promotes communal hatred or targets specific ethnic groups
    - Disrespectful portrayals of national symbols, freedom fighters, or cultural traditions
    - Content that perpetuates harmful stereotypes about Indian culture, food, or customs
    - Material that mocks poverty, social issues, or undermines national pride
    - Gender-based discrimination or content that demeans women's safety and dignity
    - Content that trivializes serious social issues like farmer suicides, unemployment, or corruption
    
    Answer with a single alphabet letter among the given choices a and b."""
    
    # English postfix (keep intact)
    postfix_prompt = "\nChoose:\na: Hate\nb: Non-hate\n"
    
    # Combine prefix + country prompt
    full_prompt = prefix_prompt + country_prompt
    
    # Create message structure
    text_prompt_1 = {"type": "text", "text": full_prompt}
    text_prompt_2 = {"type": "text", "text": postfix_prompt}
    
    if unimodal:
        # Text-only mode (no image)
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
        # Image + text mode
        base64_image = encode_image(image_path)
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
    
    return messages, id_image

def create_LLM_client():
    """Create LLM client"""
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )
    return client

def call_LLM_api(messages, client):
    """Call LLM API with single prompt"""
    try:
        response = client.chat.completions.create(
            model="anthropic/claude-3.7-sonnet",
            messages=messages,
            max_tokens=5,  # Limited to force single letter response
            temperature=0,
        )
        
        response_content = response.choices[0].message.content.strip()
        original_prompt = messages[0]["content"][0]["text"] if messages and messages[0]["content"] else ""
        response_text = original_prompt + "\nAssistant:" + response_content
        
        print(f"LLM API call successful - Response: '{response_content}'")
        return response_text
        
    except Exception as e:
        print(f"LLM API call failed: {e}")
        return f"Error: {str(e)}"

def save_results(results, model_path):
    """Save results to CSV file"""
    
    # Create results directory
    results_dir = f"vlm/results/{model_path}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = os.path.join(results_dir, "responses_hi.csv")
    df.to_csv(output_file, index=False)
    
    print(f"Results saved to: {output_file}")
    return output_file

def main():
    print("Starting LLM Single Prompt Inference")
    print("=" * 50)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run LLM with single India country prompt')
    parser.add_argument('--model_path', type=str, required=False, default='claude_single_prompt')
    parser.add_argument('--caption', action='store_true', help='Enable captioning')
    parser.add_argument('--unimodal', action='store_true', help='Enable unimodal (text-only) mode')
    parser.add_argument('--max_images', type=int, default=None, help='Limit number of images to process')
    
    args = parser.parse_args()
    
    # Get Hindi images
    print("Loading Hindi images...")
    image_paths = get_hindi_images()
    
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    
    print(f"Found {len(image_paths)} Hindi images to process")
    
    # Display configuration
    print(f"\nConfiguration:")
    print(f"  Model: anthropic/claude-3.7-sonnet")
    print(f"  Images: {len(image_paths)} Hindi images")
    print(f"  Prompts per image: 1 (India country prompt)")
    print(f"  Total API calls: {len(image_paths)}")
    print(f"  Model path: {args.model_path}")
    print(f"  Caption: {args.caption}")
    print(f"  Unimodal: {args.unimodal}")
    
    # Display the prompt being used
    print("\nPrompt being used:")
    print("=" * 50)
    prefix = """Definition of Hate Speech:

Any kind of communication [...], that attacks or uses pejorative or discriminatory language with reference to a person or a group on the basis of who they are, in other words, based on their religion, ethnicity, nationality, race, colour, descent, gender or other identity factor.

"""
    country = """Answer if this meme is hate or not in India, keeping in mind Indian sentiments and cultural sensitivities. Consider that Indians are generally offended by:
    - Content that mocks or disrespects religious beliefs, deities, or religious practices (Hinduism, Islam, Sikhism, Christianity, Buddhism, Jainism)
    - Derogatory remarks about caste, regional identities, or linguistic communities
    - Content that promotes communal hatred or targets specific ethnic groups
    - Disrespectful portrayals of national symbols, freedom fighters, or cultural traditions
    - Content that perpetuates harmful stereotypes about Indian culture, food, or customs
    - Material that mocks poverty, social issues, or undermines national pride
    - Gender-based discrimination or content that demeans women's safety and dignity
    - Content that trivializes serious social issues like farmer suicides, unemployment, or corruption
    
    Answer with a single alphabet letter among the given choices a and b."""
    postfix = "\nChoose:\na: Hate\nb: Non-hate\n"
    
    print(prefix + country + postfix)
    print("=" * 50)
    print()
    
    # Create Claude client
    print("Creating LLM client...")
    client = create_LLM_client()
    
    # Process images
    results = []
    
    print(f"Processing {len(image_paths)} images...")
    
    for i, image_path in enumerate(image_paths):
        print(f"\nProcessing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        # Create single prompt
        messages, image_id = create_single_prompt(image_path, args.caption, args.unimodal)
        
        # Call Claude API
        response = call_LLM_api(messages, client)
        
        # Store result
        results.append({
            'ID': image_id,
            'prompt': 0,  # Only one prompt (index 0)
            'response': response
        })
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"Completed {i+1}/{len(image_paths)} images")
    
    # Save results
    print(f"\nSaving results...")
    output_file = save_results(results, args.model_path)
    
    print(f"\n{'='*50}")
    print("LLM SINGLE PROMPT INFERENCE COMPLETED")
    print(f"{'='*50}")
    print(f"Images processed: {len(results)}")
    print(f"Prompts per image: 1")
    print(f"Total API calls: {len(results)}")
    print(f"Results saved to: {output_file}")
    print(f"Ready for evaluation!")

if __name__ == '__main__':
    main()