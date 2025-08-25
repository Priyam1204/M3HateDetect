import argparse
import re
from scipy import stats
from sklearn.metrics import f1_score
from scipy.stats import ranksums
import numpy as np
from tqdm import tqdm
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

current_script_dir = os.path.dirname(os.path.abspath(__file__))
one_dir_up = os.path.abspath(os.path.join(current_script_dir, '..'))
sys.path.append(one_dir_up)
from inference.local_paths import ANNOTATION_PATH

np.seterr(divide='ignore', invalid='ignore')
tqdm.pandas()

# Getting only Hindi language 
LANGUAGES = ["hi"]
MAPPING = {
    "hi": "IN"  # Hindi maps to India (IN)
}

def extract_llm_answer(response):
    """Extract the core response from LLM output"""
    response = response.lower()
    prefixes = [
        "assistant: ", "assistant:", "assistant\n", "assistant", 
        "answer:", "the answer is ", "response:", "result:", "my answer is"
    ]

    for prefix in prefixes:
        if prefix in response:
            response = response.split(prefix)[-1]
    return response.strip()

def extract_answer(response):
    """Extract 'a' or 'b' from LLM response"""
    response = extract_llm_answer(response)
    response = response.lower().strip()

    # Look for explicit patterns first
    if "answer: a" in response or "a:" in response or response.endswith("a"):
        return "a"
    if "answer: b" in response or "b:" in response or response.endswith("b"):
        return "b"
    
    #for single letter responses
    if response == "a" or response == "b":
        return response
    
    # Look for patterns like "my answer is: a" or "based on... a"
    if re.search(r'\ba\b', response):
        return "a"
    elif re.search(r'\bb\b', response):
        return "b"
    
    # Get first character as last resort
    first_part = response.split(" ")[0].split("\n")[0].strip()
    if (first_part in ['a', 'b']):
        return first_part
    
    return first_part

def mapping_response(response):
    """Map LLM response to 0 (chose a) or 1 (chose b) or -1 (invalid)"""
    extracted = extract_answer(response)
    
    if extracted == "a":
        return 0  # Chose 'a'
    elif extracted == "b":
        return 1  # Chose 'b'
    else:
        print(f"Invalid LLM response: ---{response[:200]}--- \n Extracted: ----{extracted}----")
        return -1  # Invalid response

def process_response_to_hatespeech(row):
    """Convert LLM response to hate speech prediction"""
    # Invalid response
    if row['processed_answer'] == -1:
        return -1

    # For your LLM script, you used:
    # a: Hate, b: Non-hate
    # So 'a' = hate (1), 'b' = non-hate (0)
    
    if row['processed_answer'] == 0:  # Chose 'a'
        return 1  # Hate
    elif row['processed_answer'] == 1:  # Chose 'b' 
        return 0  # Non-hate
    else:
        return -1

def calc_acc_single_prompt(df, gt_name, predict_name):
    """Calculate accuracy for single prompt scenario"""
    # Filter out invalid responses (-1) before calculating accuracy
    valid_df = df[df[predict_name] != -1]
    
    if len(valid_df) == 0:
        print("ERROR: No valid responses found!")
        return 0.0, 0.0
    
    # Since we only have 1 prompt, calculate simple accuracy
    correct_predictions = sum(valid_df[gt_name] == valid_df[predict_name])
    accuracy = (correct_predictions / len(valid_df)) * 100
    
    # No standard deviation needed for single prompt
    return round(accuracy, 2), 0.0

def calc_acc_multiple_prompts(df, gt_name, predict_name):
    """Calculate accuracy for multiple prompts scenario"""
    # Filter out invalid responses (-1) before calculating accuracy
    valid_df = df[df[predict_name] != -1]
    
    if len(valid_df) == 0:
        print("ERROR: No valid responses found!")
        return 0.0, 0.0, pd.DataFrame()
    
    # Calculate accuracy by prompt
    accuracy_by_prompt = []
    for prompt_num in valid_df['prompt'].unique():
        prompt_df = valid_df[valid_df['prompt'] == prompt_num]
        if len(prompt_df) > 0:
            correct = sum(prompt_df[gt_name] == prompt_df[predict_name])
            accuracy = correct / len(prompt_df)
            accuracy_by_prompt.append({'prompt': prompt_num, 'accuracy': accuracy})
    
    df_acc = pd.DataFrame(accuracy_by_prompt)
    
    mean_accuracy = round(df_acc['accuracy'].mean() * 100, 2)
    std_accuracy = round(np.std(df_acc['accuracy'], ddof=1) * 100, 2)

    return mean_accuracy, std_accuracy, df_acc

def detailed_analysis(df):
    """Provide detailed analysis of LLM results"""
    print("\n" + "="*50)
    print("DETAILED LLM ANALYSIS")
    print("="*50)
    
    # Total responses
    total_responses = len(df)
    print(f"Total responses: {total_responses}")
    
    # Invalid responses
    invalid_responses = sum(df['hate_prediction'] == -1)
    print(f"Invalid responses: {invalid_responses} ({invalid_responses/total_responses*100:.1f}%)")
    
    # Valid responses breakdown
    valid_df = df[df['hate_prediction'] != -1]
    if len(valid_df) > 0:
        hate_predictions = sum(valid_df['hate_prediction'] == 1)
        nonhate_predictions = sum(valid_df['hate_prediction'] == 0)
        print(f"Hate predictions: {hate_predictions}")
        print(f"Non-hate predictions: {nonhate_predictions}")
        
        # Check if ground truth exists
        if 'IN' in valid_df.columns and not valid_df['IN'].isna().all():
            # Ground truth breakdown
            actual_hate = sum(valid_df['IN'] == 1)
            actual_nonhate = sum(valid_df['IN'] == 0)
            print(f"Actual hate in dataset: {actual_hate}")
            print(f"Actual non-hate in dataset: {actual_nonhate}")
            
            # Confusion matrix
            true_positives = sum((valid_df['hate_prediction'] == 1) & (valid_df['IN'] == 1))
            false_positives = sum((valid_df['hate_prediction'] == 1) & (valid_df['IN'] == 0))
            true_negatives = sum((valid_df['hate_prediction'] == 0) & (valid_df['IN'] == 0))
            false_negatives = sum((valid_df['hate_prediction'] == 0) & (valid_df['IN'] == 1))
            
            print(f"\nConfusion Matrix:")
            print(f"True Positives (Correctly identified hate): {true_positives}")
            print(f"False Positives (Incorrectly identified as hate): {false_positives}")
            print(f"True Negatives (Correctly identified non-hate): {true_negatives}")
            print(f"False Negatives (Missed hate): {false_negatives}")
            
            # Precision, Recall, F1
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\nPrecision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"F1-Score: {f1:.3f}")
        else:
            print("No ground truth available for detailed metrics")

def plot_confusion_matrix(df, output_folder):
    """Plot and save confusion matrix for LLM results with better colors"""
    print(f"DEBUG: plot_confusion_matrix called with output_folder: {output_folder}")
    print(f"DEBUG: DataFrame shape: {df.shape}")
    
    # Filter valid responses and those with ground truth
    valid_df = df[(df['hate_prediction'] != -1) & (~df['IN'].isna())]
    
    print(f"DEBUG: Valid responses with ground truth: {len(valid_df)}")
    
    if len(valid_df) == 0:
        print("No valid responses with ground truth for confusion matrix")
        return
    
    # Check if output folder exists, create if not
    if not os.path.exists(output_folder):
        print(f"Creating output folder: {output_folder}")
        os.makedirs(output_folder, exist_ok=True)
    
    # Get true and predicted labels
    y_true = valid_df['IN'].values
    y_pred = valid_df['hate_prediction'].values
    
    print(f"DEBUG: y_true shape: {y_true.shape}, unique values: {np.unique(y_true)}")
    print(f"DEBUG: y_pred shape: {y_pred.shape}, unique values: {np.unique(y_pred)}")
    
    #confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"DEBUG: Confusion matrix shape: {cm.shape}")
    print(f"DEBUG: Confusion matrix:\n{cm}")
    
    # Create the plot with better colors for presentations
    plt.figure(figsize=(10, 8), facecolor='white')
    
    #Professional blue-red colormap
    sns.heatmap(cm, annot=True, fmt='d', 
                cmap='RdYlBu_r',  # Red-Yellow-Blue reversed (red for high values)
                xticklabels=['Non-Hate', 'Hate'],
                yticklabels=['Non-Hate', 'Hate'],
                cbar_kws={'label': 'Count'},
                annot_kws={'size': 16, 'weight': 'bold'},
                linewidths=2, linecolor='white')
    
    plt.title('Confusion Matrix - LLM Hindi Hate Speech Detection', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    
    # Add counts as text with better positioning
    total = np.sum(cm)
    plt.figtext(0.02, 0.02, f'Total Samples: {total}', 
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    # Save the plot
    plot_path = os.path.join(output_folder, 'confusion_matrix_claude_hindi.png')
    print(f"DEBUG: Attempting to save plot to: {plot_path}")
    
    try:
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        print(f"✓ Confusion matrix saved to: {plot_path}")
        
        # Verify file was created
        if os.path.exists(plot_path):
            file_size = os.path.getsize(plot_path)
            print(f"✓ File confirmed: {plot_path} ({file_size} bytes)")
        else:
            print(f"✗ ERROR: File was not created: {plot_path}")
            
    except Exception as e:
        print(f"✗ ERROR saving plot: {e}")
        return
    
    # Create normalized confusion matrix with better colors
    try:
        cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
        
        plt.figure(figsize=(10, 8), facecolor='white')
        
        #Professional green colormap for normalized matrix
        sns.heatmap(cm_normalized, annot=True, fmt='.1%', 
                    cmap='Greens',  # Green colormap
                    xticklabels=['Non-Hate', 'Hate'],
                    yticklabels=['Non-Hate', 'Hate'],
                    cbar_kws={'label': 'Percentage'},
                    annot_kws={'size': 16, 'weight': 'bold'},
                    linewidths=2, linecolor='white',
                    vmin=0, vmax=1)
        
        plt.title('Normalized Confusion Matrix - LLM Hindi Hate Speech Detection', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=14, fontweight='bold')
        
        # Save normalized plot
        norm_plot_path = os.path.join(output_folder, 'confusion_matrix_normalized_claude_hindi.png')
        print(f"DEBUG: Attempting to save normalized plot to: {norm_plot_path}")
        
        plt.tight_layout()
        plt.savefig(norm_plot_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✓ Normalized confusion matrix saved to: {norm_plot_path}")
        
        # Verify file was created
        if os.path.exists(norm_plot_path):
            file_size = os.path.getsize(norm_plot_path)
            print(f"✓ File confirmed: {norm_plot_path} ({file_size} bytes)")
        else:
            print(f"✗ ERROR: File was not created: {norm_plot_path}")
            
    except Exception as e:
        print(f"✗ ERROR saving normalized plot: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate LLM results on Hindi hate speech detection.')

    parser.add_argument('--model_predictions', type=str, required=False,
                        default='vlm/results/claude_3_7')
    args = parser.parse_args()
    
    model_predictions_folder = args.model_predictions
    
    # Load ground truth
    df_gt = pd.read_csv(ANNOTATION_PATH)
    df_gt = df_gt.reset_index()
    df_gt["Meme ID"] = df_gt["Meme ID"].astype(str)
    
    print("="*60)
    print("LLM HINDI HATE SPEECH EVALUATION")
    print("="*60)
    print(f"Model predictions folder: {model_predictions_folder}")
    print(f"Ground truth file: {ANNOTATION_PATH}")
    print()

    # Look for Hindi response file
    response_file = os.path.join(model_predictions_folder, "responses_hi.csv")
    
    if not os.path.exists(response_file):
        print(f"ERROR: Could not find {response_file}")
        print("Make sure you have run LLM inference first.")
        sys.exit(1)
    
    print(f"Processing: {response_file}")
    
    # Load LLM results
    df_inference = pd.read_csv(response_file)
    print(f"Loaded {len(df_inference)} responses")
    
    # Process responses
    df_inference.rename(columns={'ID': 'Meme ID'}, inplace=True)
    df_inference['answer'] = df_inference['response'].apply(extract_answer)
    df_inference['processed_answer'] = df_inference['response'].apply(mapping_response)
    df_inference['hate_prediction'] = df_inference.apply(process_response_to_hatespeech, axis=1)
    
    # Truncate response for readability
    df_inference['response_truncated'] = df_inference["response"].str.replace(
        "\n", " ").str.replace("assistant", "").str[-100:]
    
    # Check unique meme IDs in results
    unique_meme_ids = df_inference['Meme ID'].unique()
    print(f"Unique meme IDs in results: {len(unique_meme_ids)}")
    
    # Save processed results BEFORE merging
    df_processed = df_inference[["Meme ID", "prompt", "response_truncated", "answer", "hate_prediction"]]
    output_path = os.path.join(model_predictions_folder, "processed_responses_hi.csv")
    df_processed.to_csv(output_path, index=False)
    print(f"Processed results saved to: {output_path}")
    
    # Merge with ground truth and check for missing IDs
    df_inference["Meme ID"] = df_inference["Meme ID"].astype(str)
    
    # Check which meme IDs are in ground truth
    gt_meme_ids = set(df_gt['Meme ID'].unique())
    result_meme_ids = set(df_inference['Meme ID'].unique())
    
    missing_in_gt = result_meme_ids - gt_meme_ids
    missing_in_results = gt_meme_ids - result_meme_ids
    
    print(f"\nMeme ID Analysis:")
    print(f"Meme IDs in ground truth: {len(gt_meme_ids)}")
    print(f"Meme IDs in results: {len(result_meme_ids)}")
    print(f"Meme IDs in results but missing in ground truth: {len(missing_in_gt)}")
    print(f"Meme IDs in ground truth but missing in results: {len(missing_in_results)}")
    
    # Merge with ground truth (left join to keep all results)
    df_merged = pd.merge(df_inference, df_gt, on="Meme ID", how="left")
    
    # Check if we have ground truth data
    has_ground_truth = 'IN' in df_merged.columns and not df_merged['IN'].isna().all()
    
    if has_ground_truth:
        # Calculate accuracy only on rows with ground truth
        df_with_gt = df_merged[~df_merged['IN'].isna()]
        print(f"\nRows with ground truth labels: {len(df_with_gt)}")
        
        # Use the merged dataframe with ground truth for accuracy calculation
        df_for_accuracy = df_with_gt
    else:
        print("\nWARNING: No ground truth labels found!")
        df_for_accuracy = df_merged
    
    # Calculate accuracy
    print("\n" + "="*30)
    print("ACCURACY RESULTS")
    print("="*30)
    
    # Filter valid responses for accuracy calculation
    valid_responses = df_for_accuracy[df_for_accuracy['hate_prediction'] != -1]
    total_responses = len(df_inference)
    valid_count = len(valid_responses)
    
    print(f"Total responses: {total_responses}")
    print(f"Valid responses: {valid_count}")
    print(f"Invalid responses: {total_responses - valid_count}")
    
    # Check if we have multiple prompts or just one
    unique_prompts = df_inference['prompt'].unique()
    print(f"Number of unique prompts: {len(unique_prompts)} (prompts: {sorted(unique_prompts)})")
    
    if valid_count > 0 and has_ground_truth:
        if len(unique_prompts) == 1:
            # Single prompt - use simplified calculation
            print("\nSingle prompt detected - using simplified accuracy calculation")
            accuracy, _ = calc_acc_single_prompt(df_for_accuracy, "IN", "hate_prediction")
            print(f"Overall Accuracy: {accuracy}% (on {valid_count} valid responses)")
            
        else:
            # Multiple prompts - use group-based calculation
            print(f"\nMultiple prompts detected - using group-based accuracy calculation")
            accuracy, std, df_acc = calc_acc_multiple_prompts(df_for_accuracy, "IN", "hate_prediction")
            print(f"Overall Accuracy: {accuracy}% (on {valid_count} valid responses)")
            print(f"Standard Deviation: {std}%")
            
            # Show per-prompt accuracy
            print("\nPer-prompt accuracy:")
            for idx, row in df_acc.iterrows():
                print(f"  Prompt {int(row['prompt'])}: {row['accuracy']*100:.1f}%")
    
    elif valid_count > 0:
        print("\nNo ground truth available - showing response distribution only")
        accuracy = 0.0
    else:
        print("ERROR: No valid responses found!")
        accuracy = 0.0
    
    # Detailed analysis
    detailed_analysis(df_merged)
    
    # Plot confusion matrix if we have ground truth
    if has_ground_truth and valid_count > 0:
        plot_confusion_matrix(df_merged, model_predictions_folder)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    
    
    # Sample of results
    print("\nSample results (first 10 responses):")
    sample_cols = ['Meme ID', 'prompt', 'answer', 'hate_prediction']
    if has_ground_truth:
        sample_cols.append('IN')
    sample_df = df_merged[sample_cols].head(10)
    print(sample_df.to_string(index=False))
    
    print(f"\nFINAL SUMMARY:")
    print(f"LLM Hindi Hate Speech Detection")
    print(f"Total images processed: {len(df_inference['Meme ID'].unique())}")
    print(f"Prompts used: {len(unique_prompts)}")
    print(f"Valid responses: {valid_count}/{total_responses}")
    if valid_count > 0 and has_ground_truth:
        print(f"Final Accuracy: {accuracy}%")
