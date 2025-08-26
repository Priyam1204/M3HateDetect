<div align="center">
  <img src="https://www.strichpunkt-design.de/storage/app/media/work/technische-universitaet-nuernberg-corporate-design-corporate-identity/technische-universitaet-nuernberg-corporate-design-corporate-identity-6-1920x1080__automuted--poster.jpg" alt="University of Technology Nuremberg" width="400"/>
</div>

# M3HateDetect: Multimodal Multilingual Hate Speech Detection

An extension of the [Multi3Hate framework](https://github.com/MinhDucBui/Multi3Hate/tree/main) developed as a final project for Deep Learning for Digital Humanities (SS25) at University of Technology Nuremberg (UTN), focusing on enhanced multimodal hate speech detection in multilingual memes using state-of-the-art Vision-Language Models (VLMs).

## 🌟 Features

- **Extended Multi3Hate Framework**: Built upon and enhanced the original Multi3Hate methodology by adding cultural cues to zero-shot prompting strategies
- **Advanced VLM Integration**: Incorporates latest Vision-Language Models including Claude 3.7 and GPT-4o
- **Cultural Context Awareness**: Enhanced prompting strategies with country-specific cultural sensitivities (IN/CN)
- **Multi-Prompt Strategy**: Comprehensive evaluation using 1, 6, and 20 prompt configurations
- **Multilingual Support**: Focuses on Hindi and Chinese meme analysis with cultural context understanding
- **Academic Project**: Final project for Deep Learning for Digital Humanities course at UTN

## 📁 Project Structure

```
M3HateDetect/
├── data/                           # Dataset and annotations
│   ├── memes/                      # Meme images organized by language
│   ├── final_annotations.csv       # Ground truth labels
│   └── raw_annotations.csv         # Individual annotator responses
├── vlm/                           # Vision-Language Model implementations
│   ├── 1-prompt/                  # Single prompt experiments
│   │   ├── inference/             # Model inference scripts
│   │   ├── evaluation/            # Model evaluation scripts
│   │   └── results/               # Model prediction outputs
│   ├── 6-prompt-experiment/       # 6-prompt experiments
│   │   ├── inference/             # Model inference scripts
│   │   ├── eval/                  # Evaluation scripts
│   │   └── results/               # Model prediction outputs
│   ├── 20-prompt/                 # 20-prompt experiments
│   │   ├── inference/             # Model inference scripts
│   │   ├── evaluation/            # Model evaluation scripts
│   │   └── results/               # Model prediction outputs
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- API keys for closed-source models (Claude, GPT-4V, Gemini)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/M3HateDetect.git
cd M3HateDetect
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```


## 💻 Usage

### Running Inference

####  1-Prompt
```bash
# Basic inference (all images)
python vlm/1-prompt/inference/llm_inference.py 
```
(This file works with both Claude and GPT-4o, you just need to change the model name)

#### 6-Prompt Strategy
```bash
# India (Hindi)
python vlm/6-prompt-experiment/inference/gpt_4o.py --caption --language hi

# China (Chinese) 
python vlm/6-prompt-experiment/inference/gpt_4o.py --caption --language zh
```

#### 20-Prompt Strategy
```bash
# Run inference
python vlm/20-prompt/inference/classify_memes_hate.py
```

### Model Evaluation

#### 1-Prompt Evaluation
```bash
# Evaluate Claude results
python vlm/1-prompt/evaluation/eval_claude_hindi.py --model_predictions path-to-folder
```

#### 6-Prompt Evaluation
```bash
# Step 1: Process responses (convert a/b to 0/1)
python vlm/6-prompt-experiment/eval/eval.py --language hi --input_file vlm/6-prompt-experiment/results/gpt_4o_caption_india_original/responses_hi_original.csv

# Step 2: Calculate final metrics (Accuracy, Precision, Recall)
python vlm/6-prompt-experiment/eval/evaluation_resutls.py --lang hi --pred vlm/6-prompt-experiment/results/gpt_4o_caption_india_original/processed_responses_hi.csv
```

#### 20-Prompt Evaluation
```bash
# Evaluate hate-speech classification
python vlm/20-prompt/evaluation/evaluation_IN/evaluate_hate_accuracy.py

python vlm/20-prompt/evaluation/evaluation_CN/evaluate_hate_accuracy.py

```


## 📊 Evaluation Metrics

The evaluation scripts provide comprehensive analysis including:

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Balanced performance measure
- **Confusion Matrix**: Detailed classification breakdown with recall and precision

## 🎯 Model Performance

| Culture | Model | # Prompts | Accuracy | Precision | Recall |
|---------|--------|-----------|----------|-----------|---------|
| IN | Claude 3.7 | 1 | 57.0% | 60.2% | 50.0% |
| IN | GPT-4o | 6 | 68.7% | 87.96% | 53.98% |
| IN | GPT-4o | 20 | 74.6% | 68.0% | 75.0% |
| IN | GPT-4o | 1 | 62.54% | 73.9% | 58.3% |
| CN | GPT-4o | 6 | 67.0% | 92.31% | 47.73% |
| CN | GPT-4o | 20 | 50.7% | 58.0% | 47.0% |

*Results on Hindi (IN) and Chinese (CN) meme datasets using different prompt strategies*

### Key Findings:
- **Culturally contextualized prompting improves alignment** with IN and CN annotations over baseline
- **Persona framing is the most impactful prompt component** for enhancing model performance
- **Bias reduction is partial** as we tested with larger dataset, but the models still tend to align with US norms


``

## 🙏 Acknowledgments

- Multi3Hate dataset creators
- Prof. Dr. Eren from University of Technology, Nuremeberg for API access
- The open-source community for various VLM implementations

## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: [priyammishra1204@gmail.com] or [priyam.mishra@utn.de] 

---

**Keywords**: Hate Speech Detection, Multimodal AI, Vision-Language Models, Multilingual NLP, Meme Analysis, Claude, GPT-4V
