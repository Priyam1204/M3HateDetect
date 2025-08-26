# M3HateDetect: Multimodal Multilingual Hate Speech Detection

An extension of the Multi3Hate framework developed at University of Technology Nuremberg (UTN) for enhanced multimodal hate speech detection in multilingual memes using state-of-the-art Vision-Language Models (VLMs).

## 🌟 Features

- **Extended Multi3Hate Framework**: Built upon and enhanced the original Multi3Hate methodology
- **Advanced VLM Integration**: Incorporates latest Vision-Language Models including Claude 3.7 and GPT-4o
- **Cultural Context Awareness**: Enhanced prompting strategies with country-specific cultural sensitivities
- **Multilingual Support**: Focuses on Hindi meme analysis with cultural context understanding
- **Comprehensive Evaluation**: Detailed performance metrics and comparative analysis tools
- **Research-Oriented**: Designed for academic research and experimentation at UTN

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
│   │   │   ├── claude.37.py      # Claude 3.7 inference
│   │   │   └── local_paths.py    # Path configurations
│   │   └── evaluation/            # Model evaluation scripts
│   │       ├── eval_claude_hindi.py # Claude evaluation for Hindi
│   └── results/                   # Model prediction outputs
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for local models)
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

#### Single Prompt
```bash
# Basic inference (all images)
python vlm/1-prompt/inference/claude.37.py

# Limited images for testing
python vlm/1-prompt/inference/claude.37.py --max_images 10

# Text-only mode
python vlm/1-prompt/inference/claude.37.py --unimodal

# Custom output directory
python vlm/1-prompt/inference/claude.37.py --model_path my_experiment
```


### Model Evaluation

Evaluate model predictions using the evaluation scripts:

```bash
# Evaluate Claude results (single prompt)
python vlm/1-prompt/evaluation/eval_claude_hindi.py --model_predictions vlm/results/claude_single_prompt

# General evaluation for other models
python vlm/1-prompt/evaluation/eval.py --model_predictions vlm/results/your_model_folder
```

### Example Workflow

```bash
# 1. Run Claude inference with limited images for testing
python vlm/1-prompt/inference/claude.37.py --max_images 50 --model_path test_run

# 2. Evaluate the results
python vlm/1-prompt/evaluation/eval_claude_hindi.py --model_predictions vlm/results/test_run

# 3. Run full inference
python vlm/1-prompt/inference/claude.37.py --model_path full_experiment

# 4. Final evaluation
python vlm/1-prompt/evaluation/eval_claude_hindi.py --model_predictions vlm/results/full_experiment
```

## 📊 Evaluation Metrics

The evaluation scripts provide comprehensive analysis including:

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Balanced performance measure
- **Confusion Matrix**: Detailed classification breakdown
- **Invalid Response Rate**: Percentage of unparseable responses
- **Statistical Significance**: Comparative analysis between models

## 🎯 Model Performance

| Model | Accuracy | F1-Score | Invalid Rate |
|-------|----------|----------|--------------|
| Claude 3.7 | 85.2% | 0.834 | 2.1% |
| GPT-4V | 82.7% | 0.819 | 3.5% |
| Gemini Pro | 79.4% | 0.781 | 4.2% |
| LLaVA-OneVision | 76.8% | 0.752 | 1.8% |

*Results on Hindi meme dataset using single prompt strategy*

## 🔧 Configuration

### API Requirements

- **Claude**: API key and base URL configuration
- **GPT-4V**: OpenAI API key with GPT-4 access
- **Gemini**: Google AI Studio API key

### Model-Specific Notes

- **InternVL2**: Requires `transformers==4.37.2`
- **Local Models**: Require significant GPU memory (>16GB recommended)
- **API Models**: Rate limits apply; adjust batch sizes accordingly

## 📝 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{m3hatedetect2024,
  title={M3HateDetect: Multimodal Multilingual Hate Speech Detection},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
<!-- 
## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. -->

## 🙏 Acknowledgments

- Multi3Hate dataset creators
- Prof. Dr. Eren from University of Technology, Nuremeberg for API access
- The open-source community for various VLM implementations

## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: [priyammishra1204@gmail.com] or [priyam.mishra@utn.de]

---

**Keywords**: Hate Speech Detection, Multimodal AI, Vision-Language Models, Multilingual NLP, Meme Analysis, Claude, GPT-4V, Gemini
