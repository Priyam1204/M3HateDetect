# Multi3Hate Meme Evaluation Toolkit

An end-to-end toolkit for evaluating hate-speech classification of memes using multimodal LLMs. Supports:

- **Standard US-centric** prompts  
- **Culture-aware** prompts (Indian-centric, China-centric)  
- **Bias analysis** between perspectives  
- Metric computation (accuracy, precision, recall, F1)  
- Visualization (bar charts, confusion matrices)
- Prompt Relevance Scoring




This project builds on the methods and data described in *[Your Paper Title Here]* and includes the full dataset and code used for experiments.

---

## Repository Structure

```text
Multi3Hate/
├── data/                          
│   ├── memes/                     
│   │   └── en/…             ← all your meme images
│   └── captions/           ← your CSV captions (e.g. en.csv)
├── inference/             ← your core multimodal LLM pipeline
│   ├── llm_inference_service.py
│   ├── prompt_manager.py
│   ├── message_utils.py
│   └── meme_analysis.py
├── results/               ← where raw LLM outputs land
│   ├── llm_responses_CN.csv
│   └── llm_responses_IN.csv
└── evaluation/            ← all your evaluation scripts & outputs
    ├── evaluate_hate_accuracy.py
    ├── compare_prompts.py
    ├── evaluation_relevance.py
    ├── evaluation_CN/  
    │   └── evaluate_bias.py
    └── evaluation_IN/
```

---

## Quick Start

### 1. Install dependencies
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Download or clone this repo
```bash
git clone https://github.com/YOUR_USERNAME/Multi3Hate.git
cd Multi3Hate
```

### 3. Prepare the dataset
All meme images and captions live under `data/`.  
You can use the provided dataset directly—no extra downloads required.

### 4. Run inference
```bash
python inference/classify_memes_hate.py
```
Outputs → `results/llm_responses.csv`

### 5. Evaluate hate-speech classification
```bash
python evaluation/evaluate_hate_accuracy.py
```
- Merges ground-truth (`data/captions/en.csv`) with LLM labels  
- Computes accuracy, precision, recall, F1  
- Saves `evaluation/relevance.csv` and chart images

### 6. Compare prompt strategies
```bash
python evaluation/compare_prompts.py
```
- Compares US-centric vs. culture-aware (Indian- or China-centric) prompts  
- Saves `evaluation/baseline_vs_cultural.csv`

### 7. Bias analysis (US vs. India and China)
```bash
# Run bias analysis (example logic, ensure correct script if available)
# This is typically embedded within evaluate_bias.py
```
- Aggregates LLM responses per image  
- Computes USBiasScore & ChinaBiasScore via LLM  
- Saves JSON and `evaluation/evaluation_CN/bias_us_china.png` plot

### 8. Category relevance
```bash
# This module provides pie charts of category relevance per image
```
- Visualizes how relevant each image is to specific categories and subprompts

---

## Dataset & Paper

Dataset included: All memes and captions used from  
**“Multi3Hate: Multimodal, Multilingual, and Multicultural Hate Speech Detection with Vision–Language Models”**
Github link:
```bash
https://github.com/MinhDucBui/Multi3Hate
```
Git clone: 
  ```bash
https://github.com/MinhDucBui/Multi3Hate.git
```
---

## Contributing

1. Fork the repo  
2. Create a feature branch  
   ```bash
   git checkout -b feature/xyz
   ```
3. Commit your changes  
   ```bash
   git commit -m "Add xyz"
   ```
4. Push and open a Pull Request  
   ```bash
   git push origin feature/xyz
   ```

Please follow PEP-8 style and include docstrings or tests for new code.
