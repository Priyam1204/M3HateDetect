from pathlib import Path

OUTPUT_DIR = Path("evaluation_IN/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Ensures the directory exists

import matplotlib.pyplot as plt

# Manually specified metric values
metrics = {
    'Accuracy': 0.73,
    'Precision': 0.68,
    'Recall': 0.75,
    'F1 Score': 0.71
}

# 1) Plot classification metrics as a bar chart
fig, ax = plt.subplots()
ax.bar(metrics.keys(), metrics.values())
ax.set_ylim(0, 1)
ax.set_ylabel('Score')
ax.set_title('Hate‐Speech Classification Metrics chatgpt4o')
for i, (label, value) in enumerate(metrics.items()):
    ax.text(i, value + 0.02, f'{value:.2f}', ha='center')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'metrics_bar_chart.png')

plt.show()

# Manually specified confusion matrix counts
cm_counts = {
    'True Negative': 18,
    'False Positive': 12,
    'False Negative': 10,
    'True Positive': 25
}

# 2) Plot confusion‐matrix counts as a bar chart
fig, ax = plt.subplots()
ax.bar(cm_counts.keys(), cm_counts.values())
ax.set_ylabel('Count')
ax.set_title('Confusion Matrix Counts')
for i, (label, count) in enumerate(cm_counts.items()):
    ax.text(i, count + 0.5, str(count), ha='center')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_counts_bar_chart.png')

plt.show()
