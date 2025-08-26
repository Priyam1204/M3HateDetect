from pathlib import Path
import matplotlib.pyplot as plt

# 1. Where to save
OUTPUT_DIR = Path("evaluation_CN/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Your manually chosen metric values for the China‐aware prompt
metrics = {
    'Accuracy': 0.522,
    'Precision': 0.58,
    'Recall': 0.47,
    'F1 Score': 0.52
}

# 3. Plot the classification metrics
fig, ax = plt.subplots()
ax.bar(metrics.keys(), metrics.values())
ax.set_ylim(0, 1)
ax.set_ylabel('Score')
ax.set_title('Hate-Speech Classification Metrics (China-Aware Prompt)')
for i, (label, value) in enumerate(metrics.items()):
    ax.text(i, value + 0.02, f'{value*100:.1f}%', ha='center')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'china_metrics_bar_chart.png')
plt.show()

# 4. Example confusion‐matrix counts for the China-aware run
cm_counts = {
    'True Negative': 11,
    'False Positive': 16,
    'False Negative': 12,
    'True Positive': 9
}

# 5. Plot the confusion counts
fig, ax = plt.subplots()
ax.bar(cm_counts.keys(), cm_counts.values())
ax.set_ylabel('Count')
ax.set_title('Confusion Matrix Counts (China-Aware Prompt)')
for i, (label, count) in enumerate(cm_counts.items()):
    ax.text(i, count + 0.5, str(count), ha='center')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'china_confusion_counts_bar_chart.png')
plt.show()
