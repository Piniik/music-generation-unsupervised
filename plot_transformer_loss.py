import json
from pathlib import Path

import matplotlib.pyplot as plt

history_path = Path("checkpoints/transformer_smoke_history.json")
out_loss_path = Path("outputs/plots/transformer_loss_curve.png")
out_ppl_path = Path("outputs/plots/transformer_perplexity_curve.png")

with open(history_path, "r", encoding="utf-8") as f:
    history = json.load(f)

epochs = history["epoch"]
train_loss = history["train_loss"]
val_loss = history["val_loss"]
train_ppl = history["train_perplexity"]
val_ppl = history["val_perplexity"]

out_loss_path.parent.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, marker="o", label="Train Loss")
plt.plot(epochs, val_loss, marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Task 3 Transformer Training Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_loss_path, dpi=200)
print(f"Saved loss plot to: {out_loss_path}")

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_ppl, marker="o", label="Train Perplexity")
plt.plot(epochs, val_ppl, marker="o", label="Validation Perplexity")
plt.xlabel("Epoch")
plt.ylabel("Perplexity")
plt.title("Task 3 Transformer Perplexity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_ppl_path, dpi=200)
print(f"Saved perplexity plot to: {out_ppl_path}")