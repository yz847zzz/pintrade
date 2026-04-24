"""
Quick test for FinBERT sentiment scoring.
Run from project root: python test_sentiment.py

Checks:
  1. GPU detection (should show RTX 4090 Ti)
  2. FinBERT model load (~420MB download on first run only)
  3. Sentiment scores on 5 financial sentences
  4. Timing benchmark
"""

import time
import torch
from transformers import pipeline

# ── 1. GPU check ──────────────────────────────────────────────────────────────
print("=" * 60)
print("DEVICE CHECK")
print("=" * 60)
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU:  {gpu_name}")
    print(f"  VRAM: {vram_gb:.1f} GB")
    device = 0
else:
    print("  No GPU found — running on CPU")
    device = -1
print()

# ── 2. Load FinBERT ───────────────────────────────────────────────────────────
print("=" * 60)
print("LOADING FinBERT  (ProsusAI/finbert, ~420MB first run)")
print("=" * 60)
t0 = time.time()
finbert = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    device=device,
    top_k=None,        # return all 3 class scores
    truncation=True,
    max_length=512,
)
print(f"  Loaded in {time.time() - t0:.1f}s")
print()

# ── 3. Test sentences ─────────────────────────────────────────────────────────
sentences = [
    "Apple reported record quarterly revenue of $89.5 billion, beating analyst estimates by 5%.",
    "The company faces significant headwinds from macroeconomic uncertainty and rising interest rates.",
    "Microsoft completed its $69 billion acquisition of Activision Blizzard.",
    "Revenue declined 8% year-over-year amid slowing consumer demand and supply chain disruptions.",
    "We are pleased to report strong growth across all business segments with improving margins.",
]

print("=" * 60)
print("SENTIMENT SCORES")
print("=" * 60)
t0 = time.time()
outputs = finbert(sentences)
elapsed = time.time() - t0

for sentence, output in zip(sentences, outputs):
    scores = {item["label"]: item["score"] for item in output}
    pos = scores.get("positive", 0)
    neg = scores.get("negative", 0)
    neu = scores.get("neutral",  0)
    compound = pos - neg   # our usable factor: range [-1, +1]

    # label with emoji for quick visual check
    if compound > 0.3:
        label = "POSITIVE"
    elif compound < -0.3:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL "

    print(f"  [{label}]  compound={compound:+.3f}  "
          f"pos={pos:.2f}  neg={neg:.2f}  neu={neu:.2f}")
    print(f"           {sentence[:75]}")
    print()

print(f"  Scored {len(sentences)} sentences in {elapsed*1000:.0f}ms  "
      f"({elapsed/len(sentences)*1000:.1f}ms each)")
print()

# ── 4. Batch timing benchmark ─────────────────────────────────────────────────
print("=" * 60)
print("BATCH BENCHMARK  (256 sentences, simulating a full 8-K)")
print("=" * 60)
batch = sentences * 52    # 260 sentences
t0 = time.time()
_ = finbert(batch, batch_size=256)
elapsed = time.time() - t0
print(f"  {len(batch)} sentences in {elapsed:.2f}s  "
      f"({elapsed/len(batch)*1000:.1f}ms each)")
print(f"  Estimated time for full 10-K (~1000 chunks): "
      f"{elapsed/len(batch)*1000:.0f}ms × 1000 = "
      f"{elapsed/len(batch)*1000*1000/1000:.1f}s")
