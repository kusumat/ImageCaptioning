# ViT + GPT2 Image Captioning Project

## 📁 Project Structure

```
image_captioning/
├── data/
│   └── results.csv         # CSV file: image_name, comment
├── datasets/
│   └── flickr_dataset.py    # Dataset class with transform, padding, and tokenizer
├── models/
│   └── vit_gpt2.py           # ViT-GPT2 model integration
├── training/
│   ├── train.py              # Main training script
│   └── metrics.py            # BLEU, ROUGE, METEOR, Cosine similarity
├── utils/
│   ├── device.py            # CPU/GPU detection
│   ├── save.py              # Save best model and remove old ones
│   └── plot.py              # Plot training loss and metrics
├── inference/
│   └── infer.py             # Generate captions using trained model
└── README.md
```

---

## 🚀 Features

- **Transformer-based architecture**: Vision Transformer (ViT) encoder + GPT2 decoder
- **Full metrics**: BLEU, ROUGE, METEOR, Cosine Similarity
- **Cosine warmup scheduler**: With good initial learning rate and decay
- **Robust training loop**: Early stopping + best checkpoint saving
- **Device-aware**: GPU/MPS/CPU auto-selection
- **Clean visualization**: Matplotlib plots for all training stats
- **Modular design**: Easy to debug, extend or replace modules

---

## ⚖️ How to Run

### 1. Setup

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

- Place all Flickr30k images in `data/images/`
- Ensure `results.csv` is in `data/` with format:
  ```csv
  image_name,comment
  123.jpg,A child playing in the park.
  ```

### 3. Train Model

```bash
python training/train.py
```

### 4. Inference

```python
from inference.infer import generate_caption
caption = generate_caption("data/images/sample.jpg", "checkpoints/best_model.pt")
print(caption)
```

### 5. Plot Metrics

```bash
python utils/plot.py
```

---

## 🏆 Results

- Automatically saves:
  - `loss.png` and `metrics.png`
  - `checkpoints/best_model.pt`
- Cleaned intermediate model files

---

## 🔧 Customization

- **Hyperparameters** in `train.py`
- **Tokenizer/model** from HuggingFace (`vit-base-patch16-224`, `gpt2`)
- **Max caption length, padding** in `flickr_dataset.py`

---

## 🚜 Roadmap

-

---

## ✨ Credits

- Vision Transformer: [https://huggingface.co/google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)
- GPT2 Language Model: [https://huggingface.co/gpt2](https://huggingface.co/gpt2)
- Flickr30k Dataset: [https://shannon.cs.illinois.edu/DenotationGraph/](https://shannon.cs.illinois.edu/DenotationGraph/)

---

## 📢 Issues / Help?

Please open a GitHub issue or ping me with your questions!

