"""
=============================================================
 Sentiment Analysis on IMDB using BERT (HuggingFace + PyTorch)
=============================================================
 Class example: Fine-tuning a pre-trained Transformer on the
 IMDB movie review dataset (binary: positive / negative).

 Requirements:
   pip install transformers datasets torch scikit-learn

 Tested with:
   transformers==4.40, datasets==2.19, torch==2.2
=============================================================
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup)
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score

import numpy as np
import random

# ── 1. Reproducibility ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 2. Config — tweak these for your class ────────────────────────────────────
MODEL_NAME   = "bert-base-uncased"   # pre-trained checkpoint to fine-tune
MAX_LEN      = 256                   # max token length (IMDB reviews can be long)
BATCH_SIZE   = 16                    # reduce to 8 if GPU memory is tight
EPOCHS       = 3                     # 2–3 epochs is usually enough for fine-tuning
LR           = 2e-5                  # learning rate (standard for BERT fine-tuning)
TRAIN_SUBSET = 2000                  # use a subset for quick class demos; set None for full
TEST_SUBSET  = 500

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = device = torch.device("mps")
print(f"Using device: {DEVICE}\n")

# ── 3. Load IMDB dataset ──────────────────────────────────────────────────────
# HuggingFace datasets: 25,000 train / 25,000 test reviews, balanced pos/neg
print("Loading IMDB dataset...")
dataset = load_dataset("imdb")

# Optional: take a subset for faster demos in class
train_data = dataset["train"].shuffle(seed=SEED)
test_data  = dataset["test"].shuffle(seed=SEED)

if TRAIN_SUBSET:
    train_data = train_data.select(range(TRAIN_SUBSET))
if TEST_SUBSET:
    test_data = test_data.select(range(TEST_SUBSET))

print(f"Train samples : {len(train_data)}")
print(f"Test  samples : {len(test_data)}\n")

# Peek at a sample
sample = train_data[0]
print("── Sample review (truncated) ──")
print(sample["text"][:300], "...")
print(f"Label: {'positive' if sample['label'] == 1 else 'negative'}\n")

# ── 4. Tokenizer ──────────────────────────────────────────────────────────────
# BertTokenizer splits text into WordPiece subword tokens and adds:
#   [CLS] at the start  →  used as the sentence representation
#   [SEP] at the end    →  marks end of sequence
print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    """
    Tokenize a batch of reviews.
    - padding='max_length'  : pad all sequences to MAX_LEN
    - truncation=True       : truncate reviews longer than MAX_LEN
    - return_tensors not set here; DataLoader handles that later
    """
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )

print("Tokenizing dataset (this may take a moment)...")
train_encoded = train_data.map(tokenize, batched=True)
test_encoded  = test_data.map(tokenize, batched=True)

# Keep only the columns PyTorch needs
cols = ["input_ids", "attention_mask", "token_type_ids", "label"]
train_encoded.set_format(type="torch", columns=cols)
test_encoded.set_format(type="torch", columns=cols)

# ── 5. DataLoaders ────────────────────────────────────────────────────────────
train_loader = DataLoader(train_encoded, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_encoded,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Batches per epoch: {len(train_loader)}\n")

# ── 6. Model ──────────────────────────────────────────────────────────────────
# BertForSequenceClassification = BERT encoder + a linear classification head
# num_labels=2  →  binary classification (positive / negative)
print(f"Loading model: {MODEL_NAME}")
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    output_attentions=False,    # set True if you want to inspect attention weights
    output_hidden_states=False,
)
model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model parameters: {total_params:.1f}M\n")

# ── 7. Optimizer & Scheduler ──────────────────────────────────────────────────
# AdamW = Adam with weight decay fix (Loshchilov & Hutter, 2019)
# Standard choice for fine-tuning BERT
optimizer = AdamW(model.parameters(), lr=LR, eps=1e-8)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),   # 10% warmup
    num_training_steps=total_steps,
)

# ── 8. Training loop ──────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0

    for step, batch in enumerate(loader):
        # Move tensors to GPU/CPU
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        token_type_ids = batch["token_type_ids"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        # Forward pass — HuggingFace models return a dict-like object
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,         # passing labels computes CrossEntropyLoss internally
        )

        loss = outputs.loss        # scalar loss
        total_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
        optimizer.step()
        scheduler.step()

        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}/{len(loader)} | Loss: {loss.item():.4f}")

    return total_loss / len(loader)


# ── 9. Evaluation loop ────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            token_type_ids = batch["token_type_ids"].to(DEVICE)
            labels         = batch["label"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            # outputs.logits shape: (batch_size, num_labels)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=["negative", "positive"]
    )
    return acc, report


# ── 10. Run training ──────────────────────────────────────────────────────────
print("=" * 50)
print("Starting fine-tuning...")
print("=" * 50)

for epoch in range(1, EPOCHS + 1):
    print(f"\n── Epoch {epoch}/{EPOCHS} ──────────────────────")
    avg_loss = train_epoch(model, train_loader, optimizer, scheduler)
    print(f"  Avg training loss: {avg_loss:.4f}")

    acc, report = evaluate(model, test_loader)
    print(f"  Test accuracy    : {acc:.4f}")
    print(report)

# ── 11. Inference on custom text ──────────────────────────────────────────────
# Show students how to run the model on their own reviews
print("\n" + "=" * 50)
print("Custom inference examples")
print("=" * 50)

LABEL_MAP = {0: "NEGATIVE ❌", 1: "POSITIVE ✅"}

def predict(text: str) -> str:
    """Predict sentiment for a single movie review."""
    model.eval()
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    token_type_ids = encoding["token_type_ids"].to(DEVICE)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    pred  = int(np.argmax(probs))
    return f"{LABEL_MAP[pred]}  (confidence: {probs[pred]*100:.1f}%)"

reviews = [
    "This movie was an absolute masterpiece. The acting and story left me speechless.",
    "Terrible film. Boring plot, wooden acting, and a complete waste of two hours.",
    "It was fine. Not great, not bad — just an average Saturday afternoon watch.",
]

for review in reviews:
    print(f"\nReview : {review[:80]}...")
    print(f"Result : {predict(review)}")

# ── 12. Save model (optional) ─────────────────────────────────────────────────
# Uncomment to save fine-tuned model for later use
# model.save_pretrained("./bert-imdb-finetuned")
# tokenizer.save_pretrained("./bert-imdb-finetuned")
# print("\nModel saved to ./bert-imdb-finetuned")
