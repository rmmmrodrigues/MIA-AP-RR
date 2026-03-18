"""
=============================================================
 Neural Machine Translation using Encoder-Decoder Transformer
 (MarianMT / Helsinki-NLP via HuggingFace + PyTorch)
=============================================================
 Class example: Using a pre-trained encoder-decoder Transformer
 for English → French translation, with an optional fine-tuning
 loop on a custom dataset.

 Architecture contrast with BERT (from bert_imdb_sentiment.py):
   BERT         = Encoder only  → good for understanding tasks
                                  (classification, NER, QA)
   MarianMT     = Encoder+Decoder → good for generation tasks
                                  (translation, summarization)

 Requirements:
   pip install transformers datasets torch sentencepiece sacrebleu

 Tested with:
   transformers==4.40, datasets==2.19, torch==2.2
=============================================================
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import (
    MarianTokenizer,
    MarianMTModel,
    get_linear_schedule_with_warmup,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
import numpy as np
import random

# For BLEU score evaluation
try:
    from sacrebleu.metrics import BLEU
    HAS_BLEU = True
except ImportError:
    HAS_BLEU = False
    print("sacrebleu not found — skipping BLEU score. Install with: pip install sacrebleu")

# ── 1. Reproducibility ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 2. Config ─────────────────────────────────────────────────────────────────
#
# MarianMT model naming convention: Helsinki-NLP/opus-mt-{src}-{tgt}
# Examples:
#   "Helsinki-NLP/opus-mt-en-fr"   English → French
#   "Helsinki-NLP/opus-mt-en-de"   English → German
#   "Helsinki-NLP/opus-mt-en-es"   English → Spanish
#   "Helsinki-NLP/opus-mt-fr-en"   French  → English
#
MODEL_NAME    = "Helsinki-NLP/opus-mt-en-fr"
SRC_LANG      = "en"
TGT_LANG      = "fr"

MAX_LEN       = 128     # max token length for src and tgt sequences
BATCH_SIZE    = 16
EPOCHS        = 3       # for fine-tuning demo; set 0 to skip fine-tuning
LR            = 5e-5
TRAIN_SUBSET  = 2000    # subset of opus_books for fine-tuning demo; None = full
TEST_SUBSET   = 200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}\n")

# ── 3. Load tokenizer & model ─────────────────────────────────────────────────
# MarianTokenizer uses SentencePiece (BPE-based), different from BERT's WordPiece.
# Key difference: no [CLS]/[SEP]; uses language tags like >>fr<< instead.
print(f"Loading model & tokenizer: {MODEL_NAME}")
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model     = MarianMTModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model parameters: {total_params:.1f}M\n")

# ── 4. Architecture overview (printed for class) ──────────────────────────────
print("=" * 55)
print(" ENCODER-DECODER ARCHITECTURE OVERVIEW")
print("=" * 55)
print("""
  Source sentence (English)
        │
        ▼
  ┌─────────────┐
  │   ENCODER   │  ← Reads the full source sentence
  │  (N layers) │    Builds rich contextual representations
  │             │    Self-attention over source tokens
  └──────┬──────┘
         │  encoder hidden states (context)
         ▼
  ┌─────────────┐
  │   DECODER   │  ← Generates target tokens one by one
  │  (N layers) │    1. Self-attention over generated tokens so far
  │             │    2. Cross-attention: each decoder token attends
  │             │       to ALL encoder states → the key mechanism!
  └──────┬──────┘
         │
         ▼
  Target sentence (French)  [token by token]

  KEY DIFFERENCE FROM ENCODER-ONLY (BERT):
  • Encoder-only  → sees full sequence, learns representations
  • Encoder-Decoder → encoder understands source, decoder
    generates target using cross-attention to the encoder.
""")
print("=" * 55 + "\n")

# ── 5. Zero-shot translation (no fine-tuning needed) ─────────────────────────
# The pre-trained model already translates well out of the box.

def translate(texts: list[str], num_beams: int = 4) -> list[str]:
    """
    Translate a list of English strings to French.

    Args:
        texts     : list of source sentences
        num_beams : beam search width (higher = better quality, slower)

    Returns:
        list of translated strings
    """
    # Tokenize — MarianMT expects plain text (no language prefix needed for
    # single-language-pair models; multi-lingual models need >>fr<< prefix)
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    ).to(DEVICE)

    model.eval()
    with torch.no_grad():
        # generate() runs the decoder autoregressively:
        # at each step it predicts the next token using:
        #   1. previously generated tokens (causal self-attention)
        #   2. encoder output            (cross-attention)
        translated_ids = model.generate(
            **inputs,
            num_beams=num_beams,         # beam search for better translations
            max_length=MAX_LEN,
            early_stopping=True,
        )

    # Decode token IDs back to strings; skip_special_tokens removes </s> etc.
    return tokenizer.batch_decode(translated_ids, skip_special_tokens=True)


print("── Zero-shot Translation Examples ──────────────────────")
examples = [
    "The weather is beautiful today.",
    "Machine learning is a subset of artificial intelligence.",
    "I would like a coffee with milk, please.",
    "The transformer architecture revolutionized natural language processing.",
    "She went to the market to buy fresh vegetables and fruits.",
]

for src in examples:
    tgt = translate([src])[0]
    print(f"  EN: {src}")
    print(f"  FR: {tgt}\n")

# ── 6. Load parallel corpus for fine-tuning ───────────────────────────────────
# opus_books is a literary parallel corpus (English-French books)
# Other options: wmt14, tatoeba, europarl
print("Loading opus_books dataset (en-fr)...")
raw = load_dataset("opus_books", lang1="en", lang2="fr")
raw = raw["train"].train_test_split(test_size=0.1, seed=SEED)

train_raw = raw["train"].shuffle(seed=SEED)
test_raw  = raw["test"].shuffle(seed=SEED)

if TRAIN_SUBSET:
    train_raw = train_raw.select(range(TRAIN_SUBSET))
if TEST_SUBSET:
    test_raw = test_raw.select(range(TEST_SUBSET))

print(f"Fine-tune train : {len(train_raw)} sentence pairs")
print(f"Fine-tune test  : {len(test_raw)} sentence pairs\n")

# Peek at a sample
sample = train_raw[0]["translation"]
print(f"Sample EN: {sample['en']}")
print(f"Sample FR: {sample['fr']}\n")

# ── 7. Tokenization for seq2seq ───────────────────────────────────────────────
def preprocess(batch):
    """
    Tokenize source (EN) and target (FR) sentences.

    For seq2seq we tokenize source and target separately:
    - Source goes through encoder as input_ids / attention_mask
    - Target goes through decoder; 'labels' are the target token IDs
      with padding replaced by -100 so they're ignored in loss
    """
    sources = [ex["en"] for ex in batch["translation"]]
    targets = [ex["fr"] for ex in batch["translation"]]

    model_inputs = tokenizer(
        sources,
        max_length=MAX_LEN,
        truncation=True,
        padding=False,   # DataCollatorForSeq2Seq handles dynamic padding
    )

    # Tokenize targets inside tokenizer's target context
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=MAX_LEN,
            truncation=True,
            padding=False,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing parallel corpus...")
train_enc = train_raw.map(preprocess, batched=True, remove_columns=train_raw.column_names)
test_enc  = test_raw.map(preprocess,  batched=True, remove_columns=test_raw.column_names)
train_enc.set_format("torch")
test_enc.set_format("torch")

# DataCollatorForSeq2Seq handles dynamic padding per batch (more efficient)
# and replaces padding tokens in labels with -100 (ignored by loss function)
collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

train_loader = DataLoader(train_enc, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collator)
test_loader  = DataLoader(test_enc,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

# ── 8. Optimizer & scheduler ──────────────────────────────────────────────────
optimizer   = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler   = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)

# ── 9. Training loop ──────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        # Forward pass
        # The model internally:
        #   1. Runs encoder on input_ids → encoder hidden states
        #   2. Runs decoder on shifted labels, attending to encoder states
        #   3. Computes cross-entropy loss against labels
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}/{len(loader)} | Loss: {loss.item():.4f}")

    return total_loss / len(loader)

# ── 10. Evaluation with BLEU score ────────────────────────────────────────────
def evaluate_bleu(model, loader):
    """
    BLEU (Bilingual Evaluation Understudy) is the standard metric
    for translation quality. It measures n-gram overlap between
    machine translation and reference (human) translations.
    Score ranges 0–100; >30 is generally considered good.
    """
    model.eval()
    all_preds, all_refs = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"]

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=4,
                max_length=MAX_LEN,
            )

            preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
            # Replace -100 (padding) with pad_token_id before decoding
            labels[labels == -100] = tokenizer.pad_token_id
            refs  = tokenizer.batch_decode(labels,    skip_special_tokens=True)

            all_preds.extend(preds)
            all_refs.extend(refs)

    if HAS_BLEU:
        bleu = BLEU()
        score = bleu.corpus_score(all_preds, [all_refs])
        return str(score), all_preds[:3], all_refs[:3]
    else:
        return "sacrebleu not installed", all_preds[:3], all_refs[:3]

# ── 11. Run fine-tuning ───────────────────────────────────────────────────────
if EPOCHS > 0:
    print("=" * 55)
    print("Fine-tuning on opus_books (en→fr)")
    print("=" * 55)

    for epoch in range(1, EPOCHS + 1):
        print(f"\n── Epoch {epoch}/{EPOCHS} ──────────────────────────")
        avg_loss = train_epoch(model, train_loader, optimizer, scheduler)
        print(f"  Avg train loss : {avg_loss:.4f}")

        bleu_score, sample_preds, sample_refs = evaluate_bleu(model, test_loader)
        print(f"  BLEU score     : {bleu_score}")
        print("\n  Sample predictions:")
        for pred, ref in zip(sample_preds, sample_refs):
            print(f"    Pred : {pred}")
            print(f"    Ref  : {ref}\n")

# ── 12. Compare decoding strategies ──────────────────────────────────────────
# Great teaching moment: show how decoding strategy affects output quality
print("=" * 55)
print(" DECODING STRATEGY COMPARISON")
print("=" * 55)

test_sentence = "The students learned about attention mechanisms in the transformer model."
inputs = tokenizer([test_sentence], return_tensors="pt", padding=True).to(DEVICE)
model.eval()

strategies = {
    "Greedy search\n  (always pick top token)": dict(
        num_beams=1, do_sample=False
    ),
    "Beam search (4 beams)\n  (explore top-4 paths)": dict(
        num_beams=4, early_stopping=True
    ),
    "Beam search (8 beams)\n  (explore top-8 paths)": dict(
        num_beams=8, early_stopping=True
    ),
    "Sampling (temp=0.7)\n  (random, more creative)": dict(
        num_beams=1, do_sample=True, temperature=0.7
    ),
}

print(f"\nSource: {test_sentence}\n")
with torch.no_grad():
    for name, kwargs in strategies.items():
        out = model.generate(**inputs, max_length=MAX_LEN, **kwargs)
        translation = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"  [{name}]")
        print(f"  → {translation}\n")

# ── 13. Save fine-tuned model (optional) ─────────────────────────────────────
# Uncomment to save:
# model.save_pretrained("./marian-en-fr-finetuned")
# tokenizer.save_pretrained("./marian-en-fr-finetuned")
# print("Model saved to ./marian-en-fr-finetuned")
