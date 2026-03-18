#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 19:02:57 2026

@author: miguelrocha
"""

import re
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.feature_extraction.text import TfidfVectorizer



## FUNCTIONS FOR TEXT PROCESSING  (from previous class)

def clean_text(text): ## standardization
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def build_vocab(texts, max_words = 10000): ## building the vocabulary
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    
    most_common = counter.most_common(max_words)
    word_index = {word: i for i, (word, _) in enumerate(most_common)}
    return word_index

def encode(text, word_index, max_len=200): ## integer encoding
    tokens = clean_text(text).split()
    
    sequence = [word_index.get(word, 1) for word in tokens]
    # Truncate
    sequence = sequence[:max_len]
    # Pad
    if len(sequence) < max_len:
        sequence += [0] * (max_len - len(sequence))

    return torch.tensor(sequence, dtype=torch.long)

## Loading data / creating datasets (from previous class)
def load_split(data_dir, split):
    texts = []
    labels = []

    for label_type in ['pos', 'neg']:
        dir_path = os.path.join(data_dir, split, label_type)

        for fname in os.listdir(dir_path):
            with open(os.path.join(dir_path, fname), encoding='utf8') as f:
                texts.append(clean_text(f.read()))
                labels.append(1 if label_type == 'pos' else 0)

    return texts, labels

class IMDBDatasetEmbed(Dataset): 
    ## dataset with integer indexes (for embedding/ RNNs and so on)
    def __init__(self, texts, labels, word_index, max_words, max_len =200):
        self.texts = texts
        self.labels = labels
        self.word_index = word_index
        self.max_words = max_words
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = encode(self.texts[idx], self.word_index, self.max_len)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


## dataset for embeddings, RNNs, LSTMs, and Transformers
def load_dataset_embed(filespath, max_words = 10000, max_len = 200, batch_size = 512, val_perc = 0.8):
    ## load splits
    train_texts, train_labels = load_split(filespath, "train")
    test_texts, test_labels = load_split(filespath, "test")
    
    ## build vocabulary
    word_index = build_vocab(train_texts, max_words)
    
    ## create train and test splits
    full_train_dataset = IMDBDatasetEmbed(train_texts, train_labels, word_index, max_words, max_len)
    test_dataset = IMDBDatasetEmbed(test_texts, test_labels, word_index, max_words, max_len)
    
    train_size = int(val_perc * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size ## 20% for validation
    
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # reproducibility
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

### TRANSFORMER model
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim))
        self.norm2 = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x):
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.embedding(x) + self.pos_embedding(positions)
        attn_out, attn_weights = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        x = x.mean(dim=1)
        return self.classifier(x).squeeze(1)

## training = last class (added device to train and evaluation to allow GPUs)
def train(model, train_loader, val_loader, criterion, epochs = 5, lr = 0.001, verbose = True, device = None):
    ## verbose - print losses and accuracies per epoch
    
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if (device is not None):
        model = model.to(device)
    
    for epoch in range(epochs): 
        model.train()
        for x, y in train_loader:
            if (device is not None):
                x = x.to(device)
                y = y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(model, train_loader, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        val_losses.append(val_loss)

        if verbose: 
            print(f"Epoch {epoch+1}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")
    
    return train_accs, val_accs, train_losses, val_losses

def evaluate(model, loader, criterion, device = None):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            if device is not None:
                x = x.to(device)
                y = y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()

            preds = (outputs > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)

    return total_loss / len(loader), correct / total
    
def test_transformer():
    device = None
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps") ## apple silicon
    filespath = "../class3-code/aclImdb"

    max_words = 10000
    max_len = 32
    embed_dim = 128
    train_loader, val_loader, test_loader = load_dataset_embed(filespath, max_words, max_len = max_len)

    model = TransformerClassifier(vocab_size=max_words,
        embed_dim = embed_dim,
        num_heads=4,
        ff_dim=256,
        max_len=max_len,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    train(model, train_loader, val_loader, criterion, epochs = 10, device = device)
    _, test_acc = evaluate(model, test_loader, criterion, device = device)
    print(f"Test Accuracy: {test_acc:.4f}")

test_transformer()

