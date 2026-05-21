import os
import re
import json
import joblib
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# =========================
# CONFIG
# =========================

DATA_PATH = "data/electroplanet_products_large.csv"
MODEL_DIR = "models"

MAX_LEN = 35
EMBED_DIM = 64
HIDDEN_DIM = 64
BATCH_SIZE = 32
EPOCHS = 12
LEARNING_RATE = 0.001

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)


# =========================
# TEXT CLEANING
# =========================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    return clean_text(text).split()


# =========================
# VOCABULARY
# =========================

def build_vocab(texts, min_freq=1):
    word_count = {}

    for text in texts:
        for word in tokenize(text):
            word_count[word] = word_count.get(word, 0) + 1

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for word, count in word_count.items():
        if count >= min_freq:
            vocab[word] = len(vocab)

    return vocab


def encode_text(text, vocab, max_len=MAX_LEN):
    tokens = tokenize(text)
    ids = []

    for token in tokens:
        ids.append(vocab.get(token, vocab["<UNK>"]))

    if len(ids) < max_len:
        ids = ids + [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    return ids


# =========================
# USER PROFILES
# =========================

USER_PROFILES = [
    {
        "query": "réfrigérateur pas cher pour aid al adha",
        "keywords": ["ref", "réfrigérateur", "refrigerateur", "frigo", "combi", "comb", "no frost"],
        "budget": 9000,
        "min_discount": 5
    },
    {
        "query": "congélateur grande capacité",
        "keywords": ["cong", "congélateur", "congelateur", "freezer"],
        "budget": 8000,
        "min_discount": 10
    },
    {
        "query": "machine à laver économique",
        "keywords": ["machine", "laver", "lave", "linge", "mal", "washing"],
        "budget": 6000,
        "min_discount": 5
    },
    {
        "query": "smart tv pas cher",
        "keywords": ["tv", "smart", "led", "oled", "qled", "téléviseur", "televiseur"],
        "budget": 8000,
        "min_discount": 10
    },
    {
        "query": "ordinateur écran gaming",
        "keywords": ["pc", "ordinateur", "laptop", "monitor", "moniteur", "ecran", "écran", "gaming"],
        "budget": 10000,
        "min_discount": 10
    },
    {
        "query": "accessoires téléphone écouteurs chargeur",
        "keywords": ["ecouteur", "écouteur", "buds", "chargeur", "cable", "câble", "type c", "casque"],
        "budget": 2000,
        "min_discount": 10
    },
    {
        "query": "produits cuisine air fryer four machine café",
        "keywords": ["air", "fry", "fryer", "four", "café", "cafe", "cuisson", "friteuse", "tefal", "mixeur"],
        "budget": 4000,
        "min_discount": 10
    },
    {
        "query": "aspirateur nettoyage maison",
        "keywords": ["aspirateur", "nettoyeur", "karcher", "rowenta", "balai"],
        "budget": 4000,
        "min_discount": 10
    },
    {
        "query": "climatiseur pour maison",
        "keywords": ["climatiseur", "climatisation", "clima"],
        "budget": 7000,
        "min_discount": 5
    },
    {
        "query": "imprimante ordinateur bureau",
        "keywords": ["imprimante", "printer", "pc", "ordinateur", "bureau"],
        "budget": 5000,
        "min_discount": 5
    }
]


# =========================
# CREATE TRAINING DATA
# =========================

def product_matches_keywords(product_text, keywords):
    product_text = clean_text(product_text)

    for keyword in keywords:
        if keyword.lower() in product_text:
            return True

    return False


def create_training_data(df):
    rows = []

    for _, product in df.iterrows():
        product_name = str(product["product_name"])
        category = str(product.get("category", "Autre"))
        price = float(product["price"])
        discount = float(product.get("discount_percentage", 0))

        product_text = product_name + " " + category

        for profile in USER_PROFILES:
            query = profile["query"]
            budget = profile["budget"]
            min_discount = profile["min_discount"]

            keyword_match = product_matches_keywords(product_text, profile["keywords"])
            budget_match = price <= budget
            discount_match = discount >= min_discount

            score = 0

            if keyword_match:
                score += 0.50

            if budget_match:
                score += 0.25

            if discount_match:
                score += 0.25

            label = 1 if score >= 0.60 else 0

            combined_text = query + " sep " + product_text

            rows.append({
                "combined_text": combined_text,
                "price": price,
                "discount_percentage": discount,
                "budget": budget,
                "min_discount": min_discount,
                "budget_match": 1 if budget_match else 0,
                "discount_match": 1 if discount_match else 0,
                "label": label
            })

    training_df = pd.DataFrame(rows)

    positive_df = training_df[training_df["label"] == 1]
    negative_df = training_df[training_df["label"] == 0]

    if len(positive_df) > 0:
        negative_df = negative_df.sample(
            n=min(len(negative_df), len(positive_df) * 2),
            random_state=42
        )

    training_df = pd.concat([positive_df, negative_df], ignore_index=True)
    training_df = training_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return training_df


# =========================
# DATASET
# =========================

class RecommendationDataset(Dataset):
    def __init__(self, df, vocab):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab

        self.max_price = max(float(df["price"].max()), 1)
        self.max_budget = max(float(df["budget"].max()), 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text_ids = encode_text(row["combined_text"], self.vocab)

        price_norm = float(row["price"]) / self.max_price
        discount_norm = float(row["discount_percentage"]) / 100
        budget_norm = float(row["budget"]) / self.max_budget
        min_discount_norm = float(row["min_discount"]) / 100
        budget_match = float(row["budget_match"])
        discount_match = float(row["discount_match"])

        numerical_features = [
            price_norm,
            discount_norm,
            budget_norm,
            min_discount_norm,
            budget_match,
            discount_match
        ]

        label = float(row["label"])

        return {
            "text": torch.tensor(text_ids, dtype=torch.long),
            "num": torch.tensor(numerical_features, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32)
        }


# =========================
# GRU MODEL
# =========================

class GRURecommender(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_features):
        super(GRURecommender, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, text, num):
        embedded = self.embedding(text)

        _, hidden = self.gru(embedded)

        text_features = hidden[-1]

        combined = torch.cat((text_features, num), dim=1)

        output = self.fc(combined)

        return output.squeeze(1)


# =========================
# EVALUATION
# =========================

def evaluate(model, data_loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            text = batch["text"].to(device)
            num = batch["num"].to(device)
            labels = batch["label"].to(device)

            outputs = model(text, num)
            predictions = (outputs >= 0.5).float()

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0


# =========================
# TRAINING
# =========================

def train_model():
    print("Loading data...")

    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=["product_name", "price"])
    df["discount_percentage"] = df["discount_percentage"].fillna(0)
    df["category"] = df["category"].fillna("Autre")

    print("Products loaded:", len(df))

    print("Creating training data...")
    training_df = create_training_data(df)

    training_path = "data/training_data.csv"
    training_df.to_csv(training_path, index=False, encoding="utf-8-sig")

    print("Training samples:", len(training_df))
    print("Positive samples:", int(training_df["label"].sum()))
    print("Negative samples:", int(len(training_df) - training_df["label"].sum()))

    if training_df["label"].nunique() < 2:
        print("ERROR: Training data has only one class. Add more products or modify profiles.")
        return

    print("Building vocabulary...")
    vocab = build_vocab(training_df["combined_text"].tolist())

    print("Vocabulary size:", len(vocab))

    train_df, test_df = train_test_split(
        training_df,
        test_size=0.2,
        random_state=42,
        stratify=training_df["label"]
    )

    train_dataset = RecommendationDataset(train_df, vocab)
    test_dataset = RecommendationDataset(test_df, vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = GRURecommender(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_features=6
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nTraining model...")

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            text = batch["text"].to(device)
            num = batch["num"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            outputs = model(text, num)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predictions = (outputs >= 0.5).float()

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        test_acc = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {total_loss / len(train_loader):.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )

    model_path = os.path.join(MODEL_DIR, "gru_recommender.pth")

    torch.save(model.state_dict(), model_path)

    config = {
        "max_len": MAX_LEN,
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_features": 6,
        "vocab_size": len(vocab)
    }

    with open(os.path.join(MODEL_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    joblib.dump(vocab, os.path.join(MODEL_DIR, "vocab.pkl"))

    print("\nModel saved successfully.")
    print("Saved files:")
    print("-", model_path)
    print("-", os.path.join(MODEL_DIR, "vocab.pkl"))
    print("-", os.path.join(MODEL_DIR, "config.json"))
    print("-", training_path)


if __name__ == "__main__":
    train_model()