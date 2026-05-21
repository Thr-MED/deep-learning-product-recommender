import os
import re
import json
import joblib
import pandas as pd
import torch
import torch.nn as nn


# =========================
# PATHS
# =========================

DATA_PATH = "data/electroplanet_products_large.csv"
MODEL_PATH = "models/gru_recommender.pth"
VOCAB_PATH = "models/vocab.pkl"
CONFIG_PATH = "models/config.json"


# =========================
# TEXT FUNCTIONS
# =========================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    return clean_text(text).split()


def encode_text(text, vocab, max_len):
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
# QUERY RELEVANCE
# =========================

QUERY_GROUPS = {
    "refrigerateur": [
        "réfrigérateur", "refrigerateur", "frigo", "ref", "combi", "comb",
        "no frost", "frost", "side by side"
    ],
    "congelateur": [
        "congélateur", "congelateur", "cong", "freezer"
    ],
    "machine_laver": [
        "machine à laver", "lave linge", "lave-linge", "mal", "washing"
    ],
    "tv": [
        "tv", "smart tv", "led", "oled", "qled", "téléviseur", "televiseur"
    ],
    "ordinateur": [
        "pc", "ordinateur", "laptop", "monitor", "moniteur", "ecran", "écran", "gaming"
    ],
    "telephone": [
        "smartphone", "iphone", "galaxy", "xiaomi", "oppo",
        "telephone", "téléphone", "phone", "mobile", "apple"
    ],
    "accessoires": [
        "ecouteur", "écouteur", "buds", "chargeur", "cable", "câble", "type c", "casque"
    ],
    "cuisine": [
        "air fryer", "four", "café", "cafe", "cuisson", "friteuse",
        "tefal", "mixeur", "cuisine", "robot", "cocotte"
    ],
    "nettoyage": [
        "aspirateur", "nettoyeur", "karcher", "rowenta", "balai"
    ],
    "climatisation": [
        "climatiseur", "climatisation", "clima"
    ]
}


def detect_query_group(user_query):
    query = clean_text(user_query)

    for group, keywords in QUERY_GROUPS.items():
        for keyword in keywords:
            if keyword in query:
                return group

    return None


def query_relevance_score(user_query, product_name, category):
    query = clean_text(user_query)
    product_text = clean_text(str(product_name) + " " + str(category))

    query_words = set(query.split())
    product_words = set(product_text.split())

    # 1. Direct word overlap
    if len(query_words) == 0:
        overlap_score = 0
    else:
        overlap_score = len(query_words.intersection(product_words)) / len(query_words)

    # 2. Substring matching
    substring_score = 0
    if len(query_words) > 0:
        matches = sum(1 for word in query_words if word in product_text)
        substring_score = matches / len(query_words)

    # 3. Category/group matching
    group = detect_query_group(user_query)
    group_score = 0

    if group is not None:
        for keyword in QUERY_GROUPS[group]:
            if keyword in product_text:
                group_score = 1
                break

    relevance = (0.40 * overlap_score) + (0.40 * substring_score) + (0.20 * group_score)

    return min(relevance, 1)


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
# LOAD MODEL
# =========================

def load_recommender():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    vocab = joblib.load(VOCAB_PATH)

    model = GRURecommender(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        num_features=config["num_features"]
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()

    return model, vocab, config


# =========================
# RECOMMENDATION FUNCTION
# =========================

def recommend_products(
    user_query,
    budget,
    min_discount,
    category_filter="All",
    top_n=10,
    data_path=DATA_PATH
):
    df = pd.read_csv(data_path)

    df = df.dropna(subset=["product_name", "price"])
    df["discount_percentage"] = df["discount_percentage"].fillna(0)
    df["category"] = df["category"].fillna("Autre")

    if category_filter != "All":
        df = df[df["category"] == category_filter]

    df = df[df["discount_percentage"] >= min_discount]
    df = df[df["price"] <= budget]

    if df.empty:
        return pd.DataFrame()

    model, vocab, config = load_recommender()
    max_len = config["max_len"]

    max_price = max(float(df["price"].max()), 1)
    max_budget = max(float(budget), 1)
    max_discount = max(float(df["discount_percentage"].max()), 1)

    results = []

    for _, product in df.iterrows():
        product_name = str(product["product_name"])
        category = str(product.get("category", "Autre"))
        price = float(product["price"])
        discount = float(product.get("discount_percentage", 0))

        product_text = product_name + " " + category
        combined_text = user_query + " sep " + product_text

        text_ids = encode_text(combined_text, vocab, max_len)

        price_norm = price / max_price
        discount_norm = discount / 100
        budget_norm = budget / max_budget
        min_discount_norm = min_discount / 100
        budget_match = 1 if price <= budget else 0
        discount_match = 1 if discount >= min_discount else 0

        numerical_features = [
            price_norm,
            discount_norm,
            budget_norm,
            min_discount_norm,
            budget_match,
            discount_match
        ]

        text_tensor = torch.tensor([text_ids], dtype=torch.long)
        num_tensor = torch.tensor([numerical_features], dtype=torch.float32)

        with torch.no_grad():
            model_score = model(text_tensor, num_tensor).item()

        discount_score = discount / max_discount if max_discount > 0 else 0

        relevance_score = query_relevance_score(
            user_query=user_query,
            product_name=product_name,
            category=category
        )

        final_score = (
            0.20 * model_score
            + 0.60 * relevance_score
            + 0.20 * discount_score
        )

        product_dict = product.to_dict()
        product_dict["model_score"] = round(model_score, 4)
        product_dict["query_relevance_score"] = round(relevance_score, 4)
        product_dict["discount_score"] = round(discount_score, 4)
        product_dict["final_score"] = round(final_score, 4)

        results.append(product_dict)

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        by="final_score",
        ascending=False
    ).head(top_n)

    return result_df


# =========================
# TEST
# =========================

if __name__ == "__main__":
    query = "réfrigérateur pas cher pour aid al adha"
    budget = 9000
    min_discount = 5

    recommendations = recommend_products(
        user_query=query,
        budget=budget,
        min_discount=min_discount,
        category_filter="All",
        top_n=10
    )

    if recommendations.empty:
        print("No recommendations found.")
    else:
        print(recommendations[
            [
                "product_name",
                "category",
                "price",
                "old_price",
                "discount_percentage",
                "model_score",
                "query_relevance_score",
                "discount_score",
                "final_score",
                "source_page"
            ]
        ])