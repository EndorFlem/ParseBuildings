import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel
import os

df = pd.read_csv("data/new_buildings.csv")

categorical_cols = ["region_num", "building_class", "wall_material"]
text_cols = ["address", "developer", "metro"]
numerical = [
    "metro_time",
    "floors_max",
    "total_area",
    "latitude",
    "longitude",
    "flats_count",
    "parking_count",
]

model = SentenceTransformer(
    "intfloat/multilingual-e5-small",
    cache_folder="./cache",
)


def embed_column(series):
    texts = series.fillna("").astype(str).tolist()
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )
    return np.array(embeddings)


for col in categorical_cols + text_cols:
    print(f"Обрабатываем колонку: {col}")
    emb = embed_column(df[col])
    for i in range(emb.shape[1]):
        df[f"{col}_emb_{i}"] = emb[:, i]

df.to_csv("data/buildings_with_embeddings.csv", index=False)
