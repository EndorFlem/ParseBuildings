import pandas as pd

# df = pd.read_csv("data/new_buildings.csv")
df = pd.read_csv("data/buildings_with_embeddings.csv")

df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")

print(len(df), "new")
old_df = pd.read_csv("data/new_buildings.csv")
print(len(old_df), "old")

first_set = df[(df["start_date"].dt.year >= 2022) & (df["start_date"].dt.year <= 2023)]
# second_set = df[(df["start_date"].dt.year == 2024) & (df["start_date"].dt.month <= 6)]
# third_set = df[(df["start_date"].dt.year == 2024) & (df["start_date"].dt.month > 6)]
second_set = df[(df["start_date"].dt.year == 2024)]
third_set = df[(df["start_date"].dt.year == 2025)]

print(len(first_set))
print(len(second_set))
print(len(third_set))

first_set.to_csv("data/train.csv", index=False)
second_set.to_csv("data/validation.csv", index=False)
third_set.to_csv("data/test.csv", index=False)
