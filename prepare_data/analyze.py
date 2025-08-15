import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)


categorical = ["developer_group", "region_num", "address_code"]
numerical = ["metro_time", "floors_max", "total_area"]
target_col = "price_for_sqm"

file_path = "data/new_buildings.csv"


cols_to_use = categorical + numerical + [target_col]
df = pd.read_csv(file_path, usecols=lambda c: c in cols_to_use)

print(df["developer_group"].unique())

for col in categorical:
    df[col] = df[col].astype("category")


for col in numerical + [target_col]:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot: {col}")
    # print(max(df[col]))
    plt.show()


# слишком много столбцов , решил убрать 80%
for col in categorical:
    plt.figure(figsize=(12, 4))
    ax = sns.countplot(x=df[col], order=df[col].value_counts().index)

    labels = [item.get_text() for item in ax.get_xticklabels()]
    positions = ax.get_xticks()

    new_labels = [label if i % 5 == 0 else "" for i, label in enumerate(labels)]
    # new_labels = labels

    ax.set_xticks(positions)
    ax.set_xticklabels(new_labels, rotation=45, ha="right")

    plt.title(f"Countplot: {col}")
    plt.tight_layout()
    plt.show()

print("\n===== Выбросы (IQR) =====")
for col in numerical + [target_col]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} выбросов")


corr = df[numerical + [target_col]].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Матрица корреляций числовых признаков и таргета")
plt.show()
