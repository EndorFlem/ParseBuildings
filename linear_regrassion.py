import numpy as np
import pandas as pd

df_train = pd.read_csv("train.csv")
df_valid = pd.read_csv("validation.csv")
df_test = pd.read_csv("test.csv")


# у некоторых (на самом деле большинства домов) рядом нет метро
# так что я решил так заполнять пропуски , считая , что до метро идти очень далеко
# P.S. проведя ещё пару тестов понял , что данный критерий лишь немного влияет на итоговый результат
df_train["metro_time"] = df_train["metro_time"].fillna(45)
df_valid["metro_time"] = df_valid["metro_time"].fillna(45)
df_test["metro_time"] = df_test["metro_time"].fillna(45)

features = [
    "latitude",
    # "longitude",
    # "metro_time",
    # "floors_min",
    "floors_max",
    # "apartments",
    "total_area",
]
X_train = df_train[features].values
y_train = df_train["price_for_sqm"].values

X_valid = df_valid[features].values
y_valid = df_valid["price_for_sqm"].values

X_test = df_test[features].values
y_test = df_test["price_for_sqm"].values

X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)

X_train_norm = (X_train - X_mean) / X_std
X_valid_norm = (X_valid - X_mean) / X_std
X_test_norm = (X_test - X_mean) / X_std

X_train_scaled = np.hstack([np.ones((X_train_norm.shape[0], 1)), X_train_norm])
X_valid_scaled = np.hstack([np.ones((X_valid_norm.shape[0], 1)), X_valid_norm])
X_test_scaled = np.hstack([np.ones((X_test_norm.shape[0], 1)), X_test_norm])

theta = np.linalg.inv(X_train_scaled.T @ X_train_scaled) @ (X_train_scaled.T @ y_train)

y_train_pred = X_train_scaled @ theta
y_valid_pred = X_valid_scaled @ theta
y_test_pred = X_test_scaled @ theta


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


print(f"train mape: {mape(y_train, y_train_pred):.2f}%")
print(f"validation mape: {mape(y_valid, y_valid_pred):.2f}%")
print(f"test mape: {mape(y_test, y_test_pred):.2f}%")
