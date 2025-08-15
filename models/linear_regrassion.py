import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

df_train = pd.read_csv("data/train.csv")
df_valid = pd.read_csv("data/validation.csv")
df_test = pd.read_csv("data/test.csv")

# у некоторых (на самом деле большинства домов) рядом нет метро
# так что я решил так заполнять пропуски , считая , что до метро идти очень далеко
# P.S. проведя ещё пару тестов понял , что данный критерий лишь немного влияет на итоговый результат
for df in [df_train, df_valid, df_test]:
    df["metro_time"] = df["metro_time"].fillna(60)

# вот тут я добавил start_date и результат улучшился
categorical = ["developer_group", "region_num", "address_code", "start_date"]

numerical = [
    "metro_time",
    "floors_max",
    "total_area",
    "metro_time",
    "floors_max",
    "total_area",
    "latitude",
    "longitude",
    "flats_count",
    "parking_count",
    "inflation",
    "curs",
    "key_rate",
    # "price_for_sqm_adj",
]
target_col = "price_for_sqm"

y_train = np.log1p(df_train[target_col])
y_valid = np.log1p(df_valid[target_col])
y_test = np.log1p(df_test[target_col])

X_train_cat = pd.get_dummies(
    df_train[categorical], columns=categorical, drop_first=True
)
X_valid_cat = pd.get_dummies(
    df_valid[categorical], columns=categorical, drop_first=True
)
X_test_cat = pd.get_dummies(df_test[categorical], columns=categorical, drop_first=True)

train_cols = X_train_cat.columns
X_valid_cat = X_valid_cat.reindex(columns=train_cols, fill_value=0)
X_test_cat = X_test_cat.reindex(columns=train_cols, fill_value=0)

X_train_num = df_train[numerical].fillna(df_train[numerical].median())
X_valid_num = df_valid[numerical].fillna(df_train[numerical].median())
X_test_num = df_test[numerical].fillna(df_train[numerical].median())

X_mean = X_train_num.mean(axis=0)
X_std = X_train_num.std(axis=0).replace(0, 1)

X_train_num_scaled = (X_train_num - X_mean) / X_std
X_valid_num_scaled = (X_valid_num - X_mean) / X_std
X_test_num_scaled = (X_test_num - X_mean) / X_std

X_train = pd.concat([X_train_num_scaled, X_train_cat], axis=1).astype(float).values
X_valid = pd.concat([X_valid_num_scaled, X_valid_cat], axis=1).astype(float).values
X_test = pd.concat([X_test_num_scaled, X_test_cat], axis=1).astype(float).values

X_train_scaled = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_valid_scaled = np.hstack([np.ones((X_valid.shape[0], 1)), X_valid])
X_test_scaled = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

I = np.eye(X_train_scaled.shape[1])
I[0, 0] = 0
alpha = 0.01
theta = np.linalg.inv(X_train_scaled.T @ X_train_scaled + alpha * I) @ (
    X_train_scaled.T @ y_train
)

y_train_pred_log = X_train_scaled @ theta
y_valid_pred_log = X_valid_scaled @ theta
y_test_pred_log = X_test_scaled @ theta

y_train_pred = np.expm1(y_train_pred_log)
y_valid_pred = np.expm1(y_valid_pred_log)
y_test_pred = np.expm1(y_test_pred_log)

y_train_true = np.expm1(y_train)
y_valid_true = np.expm1(y_valid)
y_test_true = np.expm1(y_test)


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


print(f"Train MAPE: {mape(y_train_true, y_train_pred):.2f}%")
print(f"Validation MAPE: {mape(y_valid_true, y_valid_pred):.2f}%")
print(f"Test MAPE: {mape(y_test_true, y_test_pred):.2f}%")

print(f"Train RMSE: {root_mean_squared_error(y_train_true, y_train_pred):.2f}")
print(f"Validation RMSE: {root_mean_squared_error(y_valid_true, y_valid_pred):.2f}")
print(f"Test RMSE: {root_mean_squared_error(y_test_true, y_test_pred):.2f}")
