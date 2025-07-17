import pandas as pd

df = pd.read_csv("new_buildings.csv")

df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")

first_set = df[df["start_date"].dt.year < 2024]
second_set = df[(df["start_date"].dt.year == 2024)]
third_set = df[df["start_date"].dt.year > 2024]

# first_set = df[df["start_date"].dt.year < 2024]
# second_set = df[(df["start_date"].dt.year == 2024) & (df["start_date"].dt.month <= 6)]
# third_set = df[(df["start_date"].dt.year == 2024) & (df["start_date"].dt.month > 6)]

# посмотрев на этим данные я решил сделать валидационными записи за 2024 , а тестовыми за следующие за 2024 года
# print(len(first_set))
# print(len(second_set))
# print(len(third_set))


first_set.to_csv("train.csv", index=False)
second_set.to_csv("validation.csv", index=False)
third_set.to_csv("test.csv", index=False)
