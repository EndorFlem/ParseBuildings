import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("data/new_buildings.csv")

cheapest_developers = (
    df.groupby(["developer", "developer_inn"])["price_for_sqm"].mean().reset_index()
)
cheapest_developers = cheapest_developers.sort_values("price_for_sqm").head(5)

expensive_developers = (
    df.groupby(["developer", "developer_inn"])["price_for_sqm"].mean().reset_index()
)
expensive_developers = expensive_developers.sort_values(
    "price_for_sqm", ascending=False
).head(5)

mean_price_spb = df[df["region_num"] == 78]["price_for_sqm"].mean()
median_price_spb = df[df["region_num"] == 78]["price_for_sqm"].median()
mean_price_lo = df[df["region_num"] == 47]["price_for_sqm"].mean()
median_price_lo = df[df["region_num"] == 47]["price_for_sqm"].median()

print("Топ-5 самых дешевых застройщиков:")
print(cheapest_developers[["developer", "price_for_sqm"]].to_string(index=False))
print("\nТоп-5 самых дорогих застройщиков:")
print(expensive_developers[["developer", "price_for_sqm"]].to_string(index=False))
print(
    f"\nСредняя цена за кв.м в Санкт-Петербурге (region_num=78): {mean_price_spb:.2f}"
)
print(
    f"Медианная цена за кв.м в Санкт-Петербурге (region_num=78): {median_price_spb:.2f}"
)
print(
    f"Средняя цена за кв.м в Ленинградской области (region_num=47): {mean_price_lo:.2f}"
)
print(
    f"Медианная цена за кв.м в Ленинградской области (region_num=47): {median_price_lo:.2f}"
)


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="total_area", y="price_for_sqm", alpha=0.5)
sns.regplot(data=df, x="total_area", y="price_for_sqm", scatter=False, color="red")
plt.title("Зависимость цены от площади")  # её не наблюдается
plt.xlabel("Площадь (м^2)")
plt.ylabel("Цена за м^2")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="building_class", y="price_for_sqm", palette="coolwarm")
plt.title("Распределение цены за м^2 по классам зданий")
plt.xlabel("Класс здания")
plt.ylabel("Цена за м²")
plt.tight_layout()
plt.show()
