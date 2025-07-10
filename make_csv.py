import pandas as pd
import os
import json

csv_columns = {
    "objId": "id",
    "objCommercNm": "name",
    "developer.shortName": "developer",
    "developer.devInn": "developer_inn",
    "objPriceAVG": "price_for_sqm",
    "shortAddr": "address",
    "latitude": "latitude",
    "longitude": "longitude",
    "metro.name": "metro",
    "metro.time": "metro_time",
    "objFloorMin": "floors_min",
    "objFloorMax": "floors_max",
    "objElemLivingCnt": "apartments",
    "objSquareLiving": "total_area",
    "objReady100PercDt": "completion_date",
    "objGuarantyEscrowFlg": "escrow",
    "objProblemFlg": "problem_flag",
    "rpdNum": "rpd_number",
}

all_data = []

for path in os.listdir("pages"):
    with open(f"pages/{path}", "r", encoding="utf-8") as f:
        data = json.load(f)

    df_with_price = pd.json_normalize(data["data"]["list"])
    all_data.append(df_with_price[df_with_price["objPriceAVG"].notna()].copy())


combined_df = pd.concat(all_data, ignore_index=True)
result_df = combined_df[list(csv_columns.keys())].rename(columns=csv_columns)
result_df.to_csv("new_buildings.csv", index=False, encoding="utf-8-sig")
