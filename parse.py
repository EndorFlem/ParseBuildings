import json
import time
import undetected_chromedriver as uc

step = 20
max_buildings = step * 2


def write_buildings(i):
    global max_buildings
    with open(f"pages/{i}.json", "w") as file:
        url = (
            "https://xn--80az8a.xn--d1aqf.xn--p1ai/"
            "%D1%81%D0%B5%D1%80%D0%B2%D0%B8%D1%81%D1%8B/api/kn/object?"
            f"offset={i}&limit=20&sortField=obj_publ_dt&sortType=desc&"
            "residentialBuildings=1&place=47&objStatus=0&fromDatePubl=2020-01-01"
        )

        driver.get(url)

        json_text = driver.execute_script("return document.body.innerText;")

        data = json.loads(json_text)
        max_buildings = data["data"]["total"]
        file.write(json_text)


driver = uc.Chrome()

i = 0
while i < max_buildings:
    print(max_buildings)
    write_buildings(i)
    time.sleep(2)
    i += 20

if max_buildings % step != 0:
    write_buildings(max_buildings - max_buildings % step)


driver.quit()
