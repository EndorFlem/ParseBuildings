import json
import time
import undetected_chromedriver as uc

step = 20
max_buildings = step * 2
capcha = True


def write_buildings(i):
    global max_buildings
    global capcha
    with open(f"pages/{i}stp.json", "w") as file:

        # url = (
        #     f"https://xn--80az8a.xn--d1aqf.xn--p1ai/%D1%81%D0%B5%D1%80%D0%B2%D0%B8%D1%81%D1%8B/api/kn/object?offset={i}&limit=20&sortField=obj_publ_dt&sortType=desc&residentialBuildings=1&place=47&fromQuarter=2020-01-01&toQuarter=2025-12-31"
        #     "https://xn--80az8a.xn--d1aqf.xn--p1ai/"
        # )

        url = (
            f"https://xn--80az8a.xn--d1aqf.xn--p1ai/%D1%81%D0%B5%D1%80%D0%B2%D0%B8%D1%81%D1%8B/api/kn/object?offset={i}&limit=20&sortField=obj_publ_dt&sortType=desc&residentialBuildings=1&place=78&fromQuarter=2020-01-01&toQuarter=2025-12-31"
            "https://xn--80az8a.xn--d1aqf.xn--p1ai/"
        )

        driver.get(url)

        # на этот раз была капча и пришлось делать костыль
        if capcha:
            input("wait for captcha")
            capcha = False

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
