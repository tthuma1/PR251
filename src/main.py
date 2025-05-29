import time
import cloudscraper

scraper = cloudscraper.create_scraper()

# Preberemo URL-je iz datoteke
with open("../data/kvadrati_new/mojikvadratilinks.txt", "r", encoding="utf-8") as f:
    urls = [line.strip() for line in f if line.strip()]

# with open("missing.txt", "r", encoding="utf-8") as f:
#     missing = [int(line.strip()) for line in f if line.strip()]

# Shranimo HTML vsebine
for i, url in enumerate(urls):
    # if i+1 not in [568,569,570]: continue
    try:
        print(f"[{i+1}/{len(urls)}] Obiskujem: {url}")
        response = scraper.get(url)
        response.encoding = 'utf-8'
        
        if response.status_code != 200:
            print(f"❌ Napaka: {response.status_code} pri " + url)

        
        # Shranimo v datoteko (ločeno po številkah)
        with open(f"../data/kvadrati_new/pages/html_vsebina_{i+1}.html", "w", encoding="utf-8") as file:
            file.write(response.text)
        
        print(f"✅ Shranjeno: html_vsebina_{i+1}.html")
        # time.sleep(5)

    except Exception as e:
        print(f"❌ Napaka pri URL: {url} - {e}")

print("✅ Končano! Vse HTML vsebine so shranjene.")












# import undetected_chromedriver as uc
# import time

# # Inicializiramo brskalnik
# options = uc.ChromeOptions()
# options.headless = False  # Če postaviš na True, se preverjanje morda ne bo izvedlo
# options.add_argument("--disable-blink-features=AutomationControlled")

# driver = uc.Chrome(options=options)

# # Obiščemo stran
# url = "https://mojikvadrati.com"
# driver.get(url)

# # Počakamo, da Cloudflare preverjanje mine (če je potrebno, ročno potrdi CAPTCHA)
# time.sleep(10)

# # Pridobimo HTML vsebino
# html_content = driver.page_source

# # Shranimo v datoteko
# with open("mojikvadrati.html", "w", encoding="utf-8") as f:
#     f.write(html_content)

# print("✅ HTML shranjen!")

# # Zapremo brskalnik
# driver.quit()





# import cloudscraper

# scraper = cloudscraper.create_scraper()

# url = "https://mojikvadrati.com/nepremicnina/482796-prodaja-stanovanje-3-sobno-ljubljana-okolica-smartno-pri-litiji"
# response = scraper.get(url)
# response.encoding = 'utf-8'

# if response.status_code == 200:
#     with open("mojikvadrati.html", "w") as f:
#         f.write(response.text)
#     print("✅ HTML shranjen!")
# else:
#     print(f"❌ Napaka: {response.status_code}")

