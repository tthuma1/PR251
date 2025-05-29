import os
import csv
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim
import time

# Initialize the geolocator
geolocator = Nominatim(user_agent="geo_lookup")

links = []
with open("../data/kvadrati_new/mojikvadratilinks.txt", "r") as file:
    links = list(map(lambda x: x.strip("\n"), file.readlines()))

memorised_locations = dict()
# print(links)
def extract_data_from_html(file_path, house_id):
    global links
    global memorised_locations
    print("processing id:", house_id)
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
    
    data = {}
    data["id"] = house_id
    
    # Extract naslov
    naslov_tag = soup.find("h1")
    data["naslov"] = naslov_tag.get_text(strip=True) if naslov_tag else ""

    split_naslov = data["naslov"].split(",")
    for i in range(len(split_naslov)-1, -1, -1):
        kraj = split_naslov[i].strip()
        if kraj in memorised_locations:
            location = memorised_locations[kraj]
            data["longitude"] = location.longitude
            data["latitude"] = location.latitude

        try:
            location = geolocator.geocode(kraj)
            if location:
                data["longitude"] = location.longitude
                data["latitude"] = location.latitude
                memorised_locations[kraj] = location
                time.sleep(1)
                break
            else:
                print(f"Could not find coordinates for: {kraj}")
                time.sleep(1)
        except Exception as e:
            print(f"Error processing {kraj}: {e}")
            time.sleep(1)

    
    # Extract detail-category labels
    detail_category = soup.find(class_="detail-category")
    labels = detail_category.find_all("label") if detail_category else []
    
    data["vrsta_prodaje"] = labels[0].get_text(strip=True) if len(labels) > 0 else ""
    data["vrsta"] = labels[1].get_text(strip=True) if len(labels) > 1 else ""
    data["tip"] = labels[2].get_text(strip=True) if len(labels) > 2 else ""
    data["velikost"] = labels[3].get_text(strip=True) if len(labels) > 3 else ""
    data["leto_gradnje"] = labels[4].get_text(strip=True) if len(labels) > 4 else ""
    
    # Extract cena
    cena_tag = soup.find(class_="detail-cost-label")
    data["cena"] = cena_tag.get_text(strip=True) if cena_tag else ""
    
    # Extract prodajalec_oseba & prodajalec_agencija
    about_section = soup.find(class_="about")
    if about_section:
        oseba_tag = about_section.find("h2")
        agencija_tag = about_section.find("h3")
        data["prodajalec_oseba"] = oseba_tag.get_text(strip=True) if oseba_tag else ""
        data["prodajalec_agencija"] = agencija_tag.get_text(strip=True) if agencija_tag else ""
    else:
        data["prodajalec_oseba"] = ""
        data["prodajalec_agencija"] = ""
    
    # Extract splosno
    specs = soup.find_all(class_="pzl-specification")
    if len(specs) > 0:
        splosno_list = []
        for li in specs[0].find_all("li"):
            key = li.find("span").get_text(strip=True) if li.find("span") else ""
            value = li.find("strong").get_text(strip=True) if li.find("strong") else ""
            if key and value:
                splosno_list.append(f"{key}={value}")
        data["splosno"] = "|".join(splosno_list)
    else:
        data["splosno"] = ""
    
    # Extract znacilnosti
    if len(specs) > 1:
        znacilnosti_list = []
        for dt, dd in zip(specs[1].find_all("dt"), specs[1].find_all("dd")):
            key = dt.get_text(strip=True)
            values = "/".join([span.get_text(strip=True) for span in dd.find_all("span")])
            if key and values:
                znacilnosti_list.append(f"{key}={values}")
        data["znacilnosti"] = "|".join(znacilnosti_list)
    else:
        data["znacilnosti"] = ""
    
    # Extract opis
    description_section = soup.find(id="description-section")
    if description_section:
        divs = description_section.find_all("div")
        data["opis"] = (divs[1].get_text(strip=True) if len(divs) > 1 else "").replace("\n", " ")
    else:
        data["opis"] = ""

    data["url"] = links[house_id-1]
    
    return data

def main():
    input_dir = "../data/kvadrati_new/pages"
    output_file = "../data/kvadrati_new/kvadrati_new.csv"
    
    fieldnames = [
        "id", "naslov", "latitude", "longitude", "vrsta_prodaje", "vrsta", "tip", "velikost", "leto_gradnje", "cena",
        "prodajalec_oseba", "prodajalec_agencija", "splosno", "znacilnosti", "opis", "url"
    ]
    
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(len(links)):
            file_path = os.path.join(input_dir, f"html_vsebina_{i}.html")
            if os.path.exists(file_path):
                data = extract_data_from_html(file_path, i)
                writer.writerow(data)
            else:
                print(f"File {file_path} not found, skipping...")

if __name__ == "__main__":
    main()
