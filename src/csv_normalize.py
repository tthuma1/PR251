import csv

input_file = "../data/csvs/kvadrati.csv"
output_file = "../data/csvs/kvadrati_normalized.csv"

# Hold unique keys for new columns
splosno_keys = set()
znacilnosti_keys = set()
data_rows = []

# Step 1: Read and parse the input CSV
with open(input_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Parse splosno and znacilnosti to gather all unique keys
        splosno_dict = {}
        for pair in row["splosno"].split("|"):
            if "=" in pair:
                key, val = pair.split("=", 1)
                splosno_dict[key] = val
                splosno_keys.add(key)
        row["parsed_splosno"] = splosno_dict

        znacilnosti_dict = {}
        for pair in row["znacilnosti"].split("|"):
            if "=" in pair:
                key, vals = pair.split("=", 1)
                val_list = [v.strip() for v in vals.split("/")]
                znacilnosti_dict[key] = val_list
                for val in val_list:
                    znacilnosti_keys.add(f"{key}_{val}")
        row["parsed_znacilnosti"] = znacilnosti_dict

        data_rows.append(row)

# Step 2: Build the output CSV
basic_fields = [
    "id", "naslov", "latitude", "longitude", "vrsta_prodaje", "vrsta", "tip", "velikost", "leto_gradnje",
    "cena", "prodajalec_oseba", "prodajalec_agencija", "opis", "url"
]

normalized_fields = sorted(splosno_keys) + sorted(znacilnosti_keys)
all_fields = basic_fields + normalized_fields

with open(output_file, "w", newline="", encoding="utf-8") as f_out:
    writer = csv.DictWriter(f_out, fieldnames=all_fields)
    writer.writeheader()

    for row in data_rows:
        flat_row = {key: row.get(key, "") for key in basic_fields}

        # Add splosno fields
        for key in splosno_keys:
            flat_row[key] = row["parsed_splosno"].get(key, "")

        # Add znacilnosti binary flags
        for key in znacilnosti_keys:
            flat_row[key] = 0

        for key, val_list in row["parsed_znacilnosti"].items():
            for val in val_list:
                col_key = f"{key}_{val}"
                if col_key in znacilnosti_keys:
                    flat_row[col_key] = 1

        writer.writerow(flat_row)

print(f"Normalized CSV written to: {output_file}")
