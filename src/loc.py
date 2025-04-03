from geopy.geocoders import Nominatim
import time

# Initialize the geolocator
geolocator = Nominatim(user_agent="geo_lookup")

# Read addresses from file
# with open("address.txt", "r") as file:
#     addresses = file.readlines()

addresses = ["Ljubljana, Novo Polje"]

# Open a file to save results
with open("coordinates.txt", "w") as output:
    output.write("Address,Latitude,Longitude\n")  # Header

    for address in addresses:
        address = address.strip()
        if not address:
            continue  # Skip empty lines

        try:
            location = geolocator.geocode(address)
            if location:
                output.write(f"{address},{location.latitude},{location.longitude}\n")
                print(f"Processed: {address} -> ({location.latitude}, {location.longitude})")
            else:
                output.write(f"{address},Not Found,Not Found\n")
                print(f"Could not find coordinates for: {address}")
        except Exception as e:
            print(f"Error processing {address}: {e}")

        time.sleep(1)  # Delay to avoid overloading the server

print("Geolocation lookup complete! Check 'coordinates.txt' for results.")
