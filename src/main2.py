from bs4 import BeautifulSoup

# Load the HTML from the saved file
with open("../data/kvadrati_new/mojikvadratilinks.html", "r", encoding="utf-8") as file:
    html_content = file.read()

# Parse the HTML content
soup = BeautifulSoup(html_content, "html.parser")

# Find all unique hrefs starting with the desired prefix
base_url = "https://mojikvadrati.com/nepremicnina"
unique_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].startswith(base_url)]

unique_links = set(unique_links)

# for link in unique_links:
#     if unique_links.count(link) > 2:
#         print(link)

# # Print results
f = open("links2P.txt", "a")

for link in unique_links:
    f.write(link + '\n')

f.close()





# # Read the file and remove even-numbered lines
# with open("links.txt", "r", encoding="utf-8") as file:
#     lines = file.readlines()

# # Keep only odd-numbered lines (line numbers start at 1)
# filtered_lines = [line for index, line in enumerate(lines, start=1) if index % 2 != 0]

# # Write the filtered lines back to the file
# with open("links3.txt", "w", encoding="utf-8") as file:
#     file.writelines(filtered_lines)
