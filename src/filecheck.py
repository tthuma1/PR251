import os.path
for i in range(6798):
    isfile = os.path.isfile(f"../data/kvadrati/html_vsebina_{i}.html")
    if not isfile:
        print(i)









# import os.path
# from pathlib import Path

# for i in range(7000):
#     my_file = Path(f"../data/kvadrati/html_vsebina_{i}.html")
#     if not my_file.is_file:
#         print(i)