import pandas as pd

csv_file = "kvadrati.csv"
df = pd.read_csv(csv_file, encoding="utf-8")
print(df.head())
