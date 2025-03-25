from pyaxis import pyaxis
import pandas as pd

pd.set_option('display.max_rows', 500)

# EXAMPLE_URL = 'data/prve_registracije.PX'

# px = pyaxis.parse(EXAMPLE_URL, encoding='ISO-8859-2')
# print(px['DATA'].head(100))

# df = pd.read_csv('../data/nesrece/pn2009.csv', encoding='cp1252', sep=';')
# print(df.head(100))

df = pd.read_excel('../data/prometne_obremenitve/pldp2020noo.xlsx')
print(df.head(100))