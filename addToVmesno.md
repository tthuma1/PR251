Histogram časa za obnovo nepremičnin:

```Python
plt.figure(figsize=(10, 6))
plt.hist(df_obnova['cas_obnove'], bins=10, color='skyblue', edgecolor='black')
plt.title('Histogram časa za obnovo nepremičnin')
plt.xlabel('Število let od gradnje do adaptacije')
plt.ylabel('Število nepremičnin')
plt.grid(True)
plt.show()
```
SLIKA

Vidimo, da je večina nepremičnin bila obnovljena v ~100 do 200 letih izgradnje, vendar se pa število zviša pri okoli ~500 letih. To so večinoma "rustične nepremičnine" v naravi, gradovi...

Povprečne cene na m2 v sosednjih državah:
```Python
plt.figure(figsize=(10, 6))
povprecje.plot(kind='bar')
plt.ylabel('Povprečna cena na m²')
plt.title('Povprečna cena na m² za nepremičnine v tujini')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```
SLIKA

Italija je za vikende najdražja, ima pa le okoli 40 nepremičnin naprodaj. Hrvaška jih ima več, okoli 400, Avstrija je pa najcenejša, ampak je podatek iz le ene nepremičnine. Ostale države po Evropi večinoma nimajo nič naprodaj, ali pa so v isti situaciji kot Avstrija.


Povezava med ločitvami in nepremičninami na voljo po regijah:
```Python
plt.figure(figsize=(14, 9))
plt.bar(regije, razmerja)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Št. nepremičnin na ločitev")
plt.title("Razmerje med številom nepremičnin in ločitev po regijah")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()
```
SLIKA

V Obalno-kraški ločitve najverjetneje nimajo vpliva na število nepremičnin, ker je regija bolj turistično nagnjena. Lahko bi sklepali, da pa v regijah na koncu grafa, torej Zasavska, Koroška, Posavska in Savinjska, verjetno vidita dodatno nepremičnino za vsako ločitev.