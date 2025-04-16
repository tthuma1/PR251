# Ko kvadratni metri spregovorijo – ali so slovenske nepremičnine "kuhane"?​

## Opis problema

V seminarski nalogi se osredetočamo na pridobivanje znanj z analizo nepremičninskih oglasov, ki so objavljeni v aprilu 2025. Naša glavna vprašanja oz. cilji so:
- Napovedovanje cene v odvisnosti od podanih atributov.
- Ali se nakup novogradenj bolj splača od nakupa starejših nepremičnin?​
- Katere nepremičninske agencije ponujajo najboljše ponudbe?​
- Kako se nepremičninski trg sklada s prebivalstvom posamezne regije?

Do vmesnega poročila smo zbrali vse ustrezne podatke in jih predstavili z raznimi vizualizacijami. V glavnem smo se osredotočili na oglase za prodajo hiš in stanovanj.

## Zajemanje podatkov

Podatke smo zajemali iz spletnih strani [nepremicnine.net](), [mojikvadrati.com]() in [bolha.com]().

Za obhod raznih zaščit strani pred roboti smo uporabili knjižnico `hrequests`. Podatke smo dobili iz HTML-ja strani, saj nobena stran nima ustreznega odprtega vmesnika. Za branje podatkov iz HTML strukture smo uporabili knjižnico `BeautifulSoup`.

Vnosom iz zbirke strani `mojikvadrati.com` smo dodali še podatke o koordinatah. Za to smo uporabili knjižnico `geopy`. Koordinate smo dobili iz naslovov oglasov, vendar zaradi narave podatkov včasih koordinate niso točne.

Skupaj smo zbrali nad 70.000 oglasov, ki smo jih zapisali v `csv` datoteke.

## Vizualizacije

Za branje podatkov v Pythonu smo uporabili `pandas` knjižnico, grafe pa smo risali z `matplotlib` in `seaborn`. Za risanje podatkov na zemljevidu smo uporabili `cartopy` in `geopandas` knjižnici.

### Splošni pogled na cene

Za začetek smo pogledali, kako se cene odražajo na zemljevidu. Spodaj je prikazan zemljevid, kjer so narisane vse cene hiš in stanovanj.

<img src="slike/mapa_cene.png" width=1000/>

Vidimo, da se pojavijo območja, kjer je veliko dragih nepremičnin, kot sta Ljubljana in Obala. Nekaj dragih nepremičnin se pojavi še v krajih, od koder se ljudje pogosto vozijo v Ljubljano, kot so Kranj, Domžale in Grosuplje. Drage nepremičnine se pojavijo še v Jesenicah, Kranjski gori in Novi Gorici. Zanimivo je, da je v okolici Maribora in v Savinjski regiji na voljo veliko nepremičnin, ki pa so relativno poceni.

### Povezava starosti z izbranimi atributi

Da smo preverili, ali starost gradnje vpliva na ceno, smo narisali razsevni diagram in preverili Pearsonov koeficient.

<img src="slike/cena_leto.png" width="700" />

Vidimo, da ima starost gradnje nizko stopnjo korelacije s ceno. To nam pove tudi nizek Pearsonov koeficient (0,17). Stare nepremičnine so namreč pogosto adaptirane, zaradi česar jim vrednost ne pada. V povprečju je namreč nepremičnina adaptirana po 75 letih. Je pa povprečna cena na kvadratni meter nepremičnin zgrajenih po letu 2020 kar za 23% višja od povprečne cene ostalih nepremičnin.

Kot pričakovano, so novogradnje poleg višje cene tudi bolj energetsko učinkovite. Iz tega sledi, da je tudi med ceno in energetsko učinkovitostjo korelacija, čeprav dokaj nizke stopnje.

<img src="slike/energija_leto.png" width="700" />

Za novogradnje bomo torej odšteli več denarja, ampak običajno dobimo bistveno boljši produkt.

Poglejmo si še, kje se nahaja največ novogradenj:

<img src="slike/novogradnje_map.png" width="700" />

Vidimo, da je ta mapa zelo podobna mapi dragih nepremičnin. Večina novogradenj je v večjih slovenskih mestih, prav tako pa jih je kar nekaj blizu meje na hrvaški obali.

### Nepremičninske agencije

Pogledali smo, kolikšen delež nepremičninskega trga zaseda posamezna agencija in katera ima v povprečju najboljše ponudbe.

<img src="slike/agencije_delezi.png" width=700/>

<img src="slike/prodajalci_cene.png" width=1000/>

Vidimo, da velike franšize, kot so CENTURY 21, RE/MAX in KW nimajo ogromnega tržnega deleža. Ta je namreč dokaj enakomerno razporejen med stotine manjših agencij.

Samo s pogledom na cene je težko določiti, katera agencija ima najboljše ponudbe, saj vsaka agencija prodaja velik spekter vrst nepremičnin. Njena povprečna cena se tako prilagodi vrsti nepremičnine, ki je prodajajo največ (luksuzne vile ali zanemarjene hiše).

### Cena v odvnisnosti od demografskih atributov

Pogledali smo, kako se cena na kvadratni meter spreminja z regijo in njenimi podatki o prebivalstvu.

<img src="slike/cena_na_prebivalca.png" width=600 />

<img src="slike/regije_cene_placa.png" width=600 />

Na prvem zemljevidu lahko opazujemo regije, ki imajo veliko ponudbo nepremičnin na prebivalca. To lahko namiguje k temu, da ima tam manj ljudi stalne naslove, torej gre večinoma za vikende. Tu izstopa Obala, kar ni presenetljivo, zanimivo pa je, da je razmerje zelo visoko v Zasavju.

Na naslednjem zemljevidu pa vidimo razmerje med povprečno ceno nepremičnin in povprečno neto plačo prebivalcev te regije. To nam pove, koliko mesecev bi prebivalec neke regije moral delati, da bi si lahko privoščil povprečno stanovanje/hišo, brez da upoštevamo druge mesečne stroške. Na tem zemljevidu prevladujeta Osrednjeslovenska in Obalno-kraška regija.

Kot demografski atribut smo pogledali še, če je na voljo več nepremičnin v regijah, kjer je več ločitev. Novo ločeni ljudje namreč običajno rabijo tudi novo nepremičnino.

<img src="slike/locitve_ratio_map.png" width=600/>

V Obalno-kraški ločitve najverjetneje nimajo vpliva na število nepremičnin, ker je regija bolj turistično nagnjena. Lahko bi sklepali, da pa v regijah na koncu grafa, torej Zasavska, Koroška, Posavska in Savinjska, verjetno vidita dodatno nepremičnino za vsako ločitev.

### Vikendi

<img src="slike/vikendi.png" width=400/>

Od sosednjih držav ima Italija na voljo najdražje vikende. Tu moramo upoštevati še to, da je oglasov za tujino bistveno manj kot za Sloveniji.
