# Ko kvadratni metri spregovorijo – ali so slovenske nepremičnine "kuhane"?

Streamlit aplikacija je objavljena na [https://pr251-nepremicnine3.streamlit.app/](https://pr251-nepremicnine3.streamlit.app/).

## Opis problema

V seminarski nalogi se osredetočamo na pridobivanje znanj z analizo nepremičninskih oglasov, ki so objavljeni v aprilu 2025. Naša glavna vprašanja oz. cilji so:
- RQ1: Ali se nakup novogradenj bolj splača od nakupa starejših nepremičnin?
- RQ2: Katere nepremičninske agencije ponujajo najboljše ponudbe?
- RQ3: Kako se nepremičninski trg sklada s prebivalstvom posamezne regije? Kje je torej trg prenasičen in kje neizkoriščen?
- RQ4: Kateri atributi najbolj vplivajo na ceno nepremičnin?
- RQ5: Ustvarjanje napovednih modelov cene iz atributov, slik in opisov.

V glavnem smo se osredotočili na oglase za prodajo hiš in stanovanj.

## Zajemanje podatkov

Podatke smo zajemali iz spletnih strani [nepremicnine.net](https://nepremicnine.net), [mojikvadrati.com](https://mojikvadrati.com) in [bolha.com](https://bolha.com).
Skupaj smo zbrali več kot 70.000 oglasov, ki smo jih zapisali v `csv` datoteke. Podrobnejši postopek zajemanja je predstavljen v datoteki [dodatek.md](dodatek.md).

## Vizualizacije

Glej [dodatek.md](dodatek.md).


### Splošni pogled na cene

Za začetno spoznavanje s podatkovno zbirko smo pogledali, kako se cene odražajo glede na lokacijo. Spodaj je prikazan zemljevid, kjer so narisane vse cene hiš in stanovanj.

<img src="slike/mapa_cene.png" width=1000/>

Vidimo, da se pojavijo območja, kjer je veliko dragih nepremičnin, kot sta Ljubljana in Obala. Nekaj dragih nepremičnin se pojavi še v krajih, od koder se ljudje pogosto vozijo v Ljubljano, kot so Kranj, Domžale in Grosuplje. Drage nepremičnine se pojavijo še v Jesenicah, Kranjski gori in Novi Gorici. Zanimivo je, da je v okolici Maribora in v Savinjski regiji na voljo veliko nepremičnin, ki pa so relativno poceni.

### Vpliv starosti na ostale značilnosti nepremičnine

Da smo preverili, ali starost gradnje vpliva na ceno, smo narisali razsevni diagram in preverili Pearsonov koeficient.

<img src="slike/cena_leto.png" width="700" />

Vidimo, da ima starost gradnje nizko stopnjo korelacije s ceno. To nam pove tudi nizek Pearsonov koeficient (0,17). Stare nepremičnine so namreč pogosto adaptirane (v povprečju po 75 letih), zaradi česar jim vrednost ne pada. Res pa je, da je povprečna cena na kvadratni meter nepremičnin zgrajenih po letu 2020 kar za 26% višja od povprečne cene ostalih nepremičnin (2970 proti 3740 EUR/m²).

Kot pričakovano, so novogradnje poleg višje cene tudi bolj energetsko učinkovite, kar je vidno na spodnjem grafu.

<img src="slike/energija_leto.png" width="700" />

Za novogradnje bomo torej odšteli več denarja, ampak običajno dobimo boljši produkt. Kljub temu trdimo, da se nakup nepremičnin, ki so stare okoli 5 let, bolj splača od nakupa novogradenj (RQ1).

Poglejmo si še, kje se nahaja največ novogradenj:

<img src="slike/novogradnje_map.png" width="700" />

Vidimo, da je ta mapa zelo podobna mapi dragih nepremičnin. Večina novogradenj je v večjih slovenskih mestih, prav tako pa jih je kar nekaj blizu meje na hrvaški obali.

### Nepremičninske agencije

Pogledali smo, katera agencija ima v povprečju najboljše ponudbe. Spodnji graf prikazuje relativne cene za največje agencije. Tu se moramo zavedati, da te agencije nimajo ogromnega tržnega deleža, saj je ta dokaj enakomerno razporejen med stotine manjših agencij.

<img src="slike/prodajalci_cene.png" width=1000/>

Samo s pogledom na cene je težko določiti, katera agencija ima najboljše ponudbe, saj vsaka agencija prodaja velik spekter vrst nepremičnin. Njena povprečna cena se tako prilagodi vrsti nepremičnine, ki jo najpogosteje prodajajo (luksuzne vile ali zanemarjene hiše). Trdimo torej, da ne moremo določiti najboljših agencij, saj se vse prilagodijo razmeram na trgu (RQ2).

### Cena v odvnisnosti od demografskih atributov

<img src="slike/stoglasov_na_prebivalca.png" width=600 />

<img src="slike/regije_cene_placa.png" width=600 />

Na prvem zemljevidu lahko opazujemo regije, ki imajo veliko ponudbo nepremičnin na prebivalca. Na Obali je zelo veliko število oglasov na prebivalca, saj so tam večinoma na voljo vikendi. Vidimo tudi, da je v Zasavski regiji premalo ponudbe glede na število prebivalcev.

Na naslednjem zemljevidu vidimo razmerje med povprečno ceno nepremičnin in povprečno neto plačo prebivalcev te regije. To nam pove, koliko mesecev bi prebivalec neke regije moral delati, da bi si lahko privoščil povprečno stanovanje/hišo, brez da upoštevamo druge mesečne stroške. Na tem zemljevidu prevladujeta Osrednjeslovenska in Obalno-kraška regija. Najbolj ugodne regije pa so Zasavksa, Primorsko-notranjska in Pomurska regija.

Iz zgornjih grafov smo kot regijo s preveč prodaje označili Obalno-kraško. V Osrednjeslovnski in Gorenjski regiji vidimo, da je oglasov sicer dovolj, ampak so cenovno nedostopni. Prostor za razširitev prodaje smo zaznali v Zasavski in Primorsko-notranjski regiji, kjer je zaenkrat relativno malo oglasov, ki so precej poceni. Tu pa se je treba spomniti, da ti dve regiji ne vključujeta večjih gospodarskih središč, kar lahko odvrne kupce. Okolica Maribora in Celja (Podravska in Savinjska regija) na zemljevidih izgledata kot najbolj primerna prostora za iskanje nepremičnine, saj vsebujeta relativno veliko oglasov po relativno nizki ceni (RQ3).

Glej [dodatek.md](dodatek.md).

### Pomembnost atributov

Pri napovedovanju cene z uporabo metode odločitvenih dreves je, če ocenjujemo glede na tip, regijo in površino nepremičnine (pri čemer na neštevilskih atributih uporabimo metodo one-hot encoding), najbolj pomemben atribut površina (pomembnost okoli 0,73), sledita pa mu tip nepremičnine in regija (RQ4). Če pa napovedujemo ceno/m², torej le glede na tip in regijo, je najpomembnejši atribut, ali je nepremičnina tipa posest, sledijo pa ostali tipi nepremičnin in regije. Izmed regij je v obeh primerih najpomembnejša Osrednjeslovenska, nato pa Obalno-kraška. Na spodnjih slikah vidimo grafični prikaz pomembnosti atributov za absolutno in za relativno ceno.

Pomembnost atributov za absolutno ceno:

<img src="slike/atributi_absolute.png" width=900 />

Pomembnost atributov za relativno ceno:

<img src="slike/atributi_relative.png" width=900 />

### Napoved cene z regresijo

Uporabili smo več različnih regresijskih modelov za napovedovanje cene glede na površino, regijo in tip nepremičnine. Najboljše rezultate, glede na prečno preverjanje, sta dala GradientBoosting in RandomForestRegressor, pri katerem sta bili optimalni globini 5 in 6 (R² med 0,3 in 0,5). Modeli LinearRegression, Ridge in Lasso so nekoliko slabši z R² med 0 in 0,15, medtem ko R² modelov KNeighborsRegressor in MLPRegressor pogosto ne preseže 0.


### Napoved cene iz opisa

Za učenje modela smo besedilne opise pretvoriti v vektorje z uporabo knjižnice `sentence_transformers` in modela `all-MiniLM-L6-v2`.
Linearni regresijski model (RandomForest, Ridge, Lasso, ...) nam niso vrnili dobrih rezultatov,
zato smo v drugem koraku uporabili jezikovni model BERT (`bert-base-multilingual-cased`), ki pa prav tako ni vrnil uporabnega modela.

Iz tega lahko opazimo, da so opisi neprimeren podatek za napovedovanje cen nepremičnin.

### Napoved cene iz slik

Večina oglasov ima podanih eno ali več slik nepremičnine. V naši podatkovni množici smo zbrali naslovne fotografije oglasov. Z njimi smo naučili konvolucijsko nevronsko mrežo za napoved cene na kvadratni meter iz podane slike. Za osnovni model smo vzeli ResNet50 mrežo z utežmi, določenimi iz podatkovne zbirke ImageNet-1k. Osnovni model smo nato prilagodili za našo zbirko slik nepremičnin.

V povprečju se model zmoti za 810 EUR/m² oz. za 33%, kar je precej slabo. Veliko napako pripisujemo veliki količini šuma med slikami (nekatere slike ne prikazujejo nepremičnine).

Glej [dodatek.md](dodatek.md).

### Končni napovedni model

Za najbolj natančen model (RQ5) smo vzeli hibrid med modelom, ki napoveduje ceno iz slike, in med modelom, ki napoveduje ceno iz opisnih atributov. Modela sta enakomerno obtežena. S končnim modelom smo prišli do povprečne napake 28%, če za drugi model vzamemo Gradient Boosting. Model je najbolj točen pri primerih okoli povprečne cene (približno 3.000 EUR/m²), kar je razvidno tudi iz spodnjega grafa napak.

<img src="slike/image_napake.png" width=600 />

V primerjavi z Zillow Zestimate, ki ima povprečno napako 7% ([https://www.zillow.com/z/zestimate/](https://www.zillow.com/z/zestimate/)), je naš model zelo slab.

### Interaktivni zemljevid

V streamlitu smo implementirali zemljevid, ki prikazuje porazdelitev nepremičnin po Sloveniji. Z uporabo filtrov za ceno in leto gradnje lahko preverimo npr. kje se nahajajo najdražje ali najcenejše nepremičnine ter v katerih regijah je več novogradenj. Tako lahko enostavno primerjamo med različnimi območji in morda prepoznamo trende na trgu, kot so območja z intenzivnejšo gradnjo. Vabljeni na [https://pr251-nepremicnine3.streamlit.app/](https://pr251-nepremicnine3.streamlit.app/).
