# Analiza prometnih nesreč v Sloveniji

**Opis problema**: Iz podatkov o prometnih nesrečah v Sloveniji in sorodnih podatkovnih zbirk bomo skušali najti uporabne vzorce. Najdene ugotovitve bi
se lahko uporabljale za določanje glavnih ciljev javnih iniciativ za večjo prometno varnost, pomoč avtomobilskim zavarovalnicam pri poslovanju ter
pomoč policistom pri popisovanju posamezne nesreče.

Za pomoč javnih iniciativam na bomo skušali odgovoriti na naslednja vprašanja:
1. Kakšna je korelacija med prometnimi nesrečami in obremenitvijo cest? Ali obstajajo odseki z malo prometa, ampak veliko nesrečami?
2. Ali večje število novih registracij poveča število nesreč?
3. Ali se na posameznih odsekih nesreče dogajajo pod podobnimi razmerami na cesti in vremenskimi razmerami?
4. Kateri odseki cest skozi leta postajajo nevarnejši in kateri varnejši?
5. V katerih regijah je največ nesreč pod vplivom alkohola?
6. Ali določene starostne skupine utrpijo hujše poškodbe?
8. Ali je ob praznikih več nesreč?

<br>

Iz podatkov bomo poskusili tudi dobiti informacije, ki bi bile uporabne za zavarovalnice. Poskušali bomo odgovoriti na ta vprašanja:
1. Kateri demografski dejavniki (starost, spol, vozniški staž) povečujejo tveganje za povzročitev nesreče? Ali obstajajo skupine, za katere bi morale zavarovalnice določiti višje premije?
2. Ali lahko povemo, katere izmed nesreč kažejo na poskus prevare zavarovalnice (npr. neprometna cesta, lažje poškodbe, ponoči)?

<br>

Za pomoč policistom pri popisovanju nesreče bomo poskusili predvideti posamezne atribute, glede na že vpisane vrednosti. Najprej se bomo osredotočili na naslednje:
1. Napoved resnosti poškodb glede na stanje vozišča.


### Podatki

Osnovne podatke bomo dobili iz podatkovne zbirke o nesrečah iz OPSI:
- https://podatki.gov.si/dataset/mnzpprometne-nesrece-od-leta-2009-dalje
- https://podatki.gov.si/dataset/mnzpprometne-nesrece-mesecno-od-leta-2019-dalje

Te podatke bomo združili s podatki o prometnih obremenitvah, novih registracijah in praznikih:
- https://podatki.gov.si/dataset/pldp-karte-prometnih-obremenitev
- https://podatki.gov.si/dataset/surs2222100s
- https://podatki.gov.si/dataset/seznam-praznikov-in-dela-prostih-dni-v-republiki-sloveniji

Če nam podani podatki iz slovenskih cest ne bodo dali dovolj novih znanj, si bomo pomagali še s podatkovnimi zbirkami o drugih državah:
- https://www.kaggle.com/datasets/jacksondivakarr/car-crash-dataset
- https://www.kaggle.com/datasets/arpan129/insurance-fraud-detection

Člani: Tim Thuma, Gašper Dobrovoljc, Žan Ambrožič, Maj Zorko, Matej Stipetić
