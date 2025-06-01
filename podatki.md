### Zbiranje podatkov

Za obhod raznih zaščit strani pred roboti smo uporabili knjižnico `hrequests`. Podatke smo dobili iz HTML-ja strani, saj nobena stran nima ustreznega odprtega vmesnika. Za branje podatkov iz HTML strukture smo uporabili knjižnico `BeautifulSoup`.

Vnosom iz zbirke strani `mojikvadrati.com` smo dodali še podatke o koordinatah. Za to smo uporabili knjižnico `geopy`. Koordinate smo dobili iz naslovov oglasov, vendar zaradi narave podatkov včasih koordinate niso točne.


