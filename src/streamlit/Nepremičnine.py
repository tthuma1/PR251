import streamlit as st

st.set_page_config(
    page_title="PR Nepremičnine",
)

st.write("# Pozdravljen na naši streamlit aplikaciji! 👋")

st.markdown(
    """
    Izberi eno izmed strani na levi strani zaslona.👈

    Za končno poročilo obišči [https://github.com/tthuma1/PR251/blob/master/KONČNO.md](https://github.com/tthuma1/PR251/blob/master/KON%C4%8CNO.md).

    V seminarski nalogi se osredetočamo na pridobivanje znanj z analizo nepremičninskih oglasov, ki so objavljeni v aprilu 2025.
    Kot del aplikacije smo naredili več različnih napovednih modelov za napoved cene nepremičnin in interaktivno vizualizacijo
    naših podatkov. Podatke smo zajemali iz spletnih strani [nepremicnine.net](https://nepremicnine.net),
    [mojikvadrati.com](https://mojikvadrati.com) in [bolha.com](https://bolha.com).
    """
)
