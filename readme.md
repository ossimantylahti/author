# Author – kirjailijaskriptit

Tämä repository sisältää Ossi Mäntylahden kirjailija- ja äänikirjaskriptejä. Niiden avulla OpenAI:n malleja voi käyttää esimerkiksi kokonaisen käsikirjoituksen analysointiin, tekstin litterointiin ja äänikirjatyön eri vaiheisiin.

## Asennus (WSL / Linux)

Varmista ensin, että Python 3 ja virtuaaliympäristötuki ovat asennettuina:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip
```

Luo repositoryn juuressa virtuaaliympäristö ja aktivoi se:

```bash
python3 -m venv venv
source venv/bin/activate
```

Promptiin pitäisi ilmestyä `(venv)`.

Asenna Python-riippuvuudet:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`editoi.py` tarkistaa käynnistyessään myös, että `openai`-kirjasto on riittävän uusi sen käyttämälle Responses API- ja prompt caching -rajapinnalle. Jos SDK on liian vanha tai rikki, skripti tulostaa korjauskomennon. Tarvittaessa voit päivittää olennaiset paketit käsin:

```bash
python -m pip install --upgrade openai python-docx
```

Nopea tarkistus:

```bash
python -c "import docx; import openai; print('ok')"
```

---

# OpenAI API:n käyttöönotto

`editoi.py` käyttää OpenAI API:a. **ChatGPT-tilaus ja API-laskutus ovat eri asioita**: ChatGPT Plus/Pro/Business tms. ei itsessään lisää saldoa API-tilille.

## 1. Lisää API-tilille maksutapa / krediittejä

OpenAI API:n billing-näkymä:

https://platform.openai.com/account/billing/overview

Sieltä voi lisätä maksutiedot, ostaa prepaid-krediittejä ja hallita automaattista saldon latausta tilin asetuksista riippuen.

API-käytön ja kustannusten seuranta:

https://platform.openai.com/usage

## 2. Luo API-avain

API-avaimet luodaan ja hallitaan täällä:

https://platform.openai.com/api-keys

Luo uusi secret key ja kopioi se talteen heti. Avaimen koko arvo näytetään vain luomishetkellä.

**Älä tallenna API-avainta tähän repositoryyn, Python-tiedostoon tai muuhun versionhallittuun tiedostoon.**

## 3. Aseta ympäristömuuttujat

`editoi.py` lukee seuraavat ympäristömuuttujat:

| Muuttuja | Pakollinen | Oletus | Merkitys |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | **Kyllä** | – | OpenAI API:n secret key. Ilman tätä skripti ei käynnisty. |
| `OPENAI_MODEL` | Ei | `gpt-5.6-sol` | Ensisijaisesti käytettävä malli. Komentorivin `--model` ohittaa tämän. |

`editoi.py` ei tällä hetkellä lataa `.env`-tiedostoa automaattisesti, vaan muuttujien pitää olla prosessin ympäristössä.

Väliaikaisesti nykyiseen Bash/WSL-istuntoon:

```bash
export OPENAI_API_KEY="sk-...OMA_OPENAI_AVAIN..."
export OPENAI_MODEL="gpt-5.6-sol"   # valinnainen
```

Pysyvästi WSL/Bash-käyttöön esimerkiksi `~/.bashrc`-tiedostoon:

```bash
echo 'export OPENAI_API_KEY="sk-...OMA_OPENAI_AVAIN..."' >> ~/.bashrc
echo 'export OPENAI_MODEL="gpt-5.6-sol"' >> ~/.bashrc   # valinnainen
source ~/.bashrc
```

Tarkista avainta tulostamatta, että muuttuja näkyy Pythonille:

```bash
python -c 'import os; print("OPENAI_API_KEY OK" if os.environ.get("OPENAI_API_KEY") else "OPENAI_API_KEY PUUTTUU")'
```

---

# `editoi.py` – käsikirjoituksen analysointi

`editoi.py` on interaktiivinen käsikirjoitusanalyysityökalu. Se:

- lukee yhdellä ajolla 1–10 tiedostoa;
- lukee `.docx`-tiedostot `python-docx`-kirjastolla;
- säilyttää Wordin Heading 1 / Otsikko 1 -otsikot merkintöinä `[HEADING_1 ...]`;
- säilyttää Heading 2 / Otsikko 2 -otsikot varsinaisina lukumerkintöinä `[CHAPTER_HEADING_2 ...]`;
- lukee muut tiedostot tavallisena UTF-8-tekstinä;
- yhdistää useat tiedostot samaan analyysiaineistoon selkeillä tiedostorajoilla;
- käyttää OpenAI Responses API:a;
- käyttää mallin tukemaa prompt cachea, jotta samaa suurta käsikirjoitusta ei tarvitse käsitellä jokaisessa itsenäisessä kysymyksessä täysin uutena prefiksinä;
- jatkaa automaattisesti vastausta, jos mallin yhden vastauksen tulostusraja tulee vastaan.

## Peruskäyttö

Yksi Word-käsikirjoitus:

```bash
python3 editoi.py manuscript.docx
```

Tekstitiedosto:

```bash
python3 editoi.py manuscript.txt
```

Useita lähdetiedostoja samalla kertaa:

```bash
python3 editoi.py manuscript.docx notes.txt characters.txt
```

Enintään 10 tiedostoa voidaan antaa samalla käynnistyskerralla.

## Mallin valinta

Mallin voi valita kolmella tavalla. Etusijajärjestys on:

1. komentorivin `--model`;
2. ympäristömuuttuja `OPENAI_MODEL`;
3. koodin oletus `gpt-5.6-sol`.

Esimerkiksi:

```bash
python3 editoi.py --model=gpt-6-astra manuscript.docx
```

tai:

```bash
python3 editoi.py --model gpt-5.6-terra manuscript.docx
```

Skripti tarkistaa käynnistyksessä, onko pyydetty malli käytettävissä kyseisellä API-avaimella. Jos ei ole, se kokeilee koodiin määriteltyjä fallback-malleja järjestyksessä:

```text
gpt-5.6-sol
gpt-5.6
gpt-5.6-terra
gpt-5.6-luna
gpt-5.5
gpt-5.4
gpt-5.2
gpt-4
```

`gpt-6-astra` ei ole automaattinen fallback, koska se on tarkoituksella jätetty vain eksplisiittisesti valittavaksi.

## Interaktiiviset komennot

Kun tiedostot on ladattu, skripti näyttää promptin:

```text
Kysymys>
```

Tavallinen syöte aloittaa **uuden itsenäisen analyysin** samasta käsikirjoituksesta:

```text
Kysymys> Analysoi päähenkilön kaari luvuissa 1–10.
```

Komennot:

| Komento | Toiminto |
| --- | --- |
| `<prompt>` | Aloittaa uuden itsenäisen analyysin. |
| `/uusi <prompt>` | Sama kuin tavallinen prompti: aloittaa uuden analyysin. |
| `/jatka <prompt>` | Jatkaa viimeisimmän vastauksen samaa Responses API -ketjua. |
| `/status` | Näyttää pyydetyn ja aktiivisen mallin, SDK-versiot ja paikallisen cache keyn. |
| `/help` | Näyttää komentojen ohjeen. |
| `Ctrl-C` | Lopettaa ohjelman. |

Esimerkiksi:

```text
Kysymys> Etsi käsikirjoituksen viisi suurinta rakenteellista ongelmaa.

Kysymys> /jatka Keskity nyt vain kohtiin 2 ja 4 ja ehdota konkreettiset korjaukset.

Kysymys> /uusi Tarkista erikseen aikajanan sisäinen johdonmukaisuus.
```

Tavallinen uusi kysymys ei jatka edellisen analyysin keskusteluhistoriaa. `/jatka` on tarkoitettu tilanteeseen, jossa halutaan nimenomaan jatkaa viimeisintä vastausketjua.

## Prompt cache

Skripti muodostaa käsikirjoituksesta ja kustannustoimittajaohjeesta vakaan prompt-prefiksin sekä sille SHA-256-pohjaisen `prompt_cache_key`-avaimen.

Nykyinen toteutus käyttää:

- GPT-5.6- ja GPT-6-malleilla eksplisiittistä prompt cachea, jonka TTL on koodissa `30m`;
- GPT-5.5-, GPT-5.4- ja GPT-5.2-fallbackeilla `24h` extended cache retentionia;
- vanhemmilla fallbackeilla ei lähetetä eksplisiittisiä cache-parametreja.

`/status` näyttää paikallisen cache keyn. Skripti tulostaa myös debug-tietoa input-, output-, reasoning- ja cached token -määristä, jos API palauttaa nämä tiedot.

## Tyypillisiä virheitä

### `OPENAI_API_KEY-ympäristömuuttuja puuttuu`

Aseta avain:

```bash
export OPENAI_API_KEY="sk-..."
```

### API ilmoittaa saldon tai quotan loppuneen

Tarkista API-billing ja krediitit:

https://platform.openai.com/account/billing/overview

Huomaa, että ChatGPT-tilauksen saldo tai krediitit eivät ole sama asia kuin API Platformin saldo.

### Mallia ei löydy / malliin ei ole käyttöoikeutta

`editoi.py` yrittää automaattisesti fallback-malleja. Voit myös valita mallin itse:

```bash
python3 editoi.py --model=MODEL manuscript.docx
```

### OpenAI SDK on liian vanha

Päivitä SDK samassa virtuaaliympäristössä:

```bash
python -m pip install --upgrade openai python-docx
```

---

# YouTube-videon litterointi

YouTube-videon voi litteroida suoraan URL:stä. Myös pitkät videot pilkotaan automaattisesti. Komento tallentaa videon nimellä sekä litteroinnin `.txt`-muodossa että aikaleimallisen `.srt`-tekstityksen nykyiseen hakemistoon:

```bash
python litteroi.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Toiminto tarvitsee `yt-dlp`-paketin sekä järjestelmään asennetut `ffmpeg`- ja `ffprobe`-komennot.

Ubuntu/WSL:

```bash
sudo apt install -y ffmpeg
```

Python-riippuvuudet asentuvat repositoryn yhteisestä `requirements.txt`-tiedostosta.

---

# Äänikirjan generointi lokaalilla syntetisaattorilla (WSL)

Esimerkki Piperillä:

```bash
python3 tee_aanikirja.py \
  --renderer piper \
  --input-file /mnt/c/users/ossim/downloads/abook/prologue_audiobook_11labs_v2_ssml.xml \
  --out-dir /mnt/c/users/ossim/downloads/abook \
  --narrators-file ./prompt_narrators.txt \
  --voice-name Kertoja \
  --merged-file prologue_piper_merged.mp3
```

Kokorolla vastaava:

```bash
python3 tee_aanikirja.py \
  --renderer kokoro \
  --input-file /mnt/c/users/ossim/downloads/abook/prologue_audiobook_11labs_v2_ssml.xml \
  --out-dir /mnt/c/users/ossim/downloads/abook \
  --narrators-file ./prompt_narrators.txt \
  --voice-name Kertoja \
  --merged-file prologue_kokoro_merged.mp3
```

Huom: `--pronunciation-file` / PLS-lexicon välittyy vain ElevenLabs-rendererille. Piper/Kokoro-polussa sitä ei käytetä.

Jos Piper antaa virheen `Unable to find voice`, lataa ääni ensin, esimerkiksi Heidi:

```bash
python -m piper.download_voices fi_FI-heidi-low
```

Tai valitse jokin muu asennettu Piper-ääni ja anna se `--voice-id`-parametrilla.

Skripti yrittää ladata puuttuvan Piper-äänen automaattisesti ensimmäisellä ajokerralla. Jos `-medium`-mallia ei löydy, skripti kokeilee automaattisesti vastaavaa `-low`-mallia.

---

# Äänikäsikirjoituksen adaptation prompt -tiedosto

`tee_aanikasikirjoitus.py` tukee ulkoista adaptation-promptia tyyleille `immersive` ja `dramatic`.

- Valitsin: `--adaptation-prompt-file PATH`
- Oletus: `prompt_dramatise.txt` (`--code-directory`-hakemistosta)
- Jos oletustiedosto puuttuu, skripti varoittaa ja käyttää sisäänrakennettua minimipromptia.
- Jos käyttäjä antaa polun eksplisiittisesti ja tiedosto puuttuu, skripti lopettaa virheeseen.

Esimerkki:

```bash
python3 tee_aanikasikirjoitus.py \
  --content 2.14 \
  --input "/path/to/manuscript.docx" \
  --output /path/to/output/ \
  --narrators-file ./prompt_narrators.txt \
  --speaker-detection openai \
  --openai-model gpt-4.1-mini \
  --debug-speakers \
  --adaptation-style immersive \
  --adaptation-model gpt-4.1-mini \
  --adaptation-prompt-file ./prompt_dramatise.txt \
  --immersive-audio-cues ssml \
  --ambient-directory ./ambient \
  --strict-ambient-cues
```
