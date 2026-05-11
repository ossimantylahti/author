Tämä repository sisältää Ossi Mäntylahden kirjailija-skriptit. Niiden avulla OpenAI:n tekoälyä voi käyttää koko kirjan analysointiin.

Ennakkovaatimukset:
- Varmista, että Python3 + virtuaaliympäristö ovat ajokunnossa. Tämä ohje on WSL:lle

sudo apt update
sudo apt install -y python3-venv python3-pip

Tarkista, että Pythonin versio on vähintään 3

Aja tämä siellä missä run.sh ja editoi.py ovat:

cd /mnt/c/Users/ossim/GitHub/om-author
python3 -m venv venv
source venv/bin/activate

kun annat komennon
source venv/bin/activate
promptiin pitäisi ilmestyä (venv)


Asenna riippuvuudet venviin

python -m pip install --upgrade pip
python -m pip install python-docx openai
python -m pip install --upgrade openai
sudo apt install -y dos2unix

varmista:
python -c "import docx; import openai; print('ok')"

Aseta OpenAI API -avain ympäristömuuttujaan
Lisää rivi ~/.bashrc-tiedoston loppuun:

echo 'export OPENAI_API_KEY="sk-...OMA_OPENAI_AVAIN..."' >> ~/.bashrc
source ~/.bashrc

## Äänikirjan generointi lokaalilla syntetisaattorilla (WSL)

Esimerkki Piperillä (tulokset samaan hakemistoon lähdetiedoston alle):

```bash
python3 tee_aanikirja.py \
  --renderer piper \
  --input-file /mnt/c/users/ossim/downloads/abook/prologue_audiobook_11labs_v2_ssml.xml \
  --out-dir /mnt/c/users/ossim/downloads/abook \
  --narrators-file /mnt/c/users/ossim/Github/om-author/prompt_narrators.txt \
  --voice-name Kertoja \
  --merged-file prologue_piper_merged.mp3
```

Kokorolla vastaava:

```bash
python3 tee_aanikirja.py \
  --renderer kokoro \
  --input-file /mnt/c/users/ossim/downloads/abook/prologue_audiobook_11labs_v2_ssml.xml \
  --out-dir /mnt/c/users/ossim/downloads/abook \
  --narrators-file /mnt/c/users/ossim/Github/om-author/prompt_narrators.txt \
  --voice-name Kertoja \
  --merged-file prologue_kokoro_merged.mp3
```

Huom: `--pronunciation-file` / PLS-lexicon välittyy vain ElevenLabs-rendererille. Piper/Kokoro-polussa sitä ei käytetä, joten sitä ei tarvitse liittää parametreihin.

Jos Piper antaa virheen `Unable to find voice`, lataa ääni ensin (esim. Heidi):

```bash
python -m piper.download_voices fi_FI-heidi-low
```

(Tai valitse jokin muu asennettu Piper-ääni ja anna se `--voice-id`-parametrilla.)

Skripti yrittää nyt ladata puuttuvan Piper-äänen automaattisesti ensimmäisellä ajokerralla.
Jos `-medium`-mallia ei löydy, skripti kokeilee automaattisesti vastaavaa `-low`-mallia.

## Äänikäsikirjoituksen adaptation prompt -tiedosto

`tee_aanikasikirjoitus.py` tukee nyt ulkoista adaptation-promptia tyyleille `immersive` ja `dramatic`.

- Uusi valitsin: `--adaptation-prompt-file PATH`
- Oletus: `prompt_dramatise.txt` (`--code-directory`-hakemistosta)
- Jos oletuspuuttuu, skripti varoittaa ja käyttää sisäänrakennettua minimipromptia.
- Jos käyttäjä antaa polun eksplisiittisesti ja tiedosto puuttuu, skripti lopettaa virheeseen.

Esimerkki:

```bash
python3 /mnt/c/users/ossim/github/om-author/tee_aanikasikirjoitus.py \
  --content 2.14 \
  --input "/mnt/c/users/ossim/onedrive/omat/blogit/Murha Twitch-streamissa/Mantylahti_Murder-on-Twitch-stream_INTL_en-gb 3.1 f.docx" \
  --output /mnt/c/users/ossim/downloads/abook/ \
  --narrators-file /mnt/c/users/ossim/github/om-author/prompt_narrators.txt \
  --speaker-detection openai \
  --openai-model gpt-4.1-mini \
  --debug-speakers \
  --adaptation-style immersive \
  --adaptation-model gpt-4.1-mini \
  --adaptation-prompt-file /mnt/c/users/ossim/github/om-author/prompt_dramatise.txt \
  --immersive-audio-cues ssml \
  --ambient-directory /mnt/c/users/ossim/github/om-author/ambient \
  --strict-ambient-cues
```
