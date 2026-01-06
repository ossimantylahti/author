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
