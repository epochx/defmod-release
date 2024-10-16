from itertools import chain
import requests

langs = ['en', 'fr', 'es', 'it', 'de', 'pt']
all_words = []


src = input("Language: ") # Desired language

if src in langs:
    eu_lang = True
else:
    eu_lang = False

try:
    langs.remove(src)
except ValueError:
    pass

if eu_lang:
    for tgt in langs:
        url = f'https://dl.fbaipublicfiles.com/arrival/dictionaries/%s-%s.txt' % (src, tgt)
        f = requests.get(url)
        f.encoding = f.apparent_encoding
        f = f.text

        all_words = [i.split()[0] for i in f.strip().split("\n")]
        all_words = list(dict.fromkeys(all_words)) # Remove duplicates
else:
    url = f'https://dl.fbaipublicfiles.com/arrival/dictionaries/%s-en.txt' % src
    f = requests.get(url)
    f.encoding = f.apparent_encoding
    f = f.text

    all_words = [i.split()[0] for i in f.strip().split("\n")]
    all_words = list(dict.fromkeys(all_words)) # Remove duplicates

with open('muse.txt', 'w') as f:
    f.write("\n".join(all_words))
