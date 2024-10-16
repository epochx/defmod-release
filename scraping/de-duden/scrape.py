import duden
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import json
from bs4 import BeautifulSoup
import requests

word_list = 'muse.txt'
dataset = 'duden.jsonl'
unused_words = 'unused_words.txt'

with open(word_list, 'r') as f:
    words = [i.strip() for i in f.readlines()]

def scrape(word):
    def get(word):    
            w = duden.get(word)

            name = w.name
            word_classes = w.part_of_speech
            definitions = w.meaning_overview

            word_dict = {
                "word": name,
                "definitions": definitions,
                "word_classes": word_classes
            }
            
            json_string = json.dumps(word_dict, ensure_ascii=False) + "\n"

            with open(dataset, 'a') as f:
                f.write(json_string)

    try:
        get(word)
    except:
            with open(unused_words, 'a') as f:
                f.write(word + "\n")


# Clear files
with open(dataset, 'w') as f1, open(unused_words, 'w') as f2:
    f1.write('')
    f2.write('')

# Paralellize scraping
with Pool(cpu_count()) as p:
    list(tqdm(p.imap(scrape, words), total=len(words)))
