from pyrae import dle
from multiprocessing import Pool, cpu_count
import tqdm
import json
from itertools import chain
import os

word_list = 'listapalabras.txt'
dataset = 'drae_orig.jsonl'
words = []

# Get words from word list
with open(word_list, 'r') as f:
   words = [i.strip() for i in f.readlines()]

# Scrape function
def scrape(word):
    try:
        word_dict_raw = dle.search_by_word(word).to_dict()

        container = word_dict_raw["articles"][0]["definitions"]
        definitions = []
        word_classes = []

        for i in container:
            definitions.append(i["sentence"]["text"])
            word_classes.append(i["is"])

        try:
            word_classes_out = [[k for k,v in word_class.items() if v == True] for word_class in word_classes]
            word_classes = []

            for word_class in word_classes_out:
                if word_class == []:
                    word_classes.append(None)
                else:
                    word_classes.append(word_class[0])

        except NameError:
            word_classes = None

        word_dict = {
            "word": word,
            "definitions": definitions,
            "word_classes": word_classes
        }


        json_string = json.dumps(word_dict , ensure_ascii=False).encode('utf8').decode() + "\n"
    
        with open(dataset, 'a') as f:
            f.write(json_string)
    except:
        pass

# Parallelize
with Pool(cpu_count()) as p:
  r = list(tqdm.tqdm(p.imap(scrape, words), total=len(words)))


input_file = dataset
output_file = "drae.jsonl"

with open(input_file, "r") as f:
    input_lines = f.readlines()
    for i in input_lines:
        i = json.loads(i)
        word = i["word"]
        for j in range(len(i["definitions"])):
            definition = i["definitions"][j]
            word_class = i["word_classes"][j]
            word_data = {"word": word, "definition": definition, "word_class": word_class}
            with open(output_file, "a") as g:
                g.write(json.dumps(word_data, ensure_ascii=False) + "\n")
