import json
import spacy
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename")
parser.add_argument("-l", "--lang")
args = parser.parse_args()
filename = args.filename
lang = args.lang

input_file = filename
try:
    split_filename = filename.split(".")
    split_filename[-2] = split_filename[-2] + "_lemmas"
    output_file = ""
    for part in split_filename:
        output_file += part + "."
    output_file = output_file[:-1]
except:
    output_file = filename + "_lemmas"

pipelines = {"en": "en_core_web_sm", "fr": "fr_core_news_sm", "de": "de_core_news_sm", "it": "it_core_news_sm", "ja": "ja_core_news_sm", "pt": "pt_core_news_sm", "es": "es_core_news_sm"}
pipeline = pipelines[lang]
os.system(f"python -m spacy download {pipeline} > /dev/null")
nlp = spacy.load(pipeline)

input_file = filename

data = []
with open(input_file, "r") as f:
    data = [json.loads(line.strip()) for line in f.readlines()] 
words = [item["word"] for item in data]

lemmas = []
for output in tqdm(nlp.pipe(words, disable=["ner"])):
    token = output[0]
    lemmas.append(token.lemma_)
for item, lemma in tqdm(zip(data, lemmas)):
    item["lemma"] = lemma
unique_lemmas = set(lemmas)


with open(output_file, "w") as f:
    for datum in tqdm(data):
        json_datum = json.dumps(datum, ensure_ascii=False)
        f.write(json_datum + "\n")

print(f"Original data length: {len(words)}")
print(f"New data length: {len(unique_lemmas)}")