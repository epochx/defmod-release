from multiprocessing import Pool, cpu_count
import concurrent.futures 
from tqdm import tqdm
import json
from itertools import chain
import os
from bs4 import BeautifulSoup
import requests

url_list = 'urls.txt'
dataset = 'dicio.jsonl'
# word_list = 'words.txt'

# with open(word_list, 'r') as f:
#    words = [i.strip() for i in f.readlines()]

main_sitemap_url = 'https://www.dicio.com.br/sitemap.xml'
main_sitemap_page = requests.get(main_sitemap_url).text
main_sitemap_soup = BeautifulSoup(main_sitemap_page, 'xml')
sitemap_urls = [url.text for url in main_sitemap_soup.find_all('loc')]

# Short form for word classes
word_class_form = {
    "adj": "adjetivo",
    "adj+sm": "adjetivo e substantivo masculino",
    "sm": "substantivo masculino",
    "adj+sf": "adjetivo e substantivo feminino",
    "sf": "substantivo feminino"
}

# Get all urls from sitemap
def get_urls(sitemap_url):
    sitemap_page = requests.get(sitemap_url).text
    sitemap_soup = BeautifulSoup(sitemap_page, 'xml')
    urls = [url.text for url in sitemap_soup.find_all('loc') if url.text[-4:] != ".jpg" and '-palavras' not in url.text and 'palavras-' not in url.text]
    
    return urls

# Scrape function
def scrape(url):
    try:
        soup = BeautifulSoup(requests.get(url).text, 'lxml')
        word = soup.find("h1").text
        meanings = []
        word_classes = []
        examples = []
        invalid_tags = ['b', 'u', 'strong', 'em', 'i']

        main_container = soup.find("p", {"itemprop": "description"})

        if main_container.find("span", recursive=False) != None:
            container = main_container.find_all("span", recursive=False)

            current_word_class = ''
            for span in container:
                if 'class="cl"' in str(span):
                    current_word_class = span.text
                elif 'class="etim"' in str(span):
                    pass
                else:
                    meanings.append(span.text)
                    word_classes.append(current_word_class)
        else:
            meanings = ' '.join(main_container.text.split(" ")[2:][:-2])
            try:
                word_classes = word_class_form[main_container.text.split(" ")[1]]
            except KeyError:
                try:
                    word_classes = word_class_form[main_container.text.split(" ")[0]]
                except:
                    word_classes = None

        try:
            examples = soup.find_all("div", {"class": "frase"})

            for example in examples:
                for tag in invalid_tags:
                    for match in example.find_all(tag):
                        match.unwrap()
            
            examples = [e.find(text=True, recursive=False).strip() for e in examples]
        except:
            examples = []

        if meanings != [] or None:
            word_dict = {
                "word": word,
                "definitions": meanings,
                "word_classes": word_classes,
                "examples": examples
            }

            json_string = json.dumps(word_dict, ensure_ascii=False).encode('utf8').decode() + "\n"

            with open(dataset, 'a') as f:
                f.write(json_string)
    except Exception as e:
        print(e)

# Get URL list
if os.path.exists(url_list):
    with open(url_list, 'r') as f:
        all_urls = [i.strip() for i in f.readlines()]
else:
    with Pool(cpu_count()) as p:
        all_urls = list(tqdm(p.imap(get_urls, sitemap_urls), total=len(sitemap_urls)))
    all_urls = list(chain.from_iterable(all_urls))

# Parallelize scraping
with Pool(cpu_count()) as p:
    list(tqdm(p.imap(scrape, all_urls), total=len(all_urls)))
