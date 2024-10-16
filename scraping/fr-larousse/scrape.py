from bs4 import BeautifulSoup
import requests
from concurrent.futures import ThreadPoolExecutor
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os
from itertools import chain
from pprint import pprint
import shutil
import glob


main_sitemaps = [
    "sitemaps/main_sitemaps/" + s
    for s in os.listdir(
        os.path.dirname(__file__) + "/" + "sitemaps/main_sitemaps/"
    )
]
main_sitemaps = [
    [sitemap for sitemap in main_sitemaps if "dicofr" in sitemap],
    [sitemap for sitemap in main_sitemaps if "bilingues" in sitemap],
]
output_filename = "larousse.jsonl"


def urls_from_sitemap(sitemap):
    try:
        with open(sitemap) as f:
            soup = BeautifulSoup(f, "xml")
    except:
        sitemap = requests.get(sitemap).content
        soup = BeautifulSoup(sitemap, "xml")

    urls = [url.text for url in soup.find_all("loc")]

    return urls


def pre_scrape(url):
    try:
        page = requests.get(url, allow_redirects=True)
        page.encoding = page.apparent_encoding
        soup = BeautifulSoup(page.content, "lxml")
        definitions_container = soup.find(
            "article", {"class": "BlocDefinition content"}
        )

        invalid_tags = ["b", "strong", "i", "u", "em", "a"]

        word = (
            definitions_container.find("h2", {"class": "AdresseDefinition"})
            .find_all(string=True, recursive=False)[-1]
            .split(",")[0]
        )

        definitions_containers = [
            container
            for container in definitions_container.find_all("div")
            if bool("Zone-Entree" in container.get("class")[0])
            and bool(
                container.find("h2", {"class": "AdresseDefinition"})
                .find_all(string=True, recursive=False)[-1]
                .split(",")[0]
                == word
            )
        ]
        for i, container in enumerate(definitions_containers):
            temp = str(container) + "\n" + str(container.findNext("ul"))
            temp = BeautifulSoup(temp, "lxml")
            definitions_containers[i] = temp

        word_classes = []
        for container in definitions_containers:
            try:
                word_classes.append(
                    container.find("div")
                    .find("p", {"class": "CatgramDefinition"})
                    .text
                )
            except Exception as e:
                word_classes.append(None)

        definitions_items = []
        for container in definitions_containers:
            try:
                definitions_items.append(
                    container.find("ul", {"class": "Definitions"}).find_all("li")
                )
            except:
                print(f"Unusable URL: {url}")
                pre_word_data = None
                return pre_word_data

        for definitions_item in definitions_items:
            for item in definitions_item:
                try:
                    item.find("span", {"class": "numDef"}).decompose()
                    for tag in invalid_tags:
                        for match in item.find_all(tag):
                            match.unwrap()
                except AttributeError:
                    pass

        categories = []
        for i, definitions_item in enumerate(definitions_items):
            categories.append([])
            for item in definitions_item:
                if item.find("span", {"class": "indicateurDefinition"}) != None:
                    categories[i].append(
                        item.find("span", {"class": "indicateurDefinition"})
                        .text.strip()
                        .strip(".")
                    )
                    item.find(
                        "span", {"class": "indicateurDefinition"}
                    ).decompose()
                else:
                    categories[i].append(None)

        definitions = []
        for i, definitions_item in enumerate(definitions_items):
            definitions.append([])
            for item in definitions_item:
                while item.find("p") != None:
                    item.find("p").decompose()
                definition = (
                    item.text.replace("\xa0", " ")
                    .replace("\r\n\t\t\t\t", "")
                    .strip()
                )
                definitions[i].append(definition)

        # Merge similar word_classes
        word_classes_merged = []
        categories_merged = []
        definitions_merged = []

        for i, word_class in enumerate(word_classes):
            if word_class not in word_classes_merged:
                word_classes_merged.append(word_class)

                merged_categories = []
                merged_definitions = []

                for j, wc in enumerate(word_classes):
                    if wc == word_class:
                        merged_categories.extend(categories[j])
                        merged_definitions.extend(definitions[j])

                categories_merged.append(merged_categories)
                definitions_merged.append(merged_definitions)

        word_classes = word_classes_merged
        categories = categories_merged
        definitions = definitions_merged

        # Organize word data
        pre_word_data = {}
        pre_word_data["word"] = word
        pre_word_data["word_classes"] = {}

        for i, word_class in enumerate(word_classes):
            word_class_data = []
            for j in range(len(categories[i])):
                word_class_data.append(
                    {
                        "category": categories[i][j],
                        "definition": definitions[i][j],
                    }
                )
            pre_word_data["word_classes"][word_class] = word_class_data
    except Exception as e:
        print(f"Unusable URL: {url}")
        pre_word_data = None

    return pre_word_data


def scrape(url):
    pre_word_data = pre_scrape(url)

    if pre_word_data != None:
        word_data = []
        word = pre_word_data["word"]
        word_classes = pre_word_data["word_classes"]

        # Iterate over word classes
        for word_class, items in word_classes.items():
            # Iterate over definitions
            for item in items:
                definition = item["definition"]
                category = item["category"]

                word_data_dict = {
                    "word": word,
                    "word_class": word_class,
                    "category": category,
                    "definition": definition,
                }

                word_data.append(word_data_dict)
    else:
        word_data = None

    return word_data


def save_scrape(url, filename="larousse.jsonl"):
    word_data = scrape(url)
    if word_data != None:
        with open(os.path.dirname(__file__) + "/" + filename, "a") as f:
            if word_data != []:
                for item in word_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")


url_list = os.path.dirname(__file__) + "/" + "sitemaps/urls/dictionary.txt"
if os.path.exists(url_list):
    with open(url_list, "r") as f:
        urls = [i.strip() for i in f.readlines()]
else:
    urls = list(
        chain.from_iterable([urls_from_sitemap(url) for url in main_sitemaps[0]])
    )
    with open(url_list, "w") as f:
        f.write("\n".join(urls))

with open(os.path.dirname(__file__) + "/" + output_filename, "w") as f:
    f.write("")
# with Pool(cpu_count()) as p:
#     r = list(tqdm(p.imap(save_scrape, urls), total=len(urls)))
for url in tqdm(urls):
    save_scrape(url)
