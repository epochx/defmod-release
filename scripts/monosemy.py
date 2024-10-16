import json
import argparse
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename")
args = parser.parse_args()
filename = args.filename

input_file = filename
try:
    split_filename = filename.split(".")
    split_filename[-2] = split_filename[-2] + "_monosemic"
    output_file = ""
    for part in split_filename:
        output_file += part + "."
    output_file = output_file[:-1]
except:
    output_file = filename + "_monosemic"

with open(input_file, 'r') as f:
    items = [json.loads(item) for item in f.readlines()]

items_count = Counter([item["word"] for item in items])
items_monosemic_words = set([word for word in items_count if items_count[word] == 1])
items_monosemic = [item for item in items if item["word"] in items_monosemic_words]

with open(output_file, "w") as f:
    f.write('')
with open(output_file, "a") as f:
    for item in items_monosemic:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")