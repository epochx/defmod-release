import os
import json
from tqdm import tqdm

dictionary = input('Dictionary: ')
dataset = dictionary +'/' + dictionary + '.jsonl'

with open('muse.txt', 'r') as f:
    muse = [i.strip() for i in f.readlines()]

with open(dataset) as f:
    data = [json.loads(line) for line in f]

word_list = [i['word'] for i in data]

matches = set(word_list) & set(muse)
match_count = len(matches)

print(match_count)

