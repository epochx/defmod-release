import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename")
args = parser.parse_args()
filename = args.filename

input_file = filename
try:
    split_filename = filename.split(".")
    split_filename[-2] = split_filename[-2] + "_only"
    output_file = ""
    for part in split_filename:
        output_file += part + "."
    output_file = output_file[:-1]
except:
    output_file = filename + "_only"

with open(input_file, "r") as f:
    input_data = [json.loads(i.strip()) for i in f.readlines()]
output_data = []

for i in input_data:
    if i["word"] == i["lemma"]:
        output_data.append(i)

with open(output_file, "w") as f:
    f.write('')
with open(output_file, "a") as f:
    for i in output_data:
        f.write(json.dumps(i, ensure_ascii=False) + "\n")

print(f"Original lemmatized data length: {len(input_data)}")
print(f"New lemmatized data length: {len(output_data)}")