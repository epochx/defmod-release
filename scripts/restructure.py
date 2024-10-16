import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename")
args = parser.parse_args()
filename = args.filename

input_file = filename
try:
    split_filename = filename.split(".")
    split_filename[-2] = split_filename[-2] + "_rs"
    output_file = ""
    for part in split_filename:
        output_file += part + "."
    output_file = output_file[:-1]
except:
    output_file = filename + "_rs"

try:
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
except Exception as e:
    print(e)
    import ipdb; ipdb.set_trace()
