import json
import argparse

"""
Usage example: python hf_format.py --input_file larousse.jsonl --output_file larousse.json

Input format: jsonl (extension is jsonl)
Output format: jsonl (extension is json)

Input data format: 
{"word": word, "word_class": word_class, "definition": definition, "category": category}

Output data format:
{"id": idx, "translation": {"word": word, "word_class": word_class, "definition": definition, "category": category}}
"""
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file")
parser.add_argument("-o", "--output_file")
args = parser.parse_args()

input_file = args.input_file
output_file = args.output_file


def restructure_data(input_file, output_file):
    with open(input_file, 'r') as input_f:
        with open(output_file, 'w') as output_f:
            idx = 1
            for line in input_f:
                data = json.loads(line)
                word = data['word']
                # word_class = data['word_class']
                definition = data['definition']

                new_data = {
                    "id": idx,
                    "translation": {
                        "word": word,
                        "definition": definition,
                        # "word_class": word_class,
                    }
                }

                dump = json.dumps(new_data, ensure_ascii=False)
                output_f.write(dump + '\n')
                idx += 1

if __name__ == "__main__":
    restructure_data(input_file, output_file)

