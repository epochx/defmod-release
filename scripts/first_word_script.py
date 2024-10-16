import jsonlines
import argparse

# Parse the command-line arguments
parser = argparse.ArgumentParser(description='Format JSONL data.')
parser.add_argument('input_file', type=str, help='Path to the input JSONL file')
parser.add_argument('output_file', type=str, help='Path to the output JSONL file')
args = parser.parse_args()

error_count = 0

with jsonlines.open(args.input_file, 'r') as reader:
    with jsonlines.open(args.output_file, 'w') as writer:
        for line_number, json_data in enumerate(reader, start=1):
            try:
                definitions = json_data['definitions']

                # Skip words with empty definitions
                if len(definitions) == 0:
                    continue

                word = json_data['word']
                definition = definitions[0][0]  # Get the first definition only

                formatted_data = {'word': word, 'definition': definition}
                writer.write(formatted_data)
            except Exception as e:
                error_count += 1
                print(f'Error in line {line_number}: {str(e)}')

if error_count > 0:
    print(f'Total errors encountered: {error_count}')
else:
    print('No errors encountered.')

