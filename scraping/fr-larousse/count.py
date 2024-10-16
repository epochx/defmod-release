import json

def count_individual_words(file_path):
    word_count = {}
    definition_count = 0
    total_items = 0

    with open(file_path, 'r') as file:
        for line in file:
            json_data = json.loads(line.strip())
            word = json_data['word']
            definition = json_data['definition']

            # Count individual words
            if word not in word_count:
                word_count[word] = 1
            else:
                word_count[word] += 1

            # Count total definitions
            definition_count += 1

    # Count individual items
    individual_items = sum(1 for count in word_count.values() if count == 1)

    # Calculate the average number of definitions per word
    average_definitions = definition_count / len(word_count)

    return individual_items, average_definitions

# Example usage
jsonl_file_path = 'larousse.jsonl'
individual_words, average_definitions = count_individual_words(jsonl_file_path)

print(f"Number of individual words: {individual_words}")
print(f"Average number of definitions per word: {average_definitions:.2f}")

