import os
import json
from argparse import ArgumentParser
import torch

from random import sample, shuffle
from tqdm import tqdm
import math

import logging

# import evaluate

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    LlamaTokenizer,
)

from transformers.models.llama.tokenization_llama import DEFAULT_SYSTEM_PROMPT


logging.basicConfig(level=logging.CRITICAL)


torch.set_default_dtype(torch.float32)

parser = ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--data")
parser.add_argument("--output_dir")
parser.add_argument("--batch_size", type=int, default=1)

parser.add_argument("--load_in_8bit", action="store_true")
parser.add_argument("--load_in_4bit", action="store_true")
parser.add_argument("--num_shots", type=int, default=0)
parser.add_argument("--use_default_system_prompt", action="store_true")
parser.add_argument("--device_map", type=str, default="auto")
parser.add_argument("--use_bfloat16", action="store_true")


parser.add_argument(
    "--seed",
    default=1729,
    type=int,
    help="Random seed to use for experiments",
)


CUSTOM_SYSTEM_PROMPT = "You are a helpful assistant. Always answer as helpfully and concisely as possible."

MT5_LANGCODE_TO_LANG = {
    "de": "German",
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
}


def build_chat(item, language, add_examples=None):
    chat = []
    if add_examples:
        for example in examples:
            ex_word = example["translation"]["word"]
            ex_definition = example["translation"]["definition"]

            chat_example = [
                {
                    "role": "user",
                    "content": f"Define the {language} word '{ex_word}'. Use only {language} to reply."
                    #                     "content": f"Please provide a definition for the {language} word '{ex_word}'."
                },
                {
                    "role": "assistant",
                    "content": ex_definition,
                },
            ]

            chat += chat_example

    word = item["translation"]["word"]

    question = [
        {
            "role": "user",
            "content": f"Define the {language} word '{word}'. Use only {language} to reply."
            #                "content": f"Please provide a definition for the {language} word '{word}'."
        },
    ]

    chat += question

    return chat


def chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


def build_model_input(chat, use_default_system_prompt=False):
    tokenizer.use_default_system_prompt = True

    prompt = tokenizer.apply_chat_template(chat, tokenize=False)

    if use_default_system_prompt is False:
        assert DEFAULT_SYSTEM_PROMPT in prompt
        prompt = prompt.replace(DEFAULT_SYSTEM_PROMPT, CUSTOM_SYSTEM_PROMPT)

    return prompt


def parse_output(prompt, output):
    candidate_answer = output[len(prompt) :].replace("[/INST]", "")
    return candidate_answer


def read_jsonl(input_file):
    with open(input_file, "r") as f:
        data = [json.loads(line.strip()) for line in f.readlines()]
    return data


if __name__ == "__main__":
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data = read_jsonl(os.path.join(args.data, "test.json"))

    model_path = os.path.normpath(args.model)
    remain, model_name = os.path.split(model_path)

    output_folder_name = model_name
    if args.load_in_4bit:
        output_folder_name = f"{output_folder_name}_4bit_{args.num_shots}shot_{args.seed}"
    elif args.load_in_8bit:
        output_folder_name = f"{output_folder_name}_8bit_{args.num_shots}shot_{args.seed}"
    else:
        output_folder_name = f"{output_folder_name}_full_{args.num_shots}shot_{args.seed}"

    results_dir = os.path.join(args.output_dir, output_folder_name)
    if os.path.exists(results_dir):
        print(f"Output path {results_dir} already exists.")
    #     exit()

    os.makedirs(results_dir, exist_ok=True)

    print(f"Set output folder to: {results_dir}")

    language_short, dict_name = os.path.basename(
        os.path.normpath(args.data)
    ).split("-")

    language = MT5_LANGCODE_TO_LANG[language_short]

    # metrics = {
    #     "bleu": evaluate.load("sacrebleu"),
    #     "comet": evaluate.load("comet", config_name="Unbabel/wmt22-comet-da"),
    # }

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.sep_token = tokenizer.eos_token
    tokenizer.cls_token = tokenizer.eos_token
    tokenizer.mask_token = tokenizer.eos_token

    if args.use_bfloat16:
        model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code="True",
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        device_map=args.device_map,
        torch_dtype=torch.bfloat16,
    )

    else:
        model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code="True",
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        device_map=args.device_map,
        # torch_dtype=torch.bfloat16
    )

        
    if not (args.load_in_8bit or args.load_in_4bit or args.use_bfloat16):
        model.cuda()

    examples = None
    if args.num_shots > 0:
        example_data = read_jsonl(os.path.join(args.data, "train.json"))
        examples = sample(example_data, args.num_shots)

    # load data
    references = []
    predictions = []
    inputs = []
    outputs = []

    if args.batch_size > 1:
        batches = chunks(data, args.batch_size)

        total = math.ceil(len(data) / args.batch_size)
        print(total)

        for batch in tqdm(batches, desc="Generating examples", total=total):
            batch_chats = [
                build_chat(item, language, add_examples=examples)
                for item in batch
            ]

            batch_prompts = [
                build_model_input(
                    chat,
                    use_default_system_prompt=args.use_default_system_prompt,
                )
                for chat in batch_chats
            ]

            prompt_lengths = [
                len(tokenizer.tokenize(prompt)) for prompt in batch_prompts
            ]

            encoded_en = tokenizer(
                batch_prompts, return_tensors="pt", padding=True
            )

            encoded_en_gpu = {
                item: value.cuda() for item, value in encoded_en.items()
            }

            batch_generated_tokens = model.generate(
                encoded_en_gpu["input_ids"],
                attention_mask=encoded_en_gpu["attention_mask"],
                do_sample=True,
                max_new_tokens=256,
                top_k=40,
                top_p=0.95,
                repetition_penalty=1,
            )

            batch_generated_text = tokenizer.batch_decode(
                batch_generated_tokens,
                skip_special_tokens=True,
            )

            batch_generated_definitions = []
            for i, length_i in enumerate(prompt_lengths):
                generated_definition = tokenizer.batch_decode(
                    batch_generated_tokens[i : i + 1, length_i:],
                    skip_special_tokens=True,
                )[0]
                batch_generated_definitions.append(generated_definition)

            predictions.extend(batch_generated_definitions)
            outputs.extend(batch_generated_text)
            references.extend(
                [item["translation"]["definition"] for item in batch]
            )
            inputs.extend([item["translation"]["word"] for item in batch])

    else:
        for item in tqdm(data):
            chat = build_chat(item, language, add_examples=examples)
            prompt = build_model_input(
                chat, use_default_system_prompt=args.use_default_system_prompt
            )
            encoded_en = tokenizer(prompt, return_tensors="pt")
            prompt_length = encoded_en["input_ids"].shape[1]

            encoded_en_gpu = {
                item: value.cuda() for item, value in encoded_en.items()
            }

            generated_tokens = model.generate(
                encoded_en_gpu["input_ids"],
                attention_mask=encoded_en_gpu["attention_mask"],
                do_sample=True,
                max_new_tokens=256,
                top_k=40,
                top_p=0.95,
                repetition_penalty=1,
            )

            generated_text = tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
            )[0]

            generated_definition = tokenizer.batch_decode(
                generated_tokens[:, prompt_length:],
                skip_special_tokens=True,
            )[0]

            predictions.append(generated_definition)
            outputs.append(generated_text)
            references.append(item["translation"]["definition"])
            inputs.append(item["translation"]["word"])

    # for name, metric in metrics.items():
    #     if name == "comet":
    #         metric.add_batch(
    #             predictions=predictions,
    #             references=references,
    #             sources=["" for _ in references],
    #         )
    #     elif name == "bleu":
    #         metric.add_batch(
    #             predictions=predictions,
    #             references=references,
    #         )

    results = {
        "outputs": outputs,
        "predictions": predictions,
        "references": references,
        "sources": inputs,
    }

    # metric_values = {name: metric.compute() for name, metric in metrics.items()}

    # print("BLEU", metric_values["bleu"]["score"])
    # print("COMET", metric_values["comet"]["mean_score"])

    results_file_path = os.path.join(results_dir, "output.json")

    # scores_file_path = os.path.join(results_dir, "scores.json")

    with open(results_file_path, "w") as f:
        json.dump(results, f)

    # with open(scores_file_path, "w") as f:
    #     json.dump(metric_values, f)
