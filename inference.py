#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on text translation.
"""
# You can also adapt this script on your own text translation task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
from functools import reduce
import random
from pathlib import Path
from itertools import chain
import datasets
import evaluate
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import (
    DataLoader,
    BatchSampler,
    SubsetRandomSampler,
)
from tqdm.auto import tqdm

import wandb
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    MT5Tokenizer,
    MT5TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    default_data_collator,
    GenerationConfig,
)
from transformers.utils import (
    check_min_version,
    get_full_repo_name,
)
from transformers.utils.versions import require_version
from tqdm.contrib import tenumerate


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.32.0.dev0")
logger = get_logger(__name__)
# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# wandb.init(project="defmod")


# Parsing input arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="A of dataset paths.",
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["valid", "test"],
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ```predict```."
        ),
    )

    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help=(
            "The maximum total sequence length for validation "
            "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
            "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
            "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )

    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )

    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args()

    # Sanity checks

    return args


MBART_LANGCODE_TO_LANG = {
    "de": "de_DE",
    "en": "en_XX",
    "fr": "fr_XX",
    "es": "es_XX",
    "pt": "pt_XX",
}

MT5_LANGCODE_TO_LANG = {
    "de": "German",
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
}


def get_preprocess_function(
    tokenizer,
    args,
    language,
    max_target_length,
    lang_to_id,
):
    def preprocess_function(examples):
        inputs = [ex["word"] for ex in examples["translation"]]
        targets = [ex["definition"] for ex in examples["translation"]]
        # inputs = [prefix + inp for inp in inputs]
        if isinstance(
            tokenizer,
            (
                MBartTokenizer,
                MBartTokenizerFast,
                MBart50Tokenizer,
                MBart50TokenizerFast,
            ),
        ):
            tokenizer.src_lang = MBART_LANGCODE_TO_LANG[language]
            tokenizer.tgt_lang = MBART_LANGCODE_TO_LANG[language]

        elif isinstance(tokenizer, (MT5Tokenizer, MT5TokenizerFast)):
            prefix = f"Define in {MT5_LANGCODE_TO_LANG[language]}: "
            inputs = [prefix + inp for inp in inputs]

        model_inputs = tokenizer(
            inputs,
            max_length=args.max_source_length,
            truncation=True,
        )

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["language"] = [
            lang_to_id[language] for i in range(len(inputs))
        ]

        return model_inputs

    return preprocess_function


class MultiLingualDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        output = super().__call__(features, return_tensors=return_tensors)

        output["language"] = torch.tensor(
            [feature["language"] for feature in features]
        )
        return output


class MultiLingualBatchSampler(object):
    def __init__(self, dataset_sizes, batch_size, drop_last=False):
        self.batch_size = batch_size
        current = 0
        samplers = []
        for size in dataset_sizes:
            samplers.append(SubsetRandomSampler(list(range(current, size))))
            current = current + size

        self.batch_samplers = []
        for sampler in samplers:
            self.batch_samplers.append(
                BatchSampler(sampler, batch_size, drop_last=drop_last)
            )

    def __iter__(self):
        if len(self.batch_samplers) == 1:
            iterator = iter(self.batch_samplers[0])
            while True:
                try:
                    yield next(iterator)
                except StopIteration:
                    break

        else:
            iterators = [
                iter(batch_sampler) for batch_sampler in self.batch_samplers
            ]
            finished = [False for x in range(len(iterators))]
            stop_cond = reduce(lambda x, y: not x or not y, finished)
            while stop_cond:
                for i, iterator_i in enumerate(iterators):
                    try:
                        yield next(iterator_i)
                    except StopIteration:
                        finished[i] = True
                stop_cond = reduce(lambda x, y: not x or not y, finished)

    def __len__(self):
        return sum(
            [len(batch_sampler) for batch_sampler in self.batch_samplers]
        )


def main():
    # Parse the arguments
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=args.report_to, project_dir=args.output_dir)
        if args.with_tracking
        else Accelerator()
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    id_to_lang = []

    language, dict_name = os.path.basename(
        os.path.normpath(args.dataset)
    ).split("-")
    if args.split == "valid":
        file = os.path.join(args.dataset, "val.json")
    else:
        file = os.path.join(args.dataset, "test.json")

    data_files = {}
    data_files["eval"] = file
    extension = file.split(".")[-1]

    raw_datasets = load_dataset(extension, data_files=data_files)
    raw_datasets["language"] = language
    raw_datasets["dict_name"] = dict_name
    id_to_lang.append(language)

    lang_to_id = {value: key for key, value in enumerate(id_to_lang)}
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning(
            "You are instantiating a new config instance from scratch."
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # if args.val_max_target_length is None:
    #         args.val_max_target_length = args.max_target_length

    generation_config = GenerationConfig(
        min_length=0,
        max_length=args.val_max_target_length
        if args.val_max_target_length is not None
        else config.max_length,
        num_beams=args.num_beams
        if args.num_beams is not None
        else config.num_beams,
    )

    print(generation_config)

    if args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    eval_dataset = raw_datasets["eval"]

    eval_preprocess_fn = get_preprocess_function(
        tokenizer=tokenizer,
        args=args,
        language=raw_datasets["language"],
        max_target_length=args.val_max_target_length,
        lang_to_id=lang_to_id,
    )

    with accelerator.main_process_first():
        eval_dataset = eval_dataset.map(
            eval_preprocess_fn,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Running tokenizer on valid of {raw_datasets['dict_name']}",
        )

    eval_sizes = [len(eval_dataset)]

    valid_batch_sampler = MultiLingualBatchSampler(
        eval_sizes,
        batch_size=args.per_device_eval_batch_size,
        drop_last=False,
    )

    # DataLoaders creation:
    label_pad_token_id = (
        -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )

    # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
    # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
    # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
    data_collator = MultiLingualDataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    # https://stackoverflow.com/questions/51837110/pytorch-data-loading-from-multiplnoie-different-sized-datasets

    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        # batch_size=args.per_device_eval_batch_size,
        batch_sampler=valid_batch_sampler,
        num_workers=0,
    )

    # Prepare everything with our `accelerator`.
    (
        model,
        eval_dataloader,
    ) = accelerator.prepare(
        model,
        eval_dataloader,
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # We initialize the trackers only on main process because `accelerator.log`
    # only logs on main process and we don't want empty logs/runs on other processes.
    if args.with_tracking:
        if accelerator.is_main_process:
            experiment_config = vars(args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = experiment_config[
                "lr_scheduler_type"
            ].value
            accelerator.init_trackers(
                "translation_no_trainer", experiment_config
            )

    metrics = {
        # "bleu": evaluate.load("sacrebleu"),
        # "comet": evaluate.load(
        #     "comet",
        #     config_name="Unbabel/wmt22-comet-da",
        #     gpus=1,
        #     progress_bar=True,
        # ),
        "bertscore": evaluate.load("bertscore", device="cuda:0")
    }

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if (
            args.resume_from_checkpoint is not None
            or args.resume_from_checkpoint != ""
        ):
            accelerator.print(
                f"Resumed from checkpoint: {args.resume_from_checkpoint}"
            )
            accelerator.load_state(args.resume_from_checkpoint, strict=False)
            path = os.path.basename(args.resume_from_checkpoint)

    # update the progress_bar if load from checkpoint
    best_epoch = 0
    best_score = 0

    output_output_dir = os.path.join(args.output_dir, "outputs")

    model.eval()

    samples_seen = 0
    current_inputs = []
    current_preds = []
    current_labels = []
    for step, batch in tenumerate(eval_dataloader):
        with torch.no_grad():
            forced_bos_token_id = None

            if isinstance(
                tokenizer,
                (
                    MBartTokenizer,
                    MBartTokenizerFast,
                    MBart50Tokenizer,
                    MBart50TokenizerFast,
                ),
            ):
                language_id = batch["language"][0]
                language = id_to_lang[language_id]

                forced_bos_token_id = tokenizer.lang_code_to_id[
                    MBART_LANGCODE_TO_LANG[language]
                ]

            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                # attention_mask=batch["attention_mask"],
                # **gen_kwargs,
                generation_config,
                forced_bos_token_id=forced_bos_token_id,
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]
            # We did not pad to max length, we need to pad the labels too
            labels = accelerator.pad_across_processes(
                batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = (
                accelerator.gather(generated_tokens).cpu().numpy()
            )
            labels = accelerator.gather(labels).cpu().numpy()

            if args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(
                    labels != -100, labels, tokenizer.pad_token_id
                )

            decoded_preds = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            decoded_labels = tokenizer.batch_decode(
                labels, skip_special_tokens=True
            )

            decoded_preds, decoded_labels = postprocess_text(
                decoded_preds, decoded_labels
            )

            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    decoded_preds = decoded_preds[
                        : len(eval_dataloader.dataset) - samples_seen
                    ]
                    decoded_labels = decoded_labels[
                        : len(eval_dataloader.dataset) - samples_seen
                    ]
                else:
                    samples_seen += len(decoded_labels)

            for name, metric in metrics.items():
                if name == "comet":
                    metric.add_batch(
                        predictions=[item[0] for item in decoded_preds],
                        references=[item[0] for item in decoded_labels],
                        sources=["" for _ in decoded_labels],
                    )
                elif name == "bleu":
                    metric.add_batch(
                        predictions=decoded_preds,
                        references=decoded_labels,
                    )
                elif name == "bertscore":
                    metric.add_batch(
                        predictions=decoded_preds,
                        references=decoded_labels,
                    )

            # Save inputs, preds, and labels
            decoded_inputs = tokenizer.batch_decode(
                batch["input_ids"], skip_special_tokens=True
            )

            for i in range(len(decoded_preds)):
                current_inputs.append(decoded_inputs[i])
                current_preds.append(decoded_preds[i])
                current_labels.append(decoded_labels[i][0])

            # if step == 10:
            #     break

    if not os.path.exists(output_output_dir):
        os.makedirs(output_output_dir)

    current_output = [
        {"word": word, "definition": definition, "generated": generated}
        for word, definition, generated in zip(
            current_inputs, current_labels, current_preds
        )
    ]

    with open(
        os.path.join(
            output_output_dir,
            f"output_{language}-{dict_name}_{args.split}.json",
        ),
        "w",
    ) as f:
        json.dump(current_output, f)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()

    import time

    time.sleep(5)
    del model
    torch.cuda.empty_cache()

    # import ipdb; ipdb.set_trace()

    metric_values = {}
    for name, metric in metrics.items():
        # import ipdb; ipdb.set_trace()
        print(f"Computing {name}...")
        if name == "bertscore":
            value = metric.compute(lang=language)
            metric_values[name] = value
        else:
            value = metric.compute()
            metric_values[name] = value    

    metric_values["bertscore"]["mean_f1"] = sum(
        metric_values["bertscore"]["f1"]
        )/len(metric_values["bertscore"]["f1"]
              )

    logger.info(
        {
            # "bleu": metric_values["bleu"]["score"],
            # "comet": metric_values["comet"]["mean_score"],
            "bertscore": metric_values["bertscore"]["mean_f1"],
        }
    )



if __name__ == "__main__":
    main()
