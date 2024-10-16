import os
import json
from argparse import ArgumentParser
import fasttext
from huggingface_hub import hf_hub_download

import evaluate

FASTTEXT_LANGCODE_TO_LANG = {
    "de": "__label__deu_Latn",
    "en": "__label__eng_Latn",
    "fr": "__label__fra_Latn",
    "es": "__label__spa_Latn",
    "pt": "__label__por_Latn",
}

parser = ArgumentParser()
parser.add_argument("--data")


if __name__ == "__main__":
    args = parser.parse_args()

    language = os.path.basename(
                os.path.dirname(os.path.normpath(args.data))
            )

    with open(os.path.join(args.data, "output.json")) as f:
        results = json.load(f)

    metrics = {
        "bertscore": evaluate.load("bertscore")
    }

    for name, metric in metrics.items():
        metric.add_batch(
            predictions=results["predictions"],
            references=results["references"]
        )
    
    metric_values = {}
    for name, metric in metrics.items():
        if name=="bertscore":
            metric_values[name] = metric.compute(lang=language)
        else:
            pass

    metric_values["bertscore"]["mean_f1"] = sum(metric_values["bertscore"]["f1"])/len(metric_values["bertscore"]["f1"])
    metric_values["bertscore"]["mean_precision"] = sum(metric_values["bertscore"]["precision"])/len(metric_values["bertscore"]["f1"])
    metric_values["bertscore"]["mean_recall"] = sum(metric_values["bertscore"]["recall"])/len(metric_values["bertscore"]["f1"])

    print("BERTSCORE_f1", metric_values["bertscore"]["mean_f1"])
    print("BERTSCORE_recall", metric_values["bertscore"]["mean_recall"])
    print("BERTSCORE_precision", metric_values["bertscore"]["mean_precision"])

    scores_file_path = os.path.join(args.data, "bertscore.json")

    with open(scores_file_path, "w") as f:
        json.dump(metric_values, f)

