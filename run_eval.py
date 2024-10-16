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
parser.add_argument("--language", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()

    with open(os.path.join(args.data, "output.json")) as f:
        results = json.load(f)

    if args.language:
        language = os.path.basename(
            os.path.dirname(os.path.normpath(args.data))
        )

        model_path = hf_hub_download(
            repo_id="facebook/fasttext-language-identification",
            filename="model.bin",
        )
        model = fasttext.load_model(model_path)

        langs = []
        for prediction in results["predictions"]:
            labels, probas = model.predict(prediction.replace("\n", ""))
            langs.append(labels[0])

        fasttext_lang = FASTTEXT_LANGCODE_TO_LANG[language]

        matches = sum([lang == fasttext_lang for lang in langs])

        ratio = matches / len(langs)

        print("RATIO", ratio)

        language_file_path = os.path.join(args.data, "languages.json")

        with open(language_file_path, "w") as f:
            json.dump(ratio, f)

    else:

        # we get target language from dataset path         
        language = os.path.basename(
                os.path.dirname(os.path.normpath(args.data))
            )

        
        metrics = {
            "bleu": evaluate.load("sacrebleu"),
            "comet": evaluate.load(
                "comet", config_name="Unbabel/wmt22-comet-da"
            ),
            "bertscore": evaluate.load("bertscore"),
        }

        for name, metric in metrics.items():
            if name == "comet":
                metric.add_batch(
                    predictions=results["predictions"],
                    references=results["references"],
                    sources=["" for _ in results["references"]],
                )
            elif name == "bleu":
                metric.add_batch(
                    predictions=results["predictions"],
                    references=results["references"],
                )
            elif name == "bertscore":
                metric.add_batch(
                    predictions=results["predictions"],
                    references=results["references"]
        )

        metric_values = {}
        for name, metric in metrics.items():
            if name == "bertscore":
                metric_values[name] = metric.compute(lang=language)
            else:
                metric_values[name] = metric.compute()

        metric_values = {
            name: metric.compute() for name, metric in metrics.items()
        }

        metric_values["bertscore"]["mean_f1"] = sum(
            metric_values["bertscore"]["f1"])/len(metric_values["bertscore"]["f1"])
        
        metric_values["bertscore"]["mean_precision"] = sum(
            metric_values["bertscore"]["precision"])/len(metric_values["bertscore"]["f1"])
        
        metric_values["bertscore"]["mean_recall"] = sum(
            metric_values["bertscore"]["recall"])/len(metric_values["bertscore"]["f1"])

        print("BLEU", metric_values["bleu"]["score"])
        print("COMET", metric_values["comet"]["mean_score"])
        print("BERTScore_f1", metric_values["bertscore"]["mean_f1"])

        scores_file_path = os.path.join(args.data, "scores.json")

        with open(scores_file_path, "w") as f:
            json.dump(metric_values, f)


# af als am an ar arz as ast av az azb ba bar bcl be bg bh bn bo bpy br bs bxr ca cbk ce ceb ckb co cs cv cy da de diq dsb dty dv el eml en eo es et eu fa fi fr frr fy ga gd gl gn gom gu gv he hi hif hr hsb ht hu hy ia id ie ilo io is it ja jbo jv ka kk km kn ko krc ku kv kw ky la lb lez li lmo lo lrc lt lv mai mg mhr min mk ml mn mr mrj ms mt mwl my myv mzn nah nap nds ne new nl nn no oc or os pa pam pfl pl pms pnb ps pt qu rm ro ru rue sa sah sc scn sco sd sh si sk sl so sq sr su sv sw ta te tg th tk tl tr tt tyv ug uk ur uz vec vep vi vls vo wa war wuu xal xmf yi yo yue zh
