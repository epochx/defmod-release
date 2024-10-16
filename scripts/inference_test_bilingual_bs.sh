# en-fr

python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/storage-kirt/results/defmod/en-fr/mt5-large/outputs/best_model \
    --dataset ~/storage-kirt/data/defmod/monosemic-lemmas-only/en-oxford/ \
    --split valid \
    --output_dir ~/storage-kirt/results/defmod/en-fr/mt5-large \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256

# 07/17/2024 12:00:36 - INFO - __main__ - {'bertscore': 0.7801358911756996}



python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/storage-kirt/results/defmod/en-fr/mt5-large/outputs/best_model \
    --dataset ~/storage-kirt/data/defmod/monosemic-lemmas-only/en-oxford/ \
    --split test \
    --output_dir ~/storage-kirt/results/defmod/en-fr/mt5-large \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256

# 10/02/2023 18:53:08 - INFO - __main__ - {'bleu': 0.03798613026267025, 'comet': 0.3452634845002025}
# oxford en-fr 0.7784523638693045


python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/storage-kirt/results/defmod/en-fr/mt5-large/outputs/best_model \
    --dataset ~/storage-kirt/data/defmod/monosemic-lemmas-only/fr-larousse/ \
    --split valid \
    --output_dir ~/storage-kirt/results/defmod/en-fr/mt5-large \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256

# larousse en-fr 0.6453915272970369


python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/storage-kirt/results/defmod/en-fr/mt5-large/outputs/best_model \
    --dataset ~/storage-kirt/data/defmod/monosemic-lemmas-only/fr-larousse/ \
    --split test \
    --output_dir ~/storage-kirt/results/defmod/en-fr/mt5-large \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256

# 10/02/2023 18:59:33 - INFO - __main__ - {'bleu': 0.10354674718872774, 'comet': 0.29413351901715284}
# larousse en-fr 0.6441284307568776


# -------------------------------------------------------------------------------------

# fr-pt
python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/storage-kirt/results/defmod/fr-pt/mt5-large/outputs/best_model \
    --dataset ~/storage-kirt/data/defmod/monosemic-lemmas-only/fr-larousse/ \
    --split valid \
    --output_dir ~/storage-kirt/results/defmod/fr-pt/mt5-large \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256

# 10/03/2023 11:41:10 - INFO - __main__ - {'bleu': 0.9035589250291715, 'comet': 0.29923387467021245}
# larousse fr-pt 0.6991029127787053


python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/storage-kirt/results/defmod/fr-pt/mt5-large/outputs/best_model \
    --dataset ~/storage-kirt/data/defmod/monosemic-lemmas-only/fr-larousse/ \
    --split test \
    --output_dir ~/storage-kirt/results/defmod/fr-pt/mt5-large \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256

# 10/03/2023 11:35:02 - INFO - __main__ - {'bleu': 0.586795114688992, 'comet': 0.29768687392752924}
# larousse fr-pt 0.6745695393364571


python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/storage-kirt/results/defmod/fr-pt/mt5-large/outputs/best_model \
    --dataset ~/storage-kirt/data/defmod/monosemic-lemmas-only/pt-dicio/ \
    --split valid \
    --output_dir ~/storage-kirt/results/defmod/fr-pt/mt5-large \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256

# 10/03/2023 12:18:00 - INFO - __main__ - {'bleu': 7.463918569160562, 'comet': 0.3178935157958392}
# dicio fr-pt 0.6860310434269796




python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/storage-kirt/results/defmod/fr-pt/mt5-large/outputs/best_model \
    --dataset ~/storage-kirt/data/defmod/monosemic-lemmas-only/pt-dicio/ \
    --split test \
    --output_dir ~/storage-kirt/results/defmod/fr-pt/mt5-large \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256

# 10/03/2023 12:37:58 - INFO - __main__ - {'bleu': 4.96929671941849, 'comet': 0.3526699798914448}
# dicio fr-pt 0.6823847930892474



# -------------------------------------------------------------------------------------

# fr-es

python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/storage-kirt/results/defmod/fr-es/mt5-large/outputs/best_model \
    --dataset ~/storage-kirt/data/defmod/monosemic-lemmas-only/fr-larousse/ \
    --split valid \
    --output_dir ~/storage-kirt/results/defmod/fr-es/mt5-large \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256

# 10/02/2023 18:18:15 - INFO - __main__ - {'bleu': 2.6027629247923842, 'comet': 0.29475883995222185}
# larousse fr-es 0.7244842019473288


python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/storage-kirt/results/defmod/fr-es/mt5-large/outputs/best_model \
    --dataset ~/storage-kirt/data/defmod/monosemic-lemmas-only/fr-larousse/ \
    --split test \
    --output_dir ~/storage-kirt/results/defmod/fr-es/mt5-large \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256

# 10/02/2023 18:35:57 - INFO - __main__ - {'bleu': 0.9363477755061251, 'comet': 0.29331652387656026}
# larousse fr-es 0.6856202103777743


python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/storage-kirt/results/defmod/fr-es/mt5-large/outputs/best_model \
    --dataset ~/storage-kirt/data/defmod/monosemic-lemmas-only/es-drae/ \
    --split valid \
    --output_dir ~/storage-kirt/results/defmod/fr-es/mt5-large \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256

# 10/02/2023 18:22:50 - INFO - __main__ - {'bleu': 8.589311694325396, 'comet': 0.37842887207197634}
# drae fr-es 0.7503847451935037


python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/storage-kirt/results/defmod/fr-es/mt5-large/outputs/best_model \
    --dataset ~/storage-kirt/data/defmod/monosemic-lemmas-only/es-drae/ \
    --split test \
    --output_dir ~/storage-kirt/results/defmod/fr-es/mt5-large \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256

# 10/02/2023 18:44:13 - INFO - __main__ - {'bleu': 8.084030053487849, 'comet': 0.381710338200489}
# 07/17/2024 12:12:52 - INFO - __main__ - {'bertscore': 0.7433180662078486}


# -----------------------------------------------------------------------------------

# es-pt

python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/storage-kirt/results/defmod/es-pt/mt5-large/outputs/best_model \
    --dataset ~/storage-kirt/data/defmod/monosemic-lemmas-only/es-drae/ \
    --split valid \
    --output_dir ~/storage-kirt/results/defmod/es-pt/mt5-large \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256

# 07/17/2024 11:34:31 - INFO - __main__ - {'bertscore': 0.7533094394102431}


python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/storage-kirt/results/defmod/es-pt/mt5-large/outputs/best_model \
    --dataset ~/storage-kirt/data/defmod/monosemic-lemmas-only/es-drae/ \
    --split test \
    --output_dir ~/storage-kirt/results/defmod/es-pt/mt5-large \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256

# 10/03/2023 10:49:17 - INFO - __main__ - {'bleu': 9.158208634325435, 'comet': 0.3820137173155193}
# 07/17/2024 11:38:07 - INFO - __main__ - {'bertscore': 0.7457184795485344}




python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/storage-kirt/results/defmod/es-pt/mt5-large/outputs/best_model \
    --dataset ~/storage-kirt/data/defmod/monosemic-lemmas-only/pt-dicio/ \
    --split valid \
    --output_dir ~/storage-kirt/results/defmod/es-pt/mt5-large \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256

# 07/17/2024 11:46:25 - INFO - __main__ - {'bertscore': 0.8270588637150209}                                


python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/storage-kirt/results/defmod/es-pt/mt5-large/outputs/best_model \
    --dataset ~/storage-kirt/data/defmod/monosemic-lemmas-only/pt-dicio/ \
    --split test \
    --output_dir ~/storage-kirt/results/defmod/es-pt/mt5-large \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256

# 10/03/2023 10:55:42 - INFO - __main__ - {'bleu': 18.626486488270157, 'comet': 0.35151163030554494}
# 07/17/2024 12:03:33 - INFO - __main__ - {'bertscore': 0.7475215907993185}


# -----------------------------------------------------------------------------------

# en-de

python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/storage-kirt/results/defmod/en-de/mt5-large/outputs/best_model \
    --dataset ~/storage-kirt/data/defmod/monosemic-lemmas-only/de-duden/ \
    --split valid \
    --output_dir ~/storage-kirt/results/defmod/de/mt5-large \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256

# 02/13/2024 15:54:34 - INFO - __main__ - {'bleu': 0.49785368781180916, 'comet': 0.3453219516648339}
# duden en-de 0.6528264680410325



python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/storage-kirt/results/defmod/en-de/mt5-large/outputs/best_model \
    --dataset ~/storage-kirt/data/defmod/monosemic-lemmas-only/de-duden/ \
    --split test \
    --output_dir ~/storage-kirt/results/defmod/de/mt5-large \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256

# 02/13/2024 14:46:27 - INFO - __main__ - {'bleu': 0.5066898015410438, 'comet': 0.345301873089944}
# duden en-de 0.6558265137562507


python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/storage-kirt/storage-matsuo/results/defmod/en-de/mt5-large/outputs/best_model \
    --dataset ~/storage-kirt/data/defmod/monosemic-lemmas-only/en-oxford/ \
    --split valid \
    --output_dir ~/storage-kirt/results/defmod/de/mt5-large \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256

# 02/13/2024 15:32:09 - INFO - __main__ - {'bleu': 0.775144763880677, 'comet': 0.3470696670169369}
# missing



python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/storage-kirt/results/defmod/en-de/mt5-large/outputs/best_model \
    --dataset ~/storage-kirt/data/defmod/monosemic-lemmas-only/en-oxford/ \
    --split test \
    --output_dir ~/storage-kirt/results/defmod/de/mt5-large \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256

# 02/13/2024 15:12:30 - INFO - __main__ - {'bleu': 0.8630810719749588, 'comet': 0.3448487151722689}
# oxford en-de 0.8392785589946862