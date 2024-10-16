# en-oxford
python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/results/defmod/en/mt5-large/outputs/best_model \
    --dataset ~/data/defmod/monosemic-lemmas-only/en-oxford/ \
    --split test \
    --output_dir ~/results/defmod/en/mt5-large \
    --per_device_eval_batch_size 128 \
    --val_max_target_length 256

# 10/02/2023 16:37:43 - INFO - __main__ - {'bleu': 0.13297270176589882, 'comet': 0.34525106160228003}

python inference.py \
    --model_name_or_path facebook/mbart-large-50 \
    --resume_from_checkpoint ~/results/defmod/en/mbart-large-50/outputs/best_model \
    --dataset ~/data/defmod/monosemic-lemmas-only/en-oxford/ \
    --split test \
    --output_dir ~/results/defmod/en/mbart-large-50/ \
    --per_device_eval_batch_size 128 \
    --val_max_target_length 256

# 10/04/2023 14:39:21 - INFO - __main__ - {'bleu': 2.610408320084029, 'comet': 0.34065586093236455}

python inference.py \
    --model_name_or_path facebook/mbart-large-cc25 \
    --resume_from_checkpoint ~/results/defmod/en/mbart-large-cc25/outputs/best_model \
    --dataset ~/data/defmod/monosemic-lemmas-only/en-oxford/ \
    --split test \
    --output_dir ~/results/defmod/en/mbart-large-cc25/ \
    --per_device_eval_batch_size 128 \
    --val_max_target_length 256

# 10/05/2023 15:52:27 - INFO - __main__ - {'bleu': 0.0, 'comet': 0.3452634748975227}



# pt-dicio
python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/results/defmod/pt/mt5-large/outputs/best_model \
    --dataset ~/data/defmod/monosemic-lemmas-only/pt-dicio/ \
    --split test \
    --output_dir ~/results/defmod/pt/mt5-large \
    --per_device_eval_batch_size 128 \
    --val_max_target_length 256
# 10/02/2023 17:39:17 - INFO - __main__ - {'bleu': 11.539366843833571, 'comet': 0.3524891874610702}

python inference.py \
    --model_name_or_path facebook/mbart-large-50 \
    --resume_from_checkpoint ~/results/defmod/pt/mbart-large-50/outputs/best_model \
    --dataset ~/data/defmod/monosemic-lemmas-only/pt-dicio/ \
    --split test \
    --output_dir ~/results/defmod/pt/mbart-large-50 \
    --per_device_eval_batch_size 128 \
    --val_max_target_length 256

# 10/04/2023 15:49:04 - INFO - __main__ - {'bleu': 18.58697663999686, 'comet': 0.3506210756121747}


# es-drae ################################################################################

python inference.py \
    --model_name_or_path facebook/mbart-large-cc25 \
    --resume_from_checkpoint ~/results/defmod/es/mbart-large-cc25/outputs/best_model \
    --dataset ~/data/defmod/monosemic-lemmas-only/es-drae/ \
    --split test \
    --output_dir ~/results/defmod/es/mbart-large-cc25 \
    --per_device_eval_batch_size 128 \
    --val_max_target_length 256

# 10/02/2023 17:07:51 - INFO - __main__ - {'bleu': 7.071561433733866, 'comet': 0.3853755416409833}

python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/results/defmod/es/mt5-large/outputs/best_model \
    --dataset ~/data/defmod/monosemic-lemmas-only/es-drae/ \
    --split test \
    --output_dir ~/results/defmod/es/mt5-large \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256

# 10/04/2023 16:36:24 - INFO - __main__ - {'bleu': 7.7063267416920125, 'comet': 0.38358782292791754}
# 07/16/2024 17:38:31 - INFO - __main__ - {'bertscore': 0.7431620908494989}


python inference.py \
    --model_name_or_path facebook/mbart-large-50 \
    --resume_from_checkpoint ~/results/defmod/es/mbart-large-50/outputs/best_model \
    --dataset ~/data/defmod/monosemic-lemmas-only/es-drae/ \
    --split test \
    --output_dir ~/results/defmod/es/mbart-large-50 \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256

# 10/04/2023 18:13:48 - INFO - __main__ - {'bleu': 6.102857561990202, 'comet': 0.3825913299705457}




# fr-larousse ###########################################################################

python inference.py \
    --model_name_or_path facebook/mbart-large-cc25 \
    --resume_from_checkpoint ~/results/defmod/fr/mbart-large-cc25/outputs/best_model \
    --dataset ~/data/defmod/monosemic-lemmas-only/fr-larousse/ \
    --split test \
    --output_dir ~/results/defmod/fr/mbart-large-cc25 \
    --per_device_eval_batch_size 128 \
    --val_max_target_length 256

# 10/02/2023 17:20:15 - INFO - __main__ - {'bleu': 2.051945963771105, 'comet': 0.3010837691950566}

python inference.py \
    --model_name_or_path facebook/mbart-large-50 \
    --resume_from_checkpoint ~/results/defmod/fr/mbart-large-50/outputs/best_model \
    --dataset ~/data/defmod/monosemic-lemmas-only/fr-larousse/ \
    --split test \
    --output_dir ~/results/defmod/fr/mbart-large-50 \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256

# 10/04/2023 18:33:34 - INFO - __main__ - {'bleu': 1.7245125510842407, 'comet': 0.2978811984897026}

python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/results/defmod/fr/mt5-large/outputs/best_model \
    --dataset ~/data/defmod/monosemic-lemmas-only/fr-larousse/ \
    --split test \
    --output_dir ~/results/defmod/fr/mt5-large \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256

# 10/04/2023 18:51:05 - INFO - __main__ - {'bleu': 0.49080549064771456, 'comet': 0.2998777144153404}


# de-duden ###########################################################################

python inference.py \
    --model_name_or_path facebook/mbart-large-cc25 \
    --resume_from_checkpoint ~/storage-matsuo/results/defmod/de/mbart-large-cc25/outputs/best_model \
    --dataset ~/storage-matsuo/data/defmod/monosemic-lemmas-only/de-duden/ \
    --split test \
    --output_dir ~/storage-matsuo/results/defmod/de/mbart-large-cc25 \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256

# 02/13/2024 13:57:22 - INFO - __main__ - {'bleu': 1.7317420390719278, 'comet': 0.3475580071573418}

python inference.py \
    --model_name_or_path facebook/mbart-large-50 \
    --resume_from_checkpoint ~/storage-matsuo/results/defmod/de/mbart-large-50/outputs/best_model \
    --dataset ~/storage-matsuo/data/defmod/monosemic-lemmas-only/de-duden/ \
    --split test \
    --output_dir ~/storage-matsuo/results/defmod/de/mbart-large-50 \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256

# 02/13/2024 14:07:47 - INFO - __main__ - {'bleu': 2.4771505377263225, 'comet': 0.3474073993843685}

python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/storage-matsuo/results/defmod/de/mt5-large/outputs/best_model \
    --dataset ~/storage-matsuo/data/defmod/monosemic-lemmas-only/de-duden/ \
    --split test \
    --output_dir ~/storage-matsuo/results/defmod/de/mt5-large \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256

# 02/13/2024 14:21:20 - INFO - __main__ - {'bleu': 0.9392000671489084, 'comet': 0.34641684148449353}
