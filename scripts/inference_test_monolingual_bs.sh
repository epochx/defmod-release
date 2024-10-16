# en-oxford ###########################################################################
python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/storage-kirt/results/defmod/en/mt5-large/outputs/best_model \
    --dataset ~/storage-kirt/data/defmod/monosemic-lemmas-only/en-oxford/ \
    --split test \
    --output_dir ~/storage-kirt/results/defmod/en/mt5-large \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256
# 10/02/2023 16:37:43 - INFO - __main__ - {'bleu': 0.13297270176589882, 'comet': 0.34525106160228003}
# 07/16/2024 18:13:56 - INFO - __main__ - {'bertscore': 0.8125677219618472}



# pt-dicio ################################################################################
python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/storage-kirt/results/defmod/pt/mt5-large/outputs/best_model \
    --dataset ~/storage-kirt/data/defmod/monosemic-lemmas-only/pt-dicio/ \
    --split test \
    --output_dir ~/storage-kirt/results/defmod/pt/mt5-large \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256
# 10/02/2023 17:39:17 - INFO - __main__ - {'bleu': 11.539366843833571, 'comet': 0.3524891874610702}
# 07/16/2024 18:43:15 - INFO - __main__ - {'bertscore': 0.7323404999939384}



# es-drae ################################################################################
# python inference.py \
#     --model_name_or_path google/mt5-large \
#     --resume_from_checkpoint ~/results/defmod/es/mt5-large/outputs/best_model \
#     --dataset ~/data/defmod/monosemic-lemmas-only/es-drae/ \
#     --split test \
#     --output_dir ~/results/defmod/es/mt5-large \
#     --per_device_eval_batch_size 64 \
#     --val_max_target_length 256

# 10/04/2023 16:36:24 - INFO - __main__ - {'bleu': 7.7063267416920125, 'comet': 0.38358782292791754}
# 07/16/2024 17:38:31 - INFO - __main__ - {'bertscore': 0.7431620908494989}

# fr-larousse ###########################################################################
python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/storage-kirt/results/defmod/fr/mt5-large/outputs/best_model \
    --dataset ~/storage-kirt/data/defmod/monosemic-lemmas-only/fr-larousse/ \
    --split test \
    --output_dir ~/storage-kirt/results/defmod/fr/mt5-large \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256 
# 10/04/2023 18:51:05 - INFO - __main__ - {'bleu': 0.49080549064771456, 'comet': 0.2998777144153404}
# 07/16/2024 18:51:31 - INFO - __main__ - {'bertscore': 0.6846063245301608}                                


# de-duden ###########################################################################

python inference.py \
    --model_name_or_path google/mt5-large \
    --resume_from_checkpoint ~/storage-kirt/results/defmod/de/mt5-large/outputs/best_model \
    --dataset ~/storage-kirt/data/defmod/monosemic-lemmas-only/de-duden/ \
    --split test \
    --output_dir ~/storage-kirt/results/defmod/de/mt5-large \
    --per_device_eval_batch_size 64 \
    --val_max_target_length 256 
    
# 02/13/2024 14:21:20 - INFO - __main__ - {'bleu': 0.9392000671489084, 'comet': 0.34641684148449353}
# 07/16/2024 19:06:31 - INFO - __main__ - {'bertscore': 0.6637163631872423}
