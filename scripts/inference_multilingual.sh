# for DATASET in "en-oxford" "es-drae" "fr-larousse" "pt-dicio"
# do
#     python inference.py \
#         --model_name_or_path google/mt5-large \
#         --resume_from_checkpoint ~/storage-kirt/results/defmod/fr-es-pt/mt5-large/outputs/best_model \
#         --dataset ~/storage-kirt/data/defmod/monosemic-lemmas-only/$DATASET/ \
#         --split valid \
#         --output_dir ~/storage-kirt/results/defmod/fr-es-pt/mt5-large \
#         --per_device_eval_batch_size 64 \
#         --val_max_target_length 256
#     echo "ON $DATASET valid split"

#     python inference.py \
#         --model_name_or_path google/mt5-large \
#         --resume_from_checkpoint ~/storage-kirt/results/defmod/fr-es-pt/mt5-large/outputs/best_model \
#         --dataset ~/storage-kirt/data/defmod/monosemic-lemmas-only/$DATASET/ \
#         --split test \
#         --output_dir ~/storage-kirt/results/defmod/fr-es-pt/mt5-large \
#         --per_device_eval_batch_size 64 \
#         --val_max_target_length 256
#     echo "ON $DATASET test split"
# done


# 10/05/2023 18:23:44 - INFO - __main__ - {'bleu': 0.011792393736563359, 'comet': 0.31412011835418907}     
# 07/17/2024 12:30:23 - INFO - __main__ - {'bertscore': 0.7946482722856913}
# ON en-oxford valid split 
  

# 10/05/2023 18:28:06 - INFO - __main__ - {'bleu': 9.49324144856269, 'comet': 0.379225589360381}           
# 07/17/2024 12:49:06 - INFO - __main__ - {'bertscore': 0.7553526450086523}
# ON es-drae valid split


# 10/05/2023 18:33:29 - INFO - __main__ - {'bleu': 4.353839379822764, 'comet': 0.29437725094764805}
# 07/17/2024 12:59:37 - INFO - __main__ - {'bertscore': 0.7368330167878019}        
# ON fr-larousse valid split 


# 10/05/2023 18:41:38 - INFO - __main__ - {'bleu': 43.2296811277995, 'comet': 0.31720264024702816}
# 07/17/2024 13:15:37 - INFO - __main__ - {'bertscore': 0.8401661284481006}
# ON pt-dicio valid split


# 10/05/2023 17:40:11 - INFO - __main__ - {'bleu': 18.85597594161749, 'comet': 0.35213025074192816}
# 07/17/2024 13:26:49 - INFO - __main__ - {'bertscore': 0.7476440300301513}
# ON pt-dicio test split


# 10/05/2023 17:30:29 - INFO - __main__ - {'bleu': 2.6602630490502497, 'comet': 0.29276865618456094}
# 07/17/2024 13:06:23 - INFO - __main__ - {'bertscore': 0.7046963984345193}       
# ON fr-larousse test split 

# 10/05/2023 17:24:59 - INFO - __main__ - {'bleu': 9.24481220153841, 'comet': 0.3813250063651953}          
# 07/17/2024 12:53:29 - INFO - __main__ - {'bertscore': 0.7464903074408565}
# ON es-drae test split 


# 10/05/2023 17:20:58 - INFO - __main__ - {'bleu': 0.014298195966258066, 'comet': 0.3126071496450451}      
# 07/17/2024 12:44:43 - INFO - __main__ - {'bertscore': 0.7935332383089397}
# ON en-oxford test split 

# -----------------------------------------------------------------------------------



for DATASET in "en-oxford" "es-drae" "fr-larousse" "pt-dicio"
do
    python inference.py \
        --model_name_or_path google/mt5-large \
        --resume_from_checkpoint ~/storage-kirt/results/defmod/fr-en-es-pt/mt5-large/outputs/best_model \
        --dataset ~/storage-kirt/data/defmod/monosemic-lemmas-only/$DATASET/ \
        --split valid \
        --output_dir ~/storage-kirt/results/defmod/fr-es-pt/mt5-large \
        --per_device_eval_batch_size 64 \
        --val_max_target_length 256
    echo "ON $DATASET valid split"

    python inference.py \
        --model_name_or_path google/mt5-large \
        --resume_from_checkpoint ~/storage-kirt/results/defmod/fr-en-es-pt/mt5-large/outputs/best_model \
        --dataset ~/storage-kirt/data/defmod/monosemic-lemmas-only/$DATASET/ \
        --split test \
        --output_dir ~/storage-kirt/results/defmod/fr-es-pt/mt5-large \
        --per_device_eval_batch_size 64 \
        --val_max_target_length 256
    echo "ON $DATASET test split"
done


# 10/06/2023 18:57:55 - INFO - __main__ - {'bleu': 0.5393247456250218, 'comet': 0.34738475453154005}
# 07/17/2024 17:30:44 - INFO - __main__ - {'bertscore': 0.8351805617059822}
# ON en-oxford valid split 

# 10/06/2023 19:03:28 - INFO - __main__ - {'bleu': 6.365099355161924, 'comet': 0.38018930374080095}        
# 07/17/2024 17:52:22 - INFO - __main__ - {'bertscore': 0.7369990459340126}
# ON es-drae valid split 

# 10/06/2023 19:10:33 - INFO - __main__ - {'bleu': 1.8296268916742495, 'comet': 0.2928476333465449}        
# 07/17/2024 18:08:59 - INFO - __main__ - {'bertscore': 0.7147353372679531}
# ON fr-larousse valid split

# 10/06/2023 19:33:58 - INFO - __main__ - {'bleu': 12.050057246271225, 'comet': 0.3169518719717289}
# 07/17/2024 18:48:21 - INFO - __main__ - {'bertscore': 0.737256985161749}
# ON pt-dicio valid split


# 10/06/2023 13:18:04 - INFO - __main__ - {'bleu': 0.5325853140201793, 'comet': 0.3451501711056112}        
# 07/17/2024 17:45:25 - INFO - __main__ - {'bertscore': 0.8360245919538095}
# ON en-oxford test split 

# 10/06/2023 13:24:34 - INFO - __main__ - {'bleu': 5.5028311261456535, 'comet': 0.38400799350021236}       
# 07/17/2024 18:00:23 - INFO - __main__ - {'bertscore': 0.7270140633440773}
# ON es-drae test split

# 10/06/2023 13:31:51 - INFO - __main__ - {'bleu': 0.748829580416897, 'comet': 0.29340843564723845}        
# 07/17/2024 18:17:38 - INFO - __main__ - {'bertscore': 0.6819731009681328}
# ON fr-larousse test split 

# 10/06/2023 13:56:46 - INFO - __main__ - {'bleu': 4.54257224362694, 'comet': 0.35131031589022665}
# 07/17/2024 19:19:34 - INFO - __main__ - {'bertscore': 0.6852443159527221}
# ON pt-dicio test split

