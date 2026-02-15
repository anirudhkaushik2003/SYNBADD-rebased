# 2 scan masking

python robustness_exp.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype CrossAttentionTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug True --n_cpu 3 --lr 0.00005 --use_val_dataset True --num_scans 2 --mask_window False --future_impute_masking False \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 1 \
           --specials CrossAttentionTransformer+OnlyUnet_Past+Future_Fold1 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_TransformerAblations --run_name SYNBADD_128_6_5e-05_0.0_2025-05-22_13-44-20_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold1

sleep 3

python robustness_exp.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype CrossAttentionTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug True --n_cpu 3 --lr 0.00005 --use_val_dataset True --num_scans 2 --mask_window False --future_impute_masking False \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 2 \
           --specials CrossAttentionTransformer+OnlyUnet_Past+Future_Fold1 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_TransformerAblations --run_name SYNBADD_128_6_5e-05_0.0_2025-05-22_21-42-58_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold2

sleep 3

python robustness_exp.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype CrossAttentionTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug True --n_cpu 3 --lr 0.00005 --use_val_dataset True --num_scans 2 --mask_window False --future_impute_masking False \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 3 \
           --specials CrossAttentionTransformer+OnlyUnet_Past+Future_Fold1 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_TransformerAblations --run_name SYNBADD_128_6_5e-05_0.0_2025-05-24_02-29-18_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold3

sleep 3

python robustness_exp.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype CrossAttentionTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug True --n_cpu 3 --lr 0.00005 --use_val_dataset True --num_scans 2 --mask_window False --future_impute_masking False \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 4 \
           --specials CrossAttentionTransformer+OnlyUnet_Past+Future_Fold1 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_TransformerAblations --run_name SYNBADD_128_6_5e-05_0.0_2025-05-22_15-09-16_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold4

sleep 3

python robustness_exp.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype CrossAttentionTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug True --n_cpu 3 --lr 0.00005 --use_val_dataset True --num_scans 2 --mask_window False --future_impute_masking False \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 5 \
           --specials CrossAttentionTransformer+OnlyUnet_Past+Future_Fold1 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_TransformerAblations --run_name SYNBADD_128_6_5e-05_0.0_2025-05-22_21-35-52_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold5



# Future impute masking

python robustness_exp.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype CrossAttentionTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug True --n_cpu 3 --lr 0.00005 --use_val_dataset True --num_scans 1 --mask_window False --future_impute_masking True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 1 \
           --specials CrossAttentionTransformer+OnlyUnet_Past+Future_Fold1 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_TransformerAblations --run_name SYNBADD_128_6_5e-05_0.0_2025-05-22_13-44-20_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold1

sleep 3

python robustness_exp.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype CrossAttentionTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug True --n_cpu 3 --lr 0.00005 --use_val_dataset True --num_scans 1 --mask_window False --future_impute_masking True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 2 \
           --specials CrossAttentionTransformer+OnlyUnet_Past+Future_Fold1 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_TransformerAblations --run_name SYNBADD_128_6_5e-05_0.0_2025-05-22_21-42-58_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold2

sleep 3

python robustness_exp.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype CrossAttentionTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug True --n_cpu 3 --lr 0.00005 --use_val_dataset True --num_scans 1 --mask_window False --future_impute_masking True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 3 \
           --specials CrossAttentionTransformer+OnlyUnet_Past+Future_Fold1 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_TransformerAblations --run_name SYNBADD_128_6_5e-05_0.0_2025-05-24_02-29-18_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold3

sleep 3

python robustness_exp.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype CrossAttentionTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug True --n_cpu 3 --lr 0.00005 --use_val_dataset True --num_scans 1 --mask_window False --future_impute_masking True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 4 \
           --specials CrossAttentionTransformer+OnlyUnet_Past+Future_Fold1 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_TransformerAblations --run_name SYNBADD_128_6_5e-05_0.0_2025-05-22_15-09-16_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold4

sleep 3

python robustness_exp.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype CrossAttentionTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug True --n_cpu 3 --lr 0.00005 --use_val_dataset True --num_scans 1 --mask_window False --future_impute_masking True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 5 \
           --specials CrossAttentionTransformer+OnlyUnet_Past+Future_Fold1 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_TransformerAblations --run_name SYNBADD_128_6_5e-05_0.0_2025-05-22_21-35-52_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold5


# consecutive 3 scan masking

python robustness_exp.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype CrossAttentionTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug True --n_cpu 3 --lr 0.00005 --use_val_dataset True --num_scans 3 --mask_window True --future_impute_masking False \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 1 \
           --specials CrossAttentionTransformer+OnlyUnet_Past+Future_Fold1 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_TransformerAblations --run_name SYNBADD_128_6_5e-05_0.0_2025-05-22_13-44-20_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold1

sleep 3

python robustness_exp.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype CrossAttentionTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug True --n_cpu 3 --lr 0.00005 --use_val_dataset True --num_scans 3 --mask_window True --future_impute_masking False \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 2 \
           --specials CrossAttentionTransformer+OnlyUnet_Past+Future_Fold1 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_TransformerAblations --run_name SYNBADD_128_6_5e-05_0.0_2025-05-22_21-42-58_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold2

sleep 3

python robustness_exp.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype CrossAttentionTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug True --n_cpu 3 --lr 0.00005 --use_val_dataset True --num_scans 3 --mask_window True --future_impute_masking False \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 3 \
           --specials CrossAttentionTransformer+OnlyUnet_Past+Future_Fold1 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_TransformerAblations --run_name SYNBADD_128_6_5e-05_0.0_2025-05-24_02-29-18_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold3

sleep 3

python robustness_exp.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype CrossAttentionTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug True --n_cpu 3 --lr 0.00005 --use_val_dataset True --num_scans 3 --mask_window True --future_impute_masking False \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 4 \
           --specials CrossAttentionTransformer+OnlyUnet_Past+Future_Fold1 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_TransformerAblations --run_name SYNBADD_128_6_5e-05_0.0_2025-05-22_15-09-16_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold4

sleep 3

python robustness_exp.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype CrossAttentionTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug True --n_cpu 3 --lr 0.00005 --use_val_dataset True --num_scans 3 --mask_window True --future_impute_masking False \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 5 \
           --specials CrossAttentionTransformer+OnlyUnet_Past+Future_Fold1 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_TransformerAblations --run_name SYNBADD_128_6_5e-05_0.0_2025-05-22_21-35-52_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold5









