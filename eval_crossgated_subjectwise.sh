python test_model_subjectwise.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype CrossAttentionGatedTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 1 \
           --specials CrossAttentionGatedTransformer+DAFT_Past+Future_Fold1 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_TransformerAblations --run_name SYNBADD_128_6_5e-05_0.0_2025-05-23_17-20-56_ADNI_all_longitudinal_CrossAttentionGatedTransformer+DAFT_Past+Future_Fold1

sleep 3

python test_model_subjectwise.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype CrossAttentionGatedTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 2 \
           --specials CrossAttentionGatedTransformer+DAFT_Past+Future_Fold2 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_TransformerAblations --run_name SYNBADD_128_6_5e-05_0.0_2025-05-23_22-38-21_ADNI_all_longitudinal_CrossAttentionGatedTransformer+DAFT_Past+Future_Fold2

sleep 3

python test_model_subjectwise.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype CrossAttentionGatedTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 3 \
           --specials CrossAttentionGatedTransformer+DAFT_Past+Future_Fold3 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_TransformerAblations --run_name SYNBADD_128_6_5e-05_0.0_2025-05-23_05-38-28_ADNI_all_longitudinal_CrossAttentionGatedTransformer+DAFT_Past+Future_Fold3

sleep 3

python test_model_subjectwise.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype CrossAttentionGatedTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 4 \
           --specials CrossAttentionGatedTransformer+DAFT_Past+Future_Fold4 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_TransformerAblations --run_name SYNBADD_128_6_5e-05_0.0_2025-05-25_07-40-12_ADNI_all_longitudinal_CrossAttentionGatedTransformer+DAFT_Past+Future_Fold4

sleep 3

python test_model_subjectwise.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype CrossAttentionGatedTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 5 \
           --specials CrossAttentionGatedTransformer+DAFT_Past+Future_Fold5 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_TransformerAblations --run_name SYNBADD_128_6_5e-05_0.0_2025-05-25_14-49-46_ADNI_all_longitudinal_CrossAttentionGatedTransformer+DAFT_Past+Future_Fold5