python test_model_subjectwise.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype GatedCrossAttention --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 1 \
           --specials GatedCrossAttentionTransformer+DAFT_Past+Future_Fold1 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_TransformerAblations --run_name SYNBADD_128_6_5e-05_0.0_2025-05-24_02-23-23_ADNI_all_longitudinal_GatedCrossAttentionTransformer+DAFT_Past+Future_Fold1

sleep 3

python test_model_subjectwise.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype GatedCrossAttention --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 2 \
           --specials GatedCrossAttentionTransformer+DAFT_Past+Future_Fold2 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_TransformerAblations --run_name SYNBADD_128_6_5e-05_0.0_2025-05-24_09-36-19_ADNI_all_longitudinal_GatedCrossAttentionTransformer+DAFT_Past+Future_Fold2

sleep 3

python test_model_subjectwise.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype GatedCrossAttention --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 3 \
           --specials GatedCrossAttentionTransformer+DAFT_Past+Future_Fold3 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_TransformerAblations --run_name SYNBADD_128_6_5e-05_0.0_2025-05-24_16-06-03_ADNI_all_longitudinal_GatedCrossAttentionTransformer+DAFT_Past+Future_Fold3

sleep 3

python test_model_subjectwise.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype GatedCrossAttention --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 4 \
           --specials GatedCrossAttentionTransformer+DAFT_Past+Future_Fold4 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_TransformerAblations --run_name SYNBADD_128_6_5e-05_0.0_2025-05-24_23-40-09_ADNI_all_longitudinal_GatedCrossAttentionTransformer+DAFT_Past+Future_Fold4

sleep 3

python test_model_subjectwise.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype GatedCrossAttention --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 5 \
           --specials GatedCrossAttentionTransformer+DAFT_Past+Future_Fold5 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS_TransformerAblations --run_name SYNBADD_128_6_5e-05_0.0_2025-05-25_09-29-25_ADNI_all_longitudinal_GatedCrossAttentionTransformer+DAFT_Past+Future_Fold5