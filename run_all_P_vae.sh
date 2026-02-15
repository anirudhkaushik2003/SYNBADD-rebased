python train_VAE_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data False --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype GatedCrossAttention --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 1 \
           --specials GatedCrossAttentionTransformer+DAFT_OnlyPast_Fold1 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS

sleep 3

python train_VAE_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data False --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype GatedCrossAttention --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 2 \
           --specials GatedCrossAttentionTransformer+DAFT_OnlyPast_Fold2 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS

sleep 3

python train_VAE_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data False --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype GatedCrossAttention --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 3 \
           --specials GatedCrossAttentionTransformer+DAFT_OnlyPast_Fold3 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS

sleep 3

python train_VAE_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data False --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype GatedCrossAttention --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 4 \
           --specials GatedCrossAttentionTransformer+DAFT_OnlyPast_Fold4 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS

sleep 3

python train_VAE_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data False --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype GatedCrossAttention --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 5 \
           --specials GatedCrossAttentionTransformer+DAFT_OnlyPast_Fold5 --project_name SYNBADD_ADNI_5FoldCrossVal_NewVAEModel_ALL_NewDatasetAndSVFS