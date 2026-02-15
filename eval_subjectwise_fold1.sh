# ---------------- All Modules VAE TRANSFORMER ----------------
python test_model_subjectwise.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings True --batch_size 6 --use_discriminator False --use_bidir_transformer True --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 1 \
           --run_name SYNBADD_128_6_5e-05_0.0_2025-04-16_04-59-48_ADNI_all_longitudinal_FinalVAETransformer_Vanilla_Fold1 --project_name SYNBADD_ADNI_5FoldCrossVal_FinalVAEModel_ALL_Exp

# -------------------- UNET ABLATIONS --------------------
sleep 3
python test_model_subjectwise.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --using_daft False --use_positional_embeddings True --use_bidirectional_embeddings True --batch_size 6 --use_discriminator False --use_bidir_transformer True --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 1 \
           --run_name SYNBADD_128_6_5e-05_0.0_2025-04-19_04-35-29_ADNI_all_longitudinal_OnlyUNET_VAETransformer_Vanilla_Fold1 --project_name SYNBADD_ADNI_5FoldCrossVal_FinalVAEModel_ALL_Ablations


# -------------------- Transformer ABLATIONS NO BIDIRECTIONAL EMB --------------------
sleep 3
python test_model_subjectwise.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --using_daft False --use_positional_embeddings True --use_bidirectional_embeddings False --batch_size 6 --use_discriminator False --use_bidir_transformer True --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 1 \
           --run_name SYNBADD_128_6_5e-05_0.0_2025-04-21_16-33-24_ADNI_all_longitudinal_OnlyUNET_NoPosEmb_VAETransformer_Vanilla_Fold1 --project_name SYNBADD_ADNI_5FoldCrossVal_FinalVAEModel_ALL_Transformer_Ablations

