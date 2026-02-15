#only past

# UNET ABLATIONS New Dataset

python train_VAE_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data False --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings True --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype BidirectionalTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 1 \
           --specials BidirectionalTransformer+DAFT_Past+Future_Fold1 --project_name SYNBADD_ADNI_5FoldCrossVal_FinalVAEModel_ALL_Exp --run_name SYNBADD_128_6_5e-05_0.0_2025-04-16_04-59-48_ADNI_all_longitudinal_FinalVAETransformer_Vanilla_Fold1

sleep 3

python train_VAE_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data False --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings True --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype BidirectionalTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 2 \
           --specials BidirectionalTransformer+DAFT_Past+Future_Fold2 --project_name SYNBADD_ADNI_5FoldCrossVal_FinalVAEModel_ALL_Exp --run_name SYNBADD_128_6_5e-05_0.0_2025-04-16_14-29-00_ADNI_all_longitudinal_FinalVAETransformer_Vanilla_Fold2

sleep 3

python train_VAE_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data False --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings True --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype BidirectionalTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 3 \
           --specials BidirectionalTransformer+DAFT_Past+Future_Fold3 --project_name SYNBADD_ADNI_5FoldCrossVal_FinalVAEModel_ALL_Exp --run_name SYNBADD_128_6_5e-05_0.0_2025-04-17_03-35-38_ADNI_all_longitudinal_FinalVAETransformer_Vanilla_Fold3

sleep 3

python train_VAE_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data False --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings True --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype BidirectionalTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 4 \
           --specials BidirectionalTransformer+DAFT_Past+Future_Fold4 --project_name SYNBADD_ADNI_5FoldCrossVal_FinalVAEModel_ALL_Exp --run_name SYNBADD_128_6_5e-05_0.0_2025-04-18_00-45-42_ADNI_all_longitudinal_FinalVAETransformer_Vanilla_Fold4

sleep 3

python train_VAE_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data False --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings True --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype BidirectionalTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 5 \
           --specials BidirectionalTransformer+DAFT_Past+Future_Fold5 --project_name SYNBADD_ADNI_5FoldCrossVal_FinalVAEModel_ALL_Exp --run_name SYNBADD_128_6_5e-05_0.0_2025-04-18_10-33-39_ADNI_all_longitudinal_FinalVAETransformer_Vanilla_Fold5

# No DAFT

python train_VAE_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data False --using_daft False --use_positional_embeddings True --use_bidirectional_embeddings True --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype BidirectionalTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 1 \
           --specials BidirectionalTransformer+DAFT_Past+Future_Fold1 --project_name SYNBADD_ADNI_5FoldCrossVal_FinalVAEModel_ALL_Ablations --run_name SYNBADD_128_6_5e-05_0.0_2025-04-19_04-35-29_ADNI_all_longitudinal_OnlyUNET_VAETransformer_Vanilla_Fold1

sleep 3

python train_VAE_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data False --using_daft False --use_positional_embeddings True --use_bidirectional_embeddings True --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype BidirectionalTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 2 \
           --specials BidirectionalTransformer+DAFT_Past+Future_Fold2 --project_name SYNBADD_ADNI_5FoldCrossVal_FinalVAEModel_ALL_Ablations --run_name SYNBADD_128_6_5e-05_0.0_2025-04-19_22-43-55_ADNI_all_longitudinal_OnlyUNET_VAETransformer_Vanilla_Fold2

sleep 3

python train_VAE_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data False --using_daft False --use_positional_embeddings True --use_bidirectional_embeddings True --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype BidirectionalTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 3 \
           --specials BidirectionalTransformer+DAFT_Past+Future_Fold3 --project_name SYNBADD_ADNI_5FoldCrossVal_FinalVAEModel_ALL_Ablations --run_name SYNBADD_128_6_5e-05_0.0_2025-04-20_05-06-45_ADNI_all_longitudinal_OnlyUNET_VAETransformer_Vanilla_Fold3

sleep 3

python train_VAE_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data False --using_daft False --use_positional_embeddings True --use_bidirectional_embeddings True --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype BidirectionalTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 4 \
           --specials BidirectionalTransformer+DAFT_Past+Future_Fold4 --project_name SYNBADD_ADNI_5FoldCrossVal_FinalVAEModel_ALL_Ablations --run_name SYNBADD_128_6_5e-05_0.0_2025-04-20_16-48-47_ADNI_all_longitudinal_OnlyUNET_VAETransformer_Vanilla_Fold4

sleep 3

python train_VAE_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data False --using_daft False --use_positional_embeddings True --use_bidirectional_embeddings True --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype BidirectionalTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 5 \
           --specials BidirectionalTransformer+DAFT_Past+Future_Fold5 --project_name SYNBADD_ADNI_5FoldCrossVal_FinalVAEModel_ALL_Ablations --run_name SYNBADD_128_6_5e-05_0.0_2025-04-21_05-54-35_ADNI_all_longitudinal_OnlyUNET_VAETransformer_Vanilla_Fold5


# only future
# UNET ABLATIONS New Dataset

python train_VAE_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data False --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings True --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype BidirectionalTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 1 \
           --specials BidirectionalTransformer+DAFT_Past+Future_Fold1 --project_name SYNBADD_ADNI_5FoldCrossVal_FinalVAEModel_ALL_Exp --run_name SYNBADD_128_6_5e-05_0.0_2025-04-16_04-59-48_ADNI_all_longitudinal_FinalVAETransformer_Vanilla_Fold1

sleep 3

python train_VAE_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data False --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings True --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype BidirectionalTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 2 \
           --specials BidirectionalTransformer+DAFT_Past+Future_Fold2 --project_name SYNBADD_ADNI_5FoldCrossVal_FinalVAEModel_ALL_Exp --run_name SYNBADD_128_6_5e-05_0.0_2025-04-16_14-29-00_ADNI_all_longitudinal_FinalVAETransformer_Vanilla_Fold2

sleep 3

python train_VAE_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data False --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings True --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype BidirectionalTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 3 \
           --specials BidirectionalTransformer+DAFT_Past+Future_Fold3 --project_name SYNBADD_ADNI_5FoldCrossVal_FinalVAEModel_ALL_Exp --run_name SYNBADD_128_6_5e-05_0.0_2025-04-17_03-35-38_ADNI_all_longitudinal_FinalVAETransformer_Vanilla_Fold3

sleep 3

python train_VAE_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data False --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings True --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype BidirectionalTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 4 \
           --specials BidirectionalTransformer+DAFT_Past+Future_Fold4 --project_name SYNBADD_ADNI_5FoldCrossVal_FinalVAEModel_ALL_Exp --run_name SYNBADD_128_6_5e-05_0.0_2025-04-18_00-45-42_ADNI_all_longitudinal_FinalVAETransformer_Vanilla_Fold4

sleep 3

python train_VAE_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data False --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings True --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype BidirectionalTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 5 \
           --specials BidirectionalTransformer+DAFT_Past+Future_Fold5 --project_name SYNBADD_ADNI_5FoldCrossVal_FinalVAEModel_ALL_Exp --run_name SYNBADD_128_6_5e-05_0.0_2025-04-18_10-33-39_ADNI_all_longitudinal_FinalVAETransformer_Vanilla_Fold5

# No DAFT

python train_VAE_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data False --use_future_data True --using_daft False --use_positional_embeddings True --use_bidirectional_embeddings True --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype BidirectionalTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 1 \
           --specials BidirectionalTransformer+DAFT_Past+Future_Fold1 --project_name SYNBADD_ADNI_5FoldCrossVal_FinalVAEModel_ALL_Ablations --run_name SYNBADD_128_6_5e-05_0.0_2025-04-19_04-35-29_ADNI_all_longitudinal_OnlyUNET_VAETransformer_Vanilla_Fold1

sleep 3

python train_VAE_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data False --use_future_data True --using_daft False --use_positional_embeddings True --use_bidirectional_embeddings True --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype BidirectionalTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 2 \
           --specials BidirectionalTransformer+DAFT_Past+Future_Fold2 --project_name SYNBADD_ADNI_5FoldCrossVal_FinalVAEModel_ALL_Ablations --run_name SYNBADD_128_6_5e-05_0.0_2025-04-19_22-43-55_ADNI_all_longitudinal_OnlyUNET_VAETransformer_Vanilla_Fold2

sleep 3

python train_VAE_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data False --use_future_data True --using_daft False --use_positional_embeddings True --use_bidirectional_embeddings True --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype BidirectionalTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 3 \
           --specials BidirectionalTransformer+DAFT_Past+Future_Fold3 --project_name SYNBADD_ADNI_5FoldCrossVal_FinalVAEModel_ALL_Ablations --run_name SYNBADD_128_6_5e-05_0.0_2025-04-20_05-06-45_ADNI_all_longitudinal_OnlyUNET_VAETransformer_Vanilla_Fold3

sleep 3

python train_VAE_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data False --use_future_data True --using_daft False --use_positional_embeddings True --use_bidirectional_embeddings True --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype BidirectionalTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 4 \
           --specials BidirectionalTransformer+DAFT_Past+Future_Fold4 --project_name SYNBADD_ADNI_5FoldCrossVal_FinalVAEModel_ALL_Ablations --run_name SYNBADD_128_6_5e-05_0.0_2025-04-20_16-48-47_ADNI_all_longitudinal_OnlyUNET_VAETransformer_Vanilla_Fold4

sleep 3

python train_VAE_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data False --use_future_data True --using_daft False --use_positional_embeddings True --use_bidirectional_embeddings True --batch_size 6 --use_discriminator False --use_transformer True --transformer_subtype BidirectionalTransformer --svf_type baseline --n_gpu 4 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 5 \
           --specials BidirectionalTransformer+DAFT_Past+Future_Fold5 --project_name SYNBADD_ADNI_5FoldCrossVal_FinalVAEModel_ALL_Ablations --run_name SYNBADD_128_6_5e-05_0.0_2025-04-21_05-54-35_ADNI_all_longitudinal_OnlyUNET_VAETransformer_Vanilla_Fold5

             