# Fold 1
## 2 Layer transformer, Past+Future data, 
python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --batch_size 4 --use_discriminator False --n_gpu 2 \
 --effective_batch_size 8 --dataset ADNI --target_timepoint m18 --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
 --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 1 \
 --specials Bidirectional_Vanilla_Fold1 --project_name SYNBADD_ADNI_5FoldCrossVal_BidirectionalTransformer_NoDisc_Exp

sleep 3

# ## 2 Layer transformer, Past data only,
python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data False --batch_size 4 --use_discriminator False --n_gpu 2 \
 --effective_batch_size 8 --dataset ADNI --target_timepoint m18 --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
 --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 1 \
 --specials Bidirectional_OnlyPast_Fold1 --project_name SYNBADD_ADNI_5FoldCrossVal_BidirectionalTransformer_NoDisc_Exp

sleep 3

# Fold 2
## 2 Layer transformer, Past+Future data, 
python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --batch_size 4 --use_discriminator False --n_gpu 2 \
 --effective_batch_size 8 --dataset ADNI --target_timepoint m18 --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
 --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 2 \
 --specials Bidirectional_Vanilla_Fold2 --project_name SYNBADD_ADNI_5FoldCrossVal_BidirectionalTransformer_NoDisc_Exp

sleep 3

# ## 2 Layer transformer, Past data only,
python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data False --batch_size 4 --use_discriminator False --n_gpu 2 \
 --effective_batch_size 8 --dataset ADNI --target_timepoint m18 --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
 --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 2 \
 --specials Bidirectional_OnlyPast_Fold2 --project_name SYNBADD_ADNI_5FoldCrossVal_BidirectionalTransformer_NoDisc_Exp

sleep 3

# Fold 3
## 2 Layer transformer, Past+Future data, 
python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --batch_size 4 --use_discriminator False --n_gpu 2 \
 --effective_batch_size 8 --dataset ADNI --target_timepoint m18 --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
 --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 3 \
 --specials Bidirectional_Vanilla_Fold3 --project_name SYNBADD_ADNI_5FoldCrossVal_BidirectionalTransformer_NoDisc_Exp

sleep 3

## 2 Layer transformer, Past data only,
python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data False --batch_size 4 --use_discriminator False --n_gpu 2 \
 --effective_batch_size 8 --dataset ADNI --target_timepoint m18 --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
 --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 3 \
 --specials Bidirectional_OnlyPast_Fold3 --project_name SYNBADD_ADNI_5FoldCrossVal_BidirectionalTransformer_NoDisc_Exp


sleep 3

# Fold 4
## 2 Layer transformer, Past+Future data, 
python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --batch_size 4 --use_discriminator False --n_gpu 2 \
 --effective_batch_size 8 --dataset ADNI --target_timepoint m18 --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
 --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 4 \
 --specials Bidirectional_Vanilla_Fold4 --project_name SYNBADD_ADNI_5FoldCrossVal_BidirectionalTransformer_NoDisc_Exp

sleep 3

## 2 Layer transformer, Past data only,
python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data False --batch_size 4 --use_discriminator False --n_gpu 2 \
 --effective_batch_size 8 --dataset ADNI --target_timepoint m18 --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
 --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 4 \
 --specials Bidirectional_OnlyPast_Fold4 --project_name SYNBADD_ADNI_5FoldCrossVal_BidirectionalTransformer_NoDisc_Exp


sleep 3

# Fold 5
## 2 Layer transformer, Past+Future data, 
python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --batch_size 4 --use_discriminator False --n_gpu 2 \
 --effective_batch_size 8 --dataset ADNI --target_timepoint m18 --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
 --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 5 \
 --specials Bidirectional_Vanilla_Fold5 --project_name SYNBADD_ADNI_5FoldCrossVal_BidirectionalTransformer_NoDisc_Exp

sleep 3

# ## 2 Layer transformer, Past data only,
python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data False --batch_size 4 --use_discriminator False --n_gpu 2 \
 --effective_batch_size 8 --dataset ADNI --target_timepoint m18 --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
 --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 5 \
 --specials Bidirectional_OnlyPast_Fold5 --project_name SYNBADD_ADNI_5FoldCrossVal_BidirectionalTransformer_NoDisc_Exp
