# Fold 1
## 2 Layer transformer, Past+Future data, 
# python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --batch_size 8 \
#  --effective_batch_size 8 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00001 --use_val_dataset True \
#  --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 1 \
#  --specials DAFT_Resnet_SVF_2Layer_Vanilla_Fold1

###EXTRA!!!
python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data False --batch_size 8 \
 --effective_batch_size 8 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00001 --use_val_dataset True \
 --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 5 \
 --specials DAFT_Resnet_SVF_2Layer_OnlyPast_Fold5

sleep 3

# ## 2 Layer transformer, Past data only,
# python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data False --batch_size 8 \
#  --effective_batch_size 8 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00001 --use_val_dataset True \
#  --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 1 \
#  --specials DAFT_Resnet_SVF_2Layer_OnlyPast_Fold1

sleep 3

# ## 2 Layer transformer, Future data only
python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data False --use_future_data True --batch_size 8 \
 --effective_batch_size 8 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00001 --use_val_dataset True \
 --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 1 \
 --specials DAFT_Resnet_SVF_2Layer_OnlyFuture_Fold1

sleep 3

## 2 Layer transformer, No data,
python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data False --use_future_data False --batch_size 8 \
 --effective_batch_size 8 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00001 --use_val_dataset True \
 --in_channels 1 --num_layers 2 --seq_len 5 --use_deformation_field False --fold 1 \
 --specials DAFT_Resnet_SVF_2Layer_CrossSec_Fold1

sleep 3

# Fold 2
## 2 Layer transformer, Past+Future data, 
# python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --batch_size 8 \
#  --effective_batch_size 8 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00001 --use_val_dataset True \
#  --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 2 \
#  --specials DAFT_Resnet_SVF_2Layer_Vanilla_Fold2

sleep 3

# ## 2 Layer transformer, Past data only,
# python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data False --batch_size 8 \
#  --effective_batch_size 8 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00001 --use_val_dataset True \
#  --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 2 \
#  --specials DAFT_Resnet_SVF_2Layer_OnlyPast_Fold2

sleep 3

# ## 2 Layer transformer, Future data only
python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data False --use_future_data True --batch_size 8 \
 --effective_batch_size 8 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00001 --use_val_dataset True \
 --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 2 \
 --specials DAFT_Resnet_SVF_2Layer_OnlyFuture_Fold2

sleep 3

## 2 Layer transformer, No data,
python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data False --use_future_data False --batch_size 8 \
 --effective_batch_size 8 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00001 --use_val_dataset True \
 --in_channels 1 --num_layers 2 --seq_len 5 --use_deformation_field False --fold 2 \
 --specials DAFT_Resnet_SVF_2Layer_CrossSec_Fold2

sleep 3

# Fold 3
## 2 Layer transformer, Past+Future data, 
# python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --batch_size 8 \
#  --effective_batch_size 8 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00001 --use_val_dataset True \
#  --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 3 \
#  --specials DAFT_Resnet_SVF_2Layer_Vanilla_Fold3

sleep 3

## 2 Layer transformer, Past data only,
# python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data False --batch_size 8 \
#  --effective_batch_size 8 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00001 --use_val_dataset True \
#  --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 3 \
#  --specials DAFT_Resnet_SVF_2Layer_OnlyPast_Fold3

sleep 3

# ## 2 Layer transformer, Future data only
python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data False --use_future_data True --batch_size 8 \
 --effective_batch_size 8 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00001 --use_val_dataset True \
 --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 3 \
 --specials DAFT_Resnet_SVF_2Layer_OnlyFuture_Fold3

sleep 3

## 2 Layer transformer, No data,
python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data False --use_future_data False --batch_size 8 \
 --effective_batch_size 8 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00001 --use_val_dataset True \
 --in_channels 1 --num_layers 2 --seq_len 5 --use_deformation_field False --fold 3 \
 --specials DAFT_Resnet_SVF_2Layer_CrossSec_Fold3

sleep 3

# Fold 4
## 2 Layer transformer, Past+Future data, 
# python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --batch_size 8 \
#  --effective_batch_size 8 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00001 --use_val_dataset True \
#  --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 4 \
#  --specials DAFT_Resnet_SVF_2Layer_Vanilla_Fold4

sleep 3

## 2 Layer transformer, Past data only,
# python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data False --batch_size 8 \
#  --effective_batch_size 8 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00001 --use_val_dataset True \
#  --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 4 \
#  --specials DAFT_Resnet_SVF_2Layer_OnlyPast_Fold4

sleep 3

# ## 2 Layer transformer, Future data only
python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data False --use_future_data True --batch_size 8 \
 --effective_batch_size 8 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00001 --use_val_dataset True \
 --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 4 \
 --specials DAFT_Resnet_SVF_2Layer_OnlyFuture_Fold4

sleep 3

## 2 Layer transformer, No data,
python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data False --use_future_data False --batch_size 8 \
 --effective_batch_size 8 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00001 --use_val_dataset True \
 --in_channels 1 --num_layers 2 --seq_len 5 --use_deformation_field False --fold 4 \
 --specials DAFT_Resnet_SVF_2Layer_CrossSec_Fold4

sleep 3

# Fold 5
## 2 Layer transformer, Past+Future data, 
# python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data True --batch_size 8 \
#  --effective_batch_size 8 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00001 --use_val_dataset True \
#  --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 5 \
#  --specials DAFT_Resnet_SVF_2Layer_Vanilla_Fold5

sleep 3

# ## 2 Layer transformer, Past data only,
# python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data False --batch_size 8 \
#  --effective_batch_size 8 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00001 --use_val_dataset True \
#  --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 5 \
#  --specials DAFT_Resnet_SVF_2Layer_OnlyPast_Fold5

sleep 3

## 2 Layer transformer, Future data only
python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data False --use_future_data True --batch_size 8 \
 --effective_batch_size 8 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00001 --use_val_dataset True \
 --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 5 \
 --specials DAFT_Resnet_SVF_2Layer_OnlyFuture_Fold5

sleep 3

## 2 Layer transformer, No data,
python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data False --use_future_data False --batch_size 8 \
 --effective_batch_size 8 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00001 --use_val_dataset True \
 --in_channels 1 --num_layers 2 --seq_len 5 --use_deformation_field False --fold 5 \
 --specials DAFT_Resnet_SVF_2Layer_CrossSec_Fold5
