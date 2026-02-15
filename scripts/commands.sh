# full longitudinal model
python main.py --aug_p 0 --n_epochs 2000 --n_frames 6 --use_past_data True --use_future_data True --batch_size 7 --effective_batch_size 21 --dataset ADNI --target_timepoint m18 --img_height 128 --patch_size_factor 4 --debug False --n_cpu 3 

# full longitudinal model with debugging
python main.py --aug_p 0 --n_epochs 2000 --n_frames 6 --use_past_data True --use_future_data True --batch_size 7 --effective_batch_size 21 --dataset ADNI --target_timepoint m18 --img_height 128 --patch_size_factor 4 --debug True --n_cpu 3 

# only previous
python main.py --aug_p 0 --n_epochs 2000 --n_frames 3 --use_past_data True --use_future_data False --batch_size 8 --effective_batch_size 24 --dataset ADNI --target_timepoint m18 --img_height 128 --patch_size_factor 4 --debug False --n_cpu 3 

# only cross-sectional data
python main.py --aug_p 0 --n_epochs 2000 --n_frames 6 --use_past_data False --use_future_data False --batch_size 8 --effective_batch_size 24 --dataset ADNI --target_timepoint m18 --img_height 128 --patch_size_factor 4 --debug False --n_cpu 3 

# only cross-sectional data with debugging
python main.py --aug_p 0 --n_epochs 2000 --n_frames 6 --use_past_data False --use_future_data False --batch_size 8 --effective_batch_size 24 --dataset ADNI --target_timepoint m18 --img_height 128 --patch_size_factor 4 --debug True --n_cpu 3 

# higher patience
python main.py --aug_p 0 --n_epochs 2000 --n_frames 6 --use_past_data True --use_future_data True --batch_size 7 --effective_batch_size 21 --dataset ADNI --target_timepoint m18 --img_height 128 --patch_size_factor 4 --debug False --n_cpu 3 --patience 40 

# full longitudinal SVF model for Resnet + Transformer Backbone
python train_resnet_SYNBADD_svf.py --aug_p 0 --n_epochs 200 --use_past_data True --use_future_data False --batch_size 8 --effective_batch_size 8 --dataset ADNI --target_timepoint m18 --img_height 128 --debug False --n_cpu 3 --use_val_dataset True --in_channels 3 --num_layers 2 --use_deformation_field True