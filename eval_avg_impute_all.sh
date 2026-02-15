python test_avg_impute_subjectwise.py --aug_p 0 --n_epochs 1 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings True --batch_size 6 --use_discriminator False --n_gpu 2 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 1

sleep 3

python test_avg_impute_subjectwise.py --aug_p 0 --n_epochs 1 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings True --batch_size 6 --use_discriminator False --n_gpu 2 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 2

sleep 3

python test_avg_impute_subjectwise.py --aug_p 0 --n_epochs 1 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings True --batch_size 6 --use_discriminator False --n_gpu 2 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 3

sleep 3

python test_avg_impute_subjectwise.py --aug_p 0 --n_epochs 1 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings True --batch_size 6 --use_discriminator False --n_gpu 2 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 4

sleep 3

python test_avg_impute_subjectwise.py --aug_p 0 --n_epochs 1 --use_past_data True --use_future_data True --using_daft True --use_positional_embeddings True --use_bidirectional_embeddings True --batch_size 6 --use_discriminator False --n_gpu 2 \
         --effective_batch_size 6 --dataset ADNI --target_timepoint all --img_height 128 --debug False --n_cpu 3 --lr 0.00005 --use_val_dataset True \
          --in_channels 3 --num_layers 2 --seq_len 5 --use_deformation_field True --fold 5
          