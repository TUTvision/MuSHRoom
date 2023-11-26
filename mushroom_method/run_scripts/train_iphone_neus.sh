export CUDA_VISIBLE_DEVICES=1

rooms_name=("classroom")  #  
for item in "${rooms_name[@]}";
    do
    ns-train neus-facto \
    --vis wandb \
    --viewer.websocket-port 7049 \
    --experiment-name $item \
    --timestamp baseline_iphone_train_two_head_new \
    --trainer.save-only-latest-checkpoint True \
    --trainer.max-num-iterations 60001 \
    --trainer.steps-per-save 30000 \
    --pipeline.model.sdf-field.use-grid-feature True \
    --pipeline.model.sdf-field.num-layers 2 \
    --pipeline.model.sdf-field.hidden-dim 256 \
    --pipeline.model.sdf-field.geo_feat_dim 256 \
    --pipeline.model.sdf-field.num-layers-color 3 \
    --pipeline.model.sdf-field.hidden-dim-color 256 \
    --pipeline.model.sdf-field.use-appearance-embedding True \
    --pipeline.model.sdf-field.geometric-init True \
    --pipeline.model.sdf-field.inside-outside True  \
    --pipeline.model.sdf-field.bias 0.8 \
    --pipeline.model.sdf-field.beta-init 0.7 \
    --pipeline.model.sdf-field.encoding-type hash \
    --pipeline.model.far-plane 10 \
    --pipeline.model.overwrite-near-far-plane True \
    --pipeline.model.file_name ${item} \
    --pipeline.datamanager.train-num-images-to-sample-from 300 \
    --pipeline.datamanager.train_num_times_to_repeat_images 10000 \
    --pipeline.model.background-model grid \
    --pipeline.model.sensor-depth-l1-loss-mult 0.1 \
    --pipeline.model.sensor-depth-truncation 0.015 \
    --pipeline.model.sensor-depth-freespace-loss-mult 10.0 \
    --pipeline.model.sensor-depth-sdf-loss-mult 6000.0 \
    --pipeline.model.sparse_points_sdf_loss_mult 0.0 \
    --pipeline.model.mono-depth-loss-mult  0 \
    --pipeline.model.mono-normal-loss-mult 0 \
    --pipeline.datamanager.train-num-rays-per-batch 4096 \
    --machine.num-gpus 1 \
    sdfstudio-data --data  room_datasets/${item}/iphone/long_capture/sdf_dataset_train  \
    --include_sensor_depth True 
    done

# --pipeline.model.sensor-depth-l1-loss-mult 0.1 \
#     --pipeline.model.sensor-depth-truncation 0.015 \
#     --pipeline.model.sensor-depth-freespace-loss-mult 10.0 \
#     --pipeline.model.sensor-depth-sdf-loss-mult 6000.0 \
#     --pipeline.model.sparse_points_sdf_loss_mult 1.0 \

