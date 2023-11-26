export CUDA_VISIBLE_DEVICES=0

rooms_name=("coffee_room")  #   
for item in "${rooms_name[@]}";
    do
    ns-train depth-nerfacto \
    --vis wandb \
    --viewer.websocket-port 7029 \
    --experiment-name $item \
    --timestamp baseline_kinect_depth_train \
    --pipeline.model.predict-normals False \
    --trainer.save-only-latest-checkpoint True \
    --trainer.max-num-iterations 40001 \
    --trainer.steps-per-save 10000 \
    --pipeline.datamanager.train-num-images-to-sample-from -1 \
    --pipeline.datamanager.train-num-rays-per-batch 4096 \
    --machine.num-gpus 1 \
    nerfstudio-data --data "room_datasets/${item}/kinect/long_capture" \

    wait

    done