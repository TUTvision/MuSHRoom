export CUDA_VISIBLE_DEVICES=2

rooms_name=("activity" "classroom" "coffee_room" "computer" "honka" "koivu" "kokko" "olohuone" "sauna" "vr_room")  #  
for item in "${rooms_name[@]}";
    do
    ns-train nerfacto \
    --vis wandb \
    --viewer.websocket-port 7029 \
    --experiment-name $item \
    --timestamp baseline_iphone_all \
    --trainer.save-only-latest-checkpoint True \
    --pipeline.model.predict-normals True \
    --trainer.save-only-latest-checkpoint True \
    --trainer.max-num-iterations 40001 \
    --pipeline.model.file_name ${item} \
    --pipeline.datamanager.train-num-images-to-sample-from -1 \
    --pipeline.datamanager.train-num-rays-per-batch 4096 \
    --machine.num-gpus 1 \
    vrnerfstudio-data --data "room_datasets/${item}/iphone/long_capture" \

    wait

    done