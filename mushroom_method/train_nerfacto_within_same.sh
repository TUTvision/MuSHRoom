rooms_name=("activity"  )  #  other room name 
device_type="kinect" # iphone
for item in "${rooms_name[@]}";
    do
    ns-train nerfacto \
    --vis wandb \
    --experiment-name $item \
    --timestamp test_within_same \
    --pipeline.model.predict-normals True \
    --trainer.save-only-latest-checkpoint True \
    --pipeline.model.file_name $item \
    --trainer.max-num-iterations 40001 \
    --trainer.steps-per-save 1000 \
    --pipeline.datamanager.train-num-images-to-sample-from -1 \
    --pipeline.datamanager.train-num-rays-per-batch 4096 \
    --machine.num-gpus 1 \
    nerfstudio-data --data "room_datasets/${item}/${device_type}/long_capture" \

    wait

    done