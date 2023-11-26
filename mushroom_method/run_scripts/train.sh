item="kettle1_1"
ns-train instant-ngp \
--vis wandb \
--viewer.websocket-port 7029 \
--experiment-name $item \
--timestamp instant-ngp \
--pipeline.model.predict-normals True \
--trainer.save-only-latest-checkpoint True \
--pipeline.model.file_name $item \
--trainer.max-num-iterations 40001 \
--trainer.steps-per-save 10000 \
--pipeline.datamanager.train-num-images-to-sample-from -1 \
--pipeline.datamanager.train-num-rays-per-batch 4096 \
--machine.num-gpus 1 \
nerfstudio-data --data "hard_case/30_10/kettle1_1" \

 