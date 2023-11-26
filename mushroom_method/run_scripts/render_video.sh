export CUDA_VISIBLE_DEVICES=1
ns-render --load-config outputs/honka/nerfacto/baseline_iphone_all/config.yml \
--output-path outputs/honka/nerfacto/baseline_iphone_all/render.mp4 \
--traj filename \
--seconds 20.0 \
--camera-path-filename room_datasets/camera_path-5.json