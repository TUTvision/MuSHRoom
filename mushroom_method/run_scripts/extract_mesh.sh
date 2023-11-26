export CUDA_VISIBLE_DEVICES=2
rooms_name=("vr_room")  #  "coffee_room" "activity"  "honka" "koivu" "kokko" "sauna" "vr_room" "olohuone"  "computer"  
#
for item in "${rooms_name[@]}";
    do
    ns-export poisson --load-config outputs/${item}/nerfacto/baseline_iphone_all_new/config.yml \
    --output-dir room_datasets/meshes/iphone/nerfstudio/${item}

    # ns-export poisson --load-config outputs/${item}/nerfacto/baseline_kinect_all/config.yml \
    # --output-dir room_datasets/meshes/kinect/nerfstudio/${item}

    done

# ns-export poisson --load-config outputs/${item}/nerfacto/baseline_iphone_all/config.yml \
#     --output-dir room_datasets/meshes/iphone/nerfstudio/${item}
# ns-extract-mesh --load-config outputs/${item}/nerfacto/baseline_iphone_all/config.yml \
    # --output-path room_datasets/meshes/iphone/sdfstudio/${item}.ply

# export CUDA_VISIBLE_DEVICES=1
# ns-export poisson --load-config outputs/koivu/nerfacto/baseline_kinect_all/config.yml \
# --output-dir meshes/koivu_nerfstudio.ply