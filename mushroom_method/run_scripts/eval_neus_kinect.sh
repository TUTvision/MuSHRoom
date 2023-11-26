export CUDA_VISIBLE_DEVICES=1
rooms_name=( "kokko")  #  "activity"
for item in "${rooms_name[@]}";
    do
    ns-eval \
    --load-config="outputs/${item}/neus-facto/baseline_kinect_all/config.yml" \
    --output-path="outputs/${item}/neus-facto/baseline_kinect_all/output.json"
    wait

    done

