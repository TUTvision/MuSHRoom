export CUDA_VISIBLE_DEVICES=0
rooms_name=( "activity" "classroom" "coffee_room" "computer" "koivu" "kokko")  #  "activity"
for item in "${rooms_name[@]}";
    do
    ns-eval \
    --load-config="outputs/${item}/neus-facto/baseline_iphone/config.yml" \
    --output-path="outputs/${item}/neus-facto/baseline_iphone/output.json"
    wait

    done

