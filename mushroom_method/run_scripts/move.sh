rooms_name=("activity" "classroom" "coffee_room" "computer" "honka" "koivu" "kokko" "olohuone" "sauna" "vr_room")  #  "activity" "classroom" "coffee_room" "honka" "koivu" "kokko" "sauna" "vr_room"
for item in "${rooms_name[@]}";
    do
    rsync -rvaz  room_datasets/$item/iphone/  renxuqia@puhti.csc.fi:/scratch/project_2008248/sdfstudio/room_datasets/$item/iphone/
    # scp -r renxuqia@puhti.csc.fi:/scratch/project_2008248/sdfstudio/room_datasets/$item/iphone/long_capture/depth room_datasets/$item/iphone/long_capture 
    # outputs/$item/nerfacto/baseline_kinect renxuqia@puhti.csc.fi:/scratch/project_2008248/sdfstudio/outputs/$item/nerfacto
    # scp -r outputs/$item/nerfacto/baseline_iphone_all/render renxuqia@puhti.csc.fi:/scratch/project_2008248/sdfstudio/outputs/$item/nerfacto/baseline_iphone_all
    
    # rsync -rvaz room_datasets/$item/iphone renxuqia@puhti.csc.fi:/scratch/project_2008248/sdfstudio/room_datasets/$item/
    done