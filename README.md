# MuSHRoom: Multi-Sensor Hybrid Room Dataset for Joint 3D Reconstruction and Novel View Synthesis
 [Xuqian Ren](https://xuqianren.github.io/) ,  [Wenjia Wang](https://wenjiawang0312.github.io/) ,  [Dingding Cai](https://dingdingcai.github.io/) , Tuuli Tuominen,  [Juho Kannala](https://users.aalto.fi/~kannalj1/), [Esa Rahtu](https://esa.rahtu.fi/) 

[Project Page](https://xuqianren.github.io/publications/MuSHRoom/) | [Paper](https://arxiv.org/pdf/2311.02778.pdf)  


Metaverse technologies demand accurate, real-time, and immersive modeling on consumer-grade hardware for both non-human perception (e.g., drone/robot/autonomous car navigation) and immersive technologies like AR/VR, requiring both structural accuracy and photorealism. However, there exists a knowledge gap in how to apply geometric reconstruction and photorealism modeling (novel view synthesis) in a unified framework. 

To address this gap and promote the development of robust and immersive modeling and rendering with consumer-grade devices, first, we propose a real-world Multi-Sensor Hybrid Room Dataset (MuSHRoom). Our dataset presents exciting challenges and requires state-of-the-art methods to be cost-effective, robust to noisy data and devices, and can jointly learn 3D reconstruction and novel view synthesis, instead of treating them as separate tasks, making them ideal for real-world applications. Second, we benchmark several famous pipelines on our dataset for joint 3D mesh reconstruction and novel view synthesis. Finally, in order to further improve the overall performance, we propose a new method that achieves a good trade-off between the two tasks. Our dataset and benchmark show great potential in promoting the improvements for fusing 3D reconstruction and high-quality rendering in a robust and computationally efficient end-to-end fashion. The dataset will be made publicly available.

## Updates
* [x] 📣  iPhone Dataset and process scripts have been released [2023-11-19]
* [ ]     Release Kinect Dataset.
* [x] 📣  Release mesh evaluation script [2023-11-26]
* [ ]     Release our method. 

## Attribution
If you use this data, please cite the original paper presenting it:

```
@misc{ren2023mushroom,
      title={MuSHRoom: Multi-Sensor Hybrid Room Dataset for Joint 3D Reconstruction and Novel View Synthesis}, 
      author={Xuqian Ren and Wenjia Wang and Dingding Cai and Tuuli Tuominen and Juho Kannala and Esa Rahtu},
      year={2023},
      eprint={2311.02778},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



## Downloading the data
The data files are available for download on Zenodo and can be downloaded on a per dataset basis from there. 

iPhone data that contains the "images", "depth", "poses" can be downloaded from:
[Part1](https://zenodo.org/records/10154395),
[Part2](https://zenodo.org/records/10155482),
[Part3](https://zenodo.org/records/10155616)


[Mesh](https://zenodo.org/records/10156860) is the reference mesh.

If you want to try our method proposed in the paper, you can also download the generated data from iPhone [Part4](https://zenodo.org/records/10154510) and iPhone [Part5](https://zenodo.org/records/10151161).


## Data structure
To maximize compatibility, all data is published in open and simple file formats. The folder structure for one data set looks like the following:

```
<room_name>
| —— kinect
	| —— long_capture
		— images/ # extracted rgb images of keyframe
		— depth/ # extracted depth images of keyframe
  		— depth_complte_all/ # completed depth used for testing with a different sequence, depth is completed by point cloud reconstructed from all frames.
		— depth_complte_train/ # completed depth used for testing within a single sequence, depth is completed by point cloud reconstructed from training frames.
		— intrinsic/ # intrinsic parameters
		— PointCloud/ # spectacularAI point cloud of keyframe
		— pose/	# spectacularAI pose of keyframe. These poses are aligned with the metric of depth. Poses are in the OPENCV coordination.
		— sdf_dataset_all_interp_3/ # sdfstudio format dataset used for our method
		— sdf_dataset_train_interp_3/ # sdfstudio format dataset used for our method
		— calibration.json; data.jsonl; data.mkv; data2.mkv; vio_config.yaml	# raw videos and parameters from spectacularAI SDK
		— camera_parameters.txt	# camera settings during capture
		— test.txt # image id for testing within a single sequence
		— transformations_colmap.json # global optimized colmap used for testing with a different sequence
		— transformations.json	# pose saved in the json file. Poses are in the OPENGL coordination.
	| —— short_capture
		— images/ # same with long capture
		— depth/	# same with long capture
		— PointCloud/	# same with long capture
		— pose/	# same with long capture
		— intrinsic/ # same with long capture
		— calibration.json; data.jsonl; data.mkv; data2.mkv; vio_config.yaml	# raw videos and parameters from 
		— meta_data_align.json	# aligned test pose used for testing with a different sequence
		— transformations_colmap.json # same with long capture
		— transformations.json	# same with long capture
| —— iphone
	| —— long_capture
		— images/	# same with Kinect
		— depth/	# same with Kinect
		— polycam_mesh/		# mesh provided by polycam, not aligned with the pose, just for visulization.
		— polycam_pointcloud.ply	# point cloud provided by polycam, just for visulization.
		— sdf_dataset_all/	# sdfstudio format dataset used for testing with a different sequence
		— sdf_dataset_train/	# sdfstudio format dataset used for testing within a single sequence
		— sdf_dataset_all_interp_4	# same with Kinect
		— sdf_dataset_train_interp_4	# same with Kinect
		— test.txt	# same with Kinect
		— transformations_colmap.json	# same with Kinect
		— transformations.json	# same with Kinect
	| —— short_capture
		— images/	# same with Kinect
		— depth/	# same with Kinect
		— meta_data_align.json	# same with Kinect
		— transformations_colmap.json	# same with Kinect
		— transformations.json	# same with Kinect
—— gt_mesh.ply	# reference mesh used for geometry comparison
—— icp_iphone.json	# aligned transformation matrix used for iPhone sequences
—— icp_kinect.json	# aligned transformation matrix used for kinect sequences
			
```

## List of data sets
| Scene         | Scale (m)                     |  Exposure time (µs) |  White Balance (K)  |  Brightness |  Gain
|---------------|-------------------------------|-------------------------------------|-------------------------------------|-------------------------------------|-------------------------------------|
| coffee room   | 6.3 $\times$ 5 $\times$ 3.1    | 41700    | 2830                                                          |  128         |  130
| computer room | 9.6 $\times$ 6.1 $\times$ 2.5 | 33330    | 3100                                                          |  128         |  255
| classroom     | 8.9 $\times$ 7.2 $\times$ 2.8 | 33330            | 3300                                                         | 128        | 88  
| honka         | 6.1 $\times$ 3.9 $\times$ 2.3 | 16670              |3200                                                         |  128        | 128 
| koivu         | 10 $\times$ 8 $\times$ 2.5    | 16670                |4200                                                         | 128        |  128
| vr room       | 5.1 $\times$ 4.4 $\times$ 2.8 | 8300              |3300                                                          | 128         | 88
| kokko         | 6.7 $\times$ 6.0 $\times$ 2.5 | 133330            |3300                                                         | Auto       | Auto 
| sauna         | 9.9 $\times$ 6.5 $\times$ 2.4 | Auto                |3300                                                          |  Auto        |  Auto
| activity      | 12 $\times$ 9 $\times$ 2.5    | 50000                |3200                                                          | 128        | 130
| olohuone      | 19 $\times$ 6.4 $\times$ 3    | Auto             |3600                                                          | Auto        | Auto


## List of scripts for processing the data sets

Enviroment installation.
Install SpectacularAI by 
```
pip install spectacularai
```
Install radiance mapping by instruction in [RadianceMapping](https://github.com/seanywang0408/RadianceMapping).
Install [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio).

### kinect_tools

After capturing videos with Kinect, we first extract raw images, depth, pose, and point cloud of each keyframe.
```
python kinect_tools/1_process_rawvideo_perframe.py --input room_datasets/${room_name}/kinect/long_capture --output room_datasets/${room_name}/kinect/long_capture
```

Saving pose of each keyframe in the "pose" folder to a json file.
```
python kinect_tools/2_generate_train_transformations.py --input room_datasets/${room_name}/kinect/long_capture
```

We fuse the point cloud of each keyframe of the whole sequences or of the training frames.
```
python kinect_tools/3_fuse_point_cloud_train.py --input room_datasets/${room_name}/kinect/long_capture --pose_scale all/train
```

We render z-buffer depth from point clouds constructed with all frames or training frames.
```
python kinect_tools/4_generate_zbuf_depth.py —datadir room_datasets/${room_name}/kinect/long_capture —pose_scale all/train
```

We complete raw depth with z-buffer depth. Depth completed from “pointcloud_train.ply” will be used for testing within a single sequence, depth completed from “pointcloud_all.ply” will be used for testing with a different sequence.
```
python kinect_tools/5_complete_depth.py —datadir room_datasets/${room_name}/kinect/long_capture —pose_scale all/train
```

Data used for testing [NeuS-facto](https://github.com/autonomousvision/sdfstudio) with testing within a single sequence protocol can be generated from
```
python kinect_tools/6_generate_sdf_within.py —input_path room_datasets/${room_name}/kinect/long_capture —output_path room_datasets/${room_name}/kinect/long_capture/sdf_dataset_train —type sensor_depth
```

Data used for testing NeuS-facto with testing with a different sequence protocol can be generated from
```
python kinect_tools/8_generate_sdf_different.py —input_path room_datasets/${room_name}/kinect/long_capture —output_path room_datasets/${room_name}/kinect/long_capture/sdf_dataset_all —type sensor_depth 
```

The aligned poses for short capture can be generated from:
```
python kinect_tools/9_generate_sdf_short_different.py —input_path room_datasets/${room_name}/kinect/
```


Since poses used for testing with a different sequence protocol are optimized by COLMAP, the whole point cloud can be transferred to new coordination by:
```
python kinect_tools/3_fuse_point_cloud_all.py —input room_datasets/${room_name}/kinect/long_capture 
```

### iphone_tools

Data used for testing NeuS-facto with testing within a single sequence protocol can be generated from
```
python iphone_tools/1_nerfstudio_to_sdfstudio_within.py —data room_datasets/${room_name}/iphone/long_capture —output-dir room_datasets/${room_name}/iphone/long_capture/sdf_dataset_train —data-type polycam —scene-type indoor —sensor-depth 
```

Data used for testing NeuS-facto with testing with a different sequence protocol can be generated from
```
python iphone_tools/2_nerfstudio_to_sdfstudio_different.py —data room_datasets/${room_name}/iphone/long_capture —output-dir room_datasets/${room_name}/iphone/long_capture/sdf_dataset_all —data-type colmap —scene-type indoor —sensor-depth 
```
(The “koivu” and “activity” rooms are exception. The —data-type needs to be set to “polycam”)

The aligned poses for short capture can be generated from:
```
python iphone_tools/3_nerfstudio_to_sdfstudio_short_different.py —input_dir room_datasets/${room_name}/iphone/ —scene-type indoor 
```






