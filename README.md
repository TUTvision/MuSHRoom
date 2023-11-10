# MuSHRoom: Multi-Sensor Hybrid Room Dataset for Joint 3D Reconstruction and Novel View Synthesis
 [Xuqian Ren](https://xuqianren.github.io/), [Wenjia Wang](https://wenjiawang0312.github.io/), [Dingding Cai](https://dingdingcai.github.io/), Tuuli Tuominen, [Juho Kannala](https://users.aalto.fi/~kannalj1/), [Esa Rahtu](https://esa.rahtu.fi/) 

[Project Page](https://xuqianren.github.io/publications/MuSHRoom/)  [Paper](https://arxiv.org/pdf/2311.02778.pdf)  

Metaverse technologies demand accurate, real-time, and immersive modeling on consumer-grade hardware for both non-human perception (e.g., drone/robot/autonomous car navigation) and immersive technologies like AR/VR, requiring both structural accuracy and photorealism. However, there exists a knowledge gap in how to apply geometric reconstruction and photorealism modeling (novel view synthesis) in a unified framework. 

To address this gap and promote the development of robust and immersive modeling and rendering with consumer-grade devices, first, we propose a real-world Multi-Sensor Hybrid Room Dataset (MuSHRoom). Our dataset presents exciting challenges and requires state-of-the-art methods to be cost-effective, robust to noisy data and devices, and can jointly learn 3D reconstruction and novel view synthesis, instead of treating them as separate tasks, making them ideal for real-world applications. Second, we benchmark several famous pipelines on our dataset for joint 3D mesh reconstruction and novel view synthesis. Finally, in order to further improve the overall performance, we propose a new method that achieves a good trade-off between the two tasks. Our dataset and benchmark show great potential in promoting the improvements for fusing 3D reconstruction and high-quality rendering in a robust and computationally efficient end-to-end fashion. The dataset will be made publicly available.

## Attribution
If you use this data, please cite the original paper presenting it:




## Downloading the data
<!-- The data files are available for download on Zenodo:  [https://zenodo.org/record/1320824](https://zenodo.org/record/1320824)  and can be downloaded on a per dataset basis from there. You can also use wget with the following bash snippet to fetch all the data:
```
# Download all 20 data ZIPs from Zenodo
for i in $(seq -f “%02g” 1 23);
do
  wget -O advio-$i.zip https://zenodo.org/record/1476931/files/advio-$i.zip
done
``` -->

## Data structure
To maximize compatibility, all data is published in open and simple file formats. The folder structure for one data set looks like the following:

```
<room_name>
| —— kinect
	| —— long_capture
		— images/ # extracted rgb images of keyframe
		— depth/ # extracted depth images of keyframe
  		— depth_complte_all/ # completed depth used for testing with a different sequence
		— depth_complte_train/ # completed depth used for testing within a single sequence
		— intrinsic/ # intrinsic parameters
		— PointCloud/ # spectacularAI point cloud of keyframe
		— pose/	# spectacularAI pose of keyframe
		— sdf_dataset_all/ # sdfstudio format dataset used for testing with a different sequence
		— sdf_dataset_train/ # sdfstudio format dataset used for testing within a single sequence
		— sdf_dataset_all_interp_3/ # sdfstudio format dataset used for our method
		— sdf_dataset_train_interp_3/ # sdfstudio format dataset used for our method
		— calibration.json; data.jsonl; data.mkv; data2.mkv; vio_config.yaml	# raw videos and parameters from spectacularAI SDK
		— camera_parameter.txt	# camera settings during capture
		— test.txt # image id for testing within a single sequence
		— transformations_colmap.json # global optimized colmap used for testing with a different sequence
		— transformations_train.json	#	pose used for testing within a single sequence
		— transformations.json	# raw pose 
		— transformations_interp_all_3.json	# interplated pose sed for testing with a different sequence
		— transformations_interp_train_3.json	# interplated pose sed for testing within a single sequence
	| —— short_capture
		— images/ # same with long capture
		— depth/	# same with long capture
		— PointCloud/	# same with long capture
		— pose/	# same with long capture
		— calibration.json; data.jsonl; data.mkv; data2.mkv; vio_config.yaml	# raw videos and parameters from 
		— meta_data_align.json	# aligned test pose used for testing with a different sequence
		— transformations_colmap.json # same with long capture
		— transformations.json	# raw pose 
| —— iphone
	| —— long_capture
		— images/	# same with Kinect
		— depth/	# same with Kinect
		— polycam_mesh/		# mesh provided by polycam
		— sdf_dataset_all/	# same with Kinect
		— sdf_dataset_train/	# same with Kinect
		— sdf_dataset_all_interp_4	# same with Kinect
		— sdf_dataset_train_interp_4	# same with Kinect
		— polycam_pointcloud.ply	# point cloud provided by polycam
		— mesh_info.json	# transformation matrix used for polycam mesh
		— test.txt	# same with Kinect
		— transformations_colmap.json	# same with Kinect
		— transformations_interp_all_4.json	# same with Kinect
		— transformations_interp_train_4.json	# same with Kinect
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

