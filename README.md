# MuSHRoom: Multi-Sensor Hybrid Room Dataset for Joint 3D Reconstruction and Novel View Synthesis
 [Xuqian Ren](https://xuqianren.github.io/) ,  [Wenjia Wang](https://wenjiawang0312.github.io/) ,  [Dingding Cai](https://dingdingcai.github.io/) , Tuuli Tuominen,  [Juho Kannala](https://users.aalto.fi/~kannalj1/), [Esa Rahtu](https://esa.rahtu.fi/) 

[Project Page](https://xuqianren.github.io/publications/MuSHRoom/) | [Paper](https://arxiv.org/pdf/2311.02778.pdf)  


Metaverse technologies demand accurate, real-time, and immersive modeling on consumer-grade hardware for both non-human perception (e.g., drone/robot/autonomous car navigation) and immersive technologies like AR/VR, requiring both structural accuracy and photorealism. However, there exists a knowledge gap in how to apply geometric reconstruction and photorealism modeling (novel view synthesis) in a unified framework. 

To address this gap and promote the development of robust and immersive modeling and rendering with consumer-grade devices, first, we propose a real-world Multi-Sensor Hybrid Room Dataset (MuSHRoom). Our dataset presents exciting challenges and requires state-of-the-art methods to be cost-effective, robust to noisy data and devices, and can jointly learn 3D reconstruction and novel view synthesis, instead of treating them as separate tasks, making them ideal for real-world applications. Second, we benchmark several famous pipelines on our dataset for joint 3D mesh reconstruction and novel view synthesis. Finally, in order to further improve the overall performance, we propose a new method that achieves a good trade-off between the two tasks. Our dataset and benchmark show great potential in promoting the improvements for fusing 3D reconstruction and high-quality rendering in a robust and computationally efficient end-to-end fashion. The dataset will be made publicly available.

## Updates
* [x] 📣  Updates results with only COLMAP pose [2024-03-14]
* [x] 📣  Dataset process scripts have been released [2023-11-19]
* [x] 📣  Release Kinect and iPhone Dataset. [2023-11-28]
* [x] 📣  Release mesh evaluation script [2023-11-26]
* [x] 📣  Release our method. [2023-11-26]
* [x] 📣  Release mesh. [2023-11-29]

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
The data files are available for download on [Zenodo](https://zenodo.org/communities/mushroom?q=&l=list&p=1&s=10&sort=newest) and can be downloaded on a per dataset basis from there. 


<!-- iPhone data that contains the "images", "depth", "poses" can be downloaded from:
[Part1](https://zenodo.org/records/10154395),
[Part2](https://zenodo.org/records/10155482),
[Part3](https://zenodo.org/records/10155616)


[Mesh](https://zenodo.org/records/10156860) is the reference mesh.

If you want to try our method proposed in the paper, you can also download the generated data from iPhone [Part4](https://zenodo.org/records/10154510) and iPhone [Part5](https://zenodo.org/records/10151161). -->


## Data structure
To maximize compatibility, all data is published in open and simple file formats. The folder structure for one data set looks like the following:

```
<room_name>
| —— kinect
	| —— long_capture
		— images/ # extracted rgb images of keyframe
		— depth/ # extracted depth images of keyframe
		— intrinsic/ # intrinsic parameters
		— PointCloud/ # spectacularAI point cloud of keyframe
		— pose/	# spectacularAI pose of keyframe. These poses are aligned with the metric of depth. Poses are in the OPENCV coordination.
		— calibration.json; data.jsonl; data.mkv; data2.mkv; vio_config.yaml	# raw videos and parameters from spectacularAI SDK
		— camera_parameters.txt	# camera settings during capture
		— test.txt # image id for testing within a single sequence
		— transformations_colmap.json # global optimized colmap used for testing with a different sequence
		— transformations.json	# spectacularAI pose saved in the json file. Poses are in the OPENGL coordination.
	| —— short_capture
		— images/ # same with long capture
		— depth/	# same with long capture
		— PointCloud/	# same with long capture
		— pose/	# same with long capture
		— intrinsic/ # same with long capture
		— calibration.json; data.jsonl; data.mkv; data2.mkv; vio_config.yaml	# raw videos and parameters from 
		— transformations_colmap.json # same with long capture
		— transformations.json	# same with long capture
| —— iphone
	| —— long_capture
		— images/	# same with Kinect
		— depth/	# same with Kinect
		— polycam_mesh/		# mesh provided by polycam, not aligned with the pose, just for visulization.
		— polycam_pointcloud.ply	# point cloud provided by polycam, just for visulization.
		— sdf_dataset_all_interp_4	# same with Kinect
		— sdf_dataset_train_interp_4	# same with Kinect
		— test.txt	# same with Kinect
		— transformations_colmap.json	# same with Kinect
		— transformations.json	# polycam pose
	| —— short_capture
		— images/	# same with Kinect
		— depth/	# same with Kinect
		— transformations_colmap.json	# same with long capture
		— transformations.json	# same with long capture
—— gt_mesh.ply	# reference mesh used for geometry comparison
—— gt_pd.ply	# reference point cloud used for geometry comparison
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


## Novel view synthesis and mesh reconstruction results

We update Nerfacto/Depth-Nerfacto/Neusfacto/Splatfacto trained only with COLMAP pose there. We trained once time to evaluate both the two evaluation protocols there to improve efficiency, instead of training two times which was used in the previous paper before. The test ID used for evaluating the "test within a single sequence" is stored in "test.txt" in each "long_capture" folder, the remaining ID in the long sequence is used for training. We use the same model to evaluate the images in the short sequence. Mesh extracted from this model is used for evaluating the mesh reconstruction ability. Please follow this training and comparsion method reported here for efficiency.


<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.math-mode {
    font-style: normal;
    font-family: 'Computer Modern', 'Latin Modern Math', 'Arial', sans-serif;
}

</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" rowspan="3">Device</th>
    <th class="tg-c3ow" rowspan="3">Methods</th>
    <th class="tg-c3ow" colspan="5" rowspan="2">Reconstruction quality</th>
    <th class="tg-c3ow" colspan="6">Rendering quality</th>
  </tr>
  <tr>
    <th class="tg-c3ow" colspan="3">Test within a single sequence</th>
    <th class="tg-c3ow" colspan="3">Test with a different sequence</th>
  </tr>
  <tr>
    <th class="tg-c3ow"><span class="latex">Acc ↓</span></th>
    <th class="tg-c3ow">Comp ↓</th>
    <th class="tg-c3ow">C-l1 ↓</th>
    <th class="tg-c3ow">NC ↑</th>
    <th class="tg-c3ow">F-score ↑</th>
    <th class="tg-c3ow">PSNR ↑</th>
    <th class="tg-c3ow">SSIM ↑</th>
    <th class="tg-c3ow">LPIPS ↓</th>
    <th class="tg-c3ow">PSNR ↑</th>
    <th class="tg-c3ow">SSIM$ ↑</th>
    <th class="tg-c3ow">LPIPS ↓</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow" rowspan="4">iPhone</td>
    <td class="tg-c3ow"><a href="https://github.com/nerfstudio-project/nerfstudio">Nerfacto</a></td>
    <td class="tg-c3ow">0.0652</td>
    <td class="tg-c3ow">0.0603</td>
    <td class="tg-c3ow">0.0628</td>
    <td class="tg-c3ow">0.7491</td>
    <td class="tg-c3ow">0.6390</td>
    <td class="tg-c3ow">20.83</td>
    <td class="tg-c3ow">0.7653</td>
    <td class="tg-c3ow">0.2506</td>
    <td class="tg-c3ow">20.36</td>
    <td class="tg-c3ow">0.7448</td>
    <td class="tg-c3ow">0.2781</td>
  </tr>
  <tr>
    <td class="tg-c3ow"><a href="https://github.com/nerfstudio-project/nerfstudio">Depth-Nerfacto</a></td>
    <td class="tg-c3ow">0.0653</td>
    <td class="tg-c3ow">0.0614</td>
    <td class="tg-c3ow">0.0634</td>
    <td class="tg-c3ow">0.7354</td>
    <td class="tg-c3ow">0.6126</td>
    <td class="tg-c3ow">21.23</td>
    <td class="tg-c3ow">0.7623</td>
    <td class="tg-c3ow">0.2612</td>
    <td class="tg-c3ow">20.67</td>
    <td class="tg-c3ow">0.7423</td>
    <td class="tg-c3ow">0.2873</td>
  </tr>
  <tr>
    <td class="tg-c3ow"><a href="https://github.com/autonomousvision/sdfstudio">MonoSDF</a></td>
    <td class="tg-c3ow">0.0792</td>
    <td class="tg-c3ow">0.0237</td>
    <td class="tg-c3ow">0.0514</td>
    <td class="tg-c3ow">0.8200</td>
    <td class="tg-c3ow">0.7596</td>
    <td class="tg-c3ow">19.79</td>
    <td class="tg-c3ow">0.6972</td>
    <td class="tg-c3ow">0.4122</td>
    <td class="tg-c3ow">17.92</td>
    <td class="tg-c3ow">0.6683</td>
    <td class="tg-c3ow">0.4384</td>
  </tr>
  <tr>
    <td class="tg-c3ow"><a href="https://docs.nerf.studio/nerfology/methods/splat.html">Splatfacto</a></td>
    <td class="tg-c3ow">0.1074</td>
    <td class="tg-c3ow">0.0708</td>
    <td class="tg-c3ow">0.0881</td>
    <td class="tg-c3ow">0.7602</td>
    <td class="tg-c3ow">0.4405</td>
    <td class="tg-c3ow">24.22</td>
    <td class="tg-c3ow">0.8375</td>
    <td class="tg-c3ow">0.1421</td>
    <td class="tg-c3ow">21.39</td>
    <td class="tg-c3ow">0.7738</td>
    <td class="tg-c3ow">0.1986</td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="4">Kinect</td>
    <td class="tg-c3ow"><a href="https://github.com/nerfstudio-project/nerfstudio">Nerfacto</a></td>
    <td class="tg-c3ow">0.0669</td>
    <td class="tg-c3ow">0.0695</td>
    <td class="tg-c3ow">0.0682</td>
    <td class="tg-c3ow">0.7458</td>
    <td class="tg-c3ow">0.6252</td>
    <td class="tg-c3ow">23.89</td>
    <td class="tg-c3ow">0.8375</td>
    <td class="tg-c3ow">0.2048</td>
    <td class="tg-c3ow">22.43</td>
    <td class="tg-c3ow">0.8331</td>
    <td class="tg-c3ow">0.2010</td>
  </tr>
  <tr>
    <td class="tg-c3ow"><a href="https://github.com/nerfstudio-project/nerfstudio">Depth-Nerfacto</a></td>
    <td class="tg-c3ow">0.0710</td>
    <td class="tg-c3ow">0.0691</td>
    <td class="tg-c3ow">0.0701</td>
    <td class="tg-c3ow">0.7274</td>
    <td class="tg-c3ow">0.5905</td>
    <td class="tg-c3ow">24.21</td>
    <td class="tg-c3ow">0.8370</td>
    <td class="tg-c3ow">0.2107</td>
    <td class="tg-c3ow">22.77</td>
    <td class="tg-c3ow">0.8345</td>
    <td class="tg-c3ow">0.2036</td>
  </tr>
  <tr>
    <td class="tg-c3ow"><a href="https://github.com/autonomousvision/sdfstudio">MonoSDF</a></td>
    <td class="tg-c3ow">0.0439</td>
    <td class="tg-c3ow">0.0204</td>
    <td class="tg-c3ow">0.0321</td>
    <td class="tg-c3ow">0.8616</td>
    <td class="tg-c3ow">0.8753</td>
    <td class="tg-c3ow">23.05</td>
    <td class="tg-c3ow">0.8315</td>
    <td class="tg-c3ow">0.2434</td>
    <td class="tg-c3ow">21.60</td>
    <td class="tg-c3ow">0.8267</td>
    <td class="tg-c3ow">0.2219</td>
  </tr>
  <tr>
    <td class="tg-c3ow"><a href="https://docs.nerf.studio/nerfology/methods/splat.html">Splatfacto</a></td>
    <td class="tg-c3ow">0.1007</td>
    <td class="tg-c3ow">0.0704</td>
    <td class="tg-c3ow">0.0855</td>
    <td class="tg-c3ow">0.7689</td>
    <td class="tg-c3ow">0.4697</td>
    <td class="tg-c3ow">26.07</td>
    <td class="tg-c3ow">0.8844</td>
    <td class="tg-c3ow">0.1378</td>
    <td class="tg-c3ow">23.28</td>
    <td class="tg-c3ow">0.8604</td>
    <td class="tg-c3ow">0.1579</td>
  </tr>
</tbody>
</table>


