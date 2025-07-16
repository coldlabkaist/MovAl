# MovAl (Move Altogether)
![background_image](https://github.com/user-attachments/assets/d3af7702-ae83-4f63-95d7-0907b87eeac7)

**MovAl:** Move Altogether is an integrated pipeline of YOLO (pose) and Cutie (Instance segmentation) for multi animal key point detection. 
This pipeline overcomes the Id-switching problem of existing multi-animal tracking methods and give better key point detection quality. Try it for your multi-instance!

![그림2](https://github.com/user-attachments/assets/c5652ecb-3ee8-402a-8a71-4e1059db3ea8)
![그림3](https://github.com/user-attachments/assets/c4d4b78c-3bfe-4d05-8835-f48ba638381d)

[🔗 See more tracking results!](https://github.com/coldlabkaist/MovAl/tutorial/Tracking_Result)

## Requirement
- CUDA 11.8 or 12.1
- We recommend using Conda for setting up the environment.

### Installation Tutorial
[0. Installation](https://github.com/coldlabkaist/MovAl/tutorial/1_Installation)

### Tutorial on using the MovAl
[1. Create Project](https://github.com/coldlabkaist/MovAl/tutorial/2_Create_Project)

[2. Preprocess](https://github.com/coldlabkaist/MovAl/tutorial/3_Preprocess)

[3. Labelary](https://github.com/coldlabkaist/MovAl/tutorial/4_Labelary)

[4. Pose Estimation](https://github.com/coldlabkaist/MovAl/tutorial/4_Pose_Estimation)


## Liscense
MovAl basically follow the MIT license. We allows free use for academic/research purposes, but not for commercial purposes.
We recommends using Cutie and YOLO as part of the pipeline. The usage rights for each part follow the license of each program. 
- Cutie : https://github.com/hkchengrex/Cutie?tab=MIT-1-ov-file
- YOLO : https://www.ultralytics.com/
