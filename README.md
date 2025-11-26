# MovAl (Move Altogether)
![background_image](https://github.com/user-attachments/assets/d3af7702-ae83-4f63-95d7-0907b87eeac7)

**MovAl:** Move Altogether is an integrated pipeline of YOLO (pose) and Cutie (Instance segmentation) for multi animal key point detection. 
This pipeline overcomes the Id-switching problem of existing multi-animal tracking methods and give better key point detection quality. Try it for your multi-instance!

![그림2](https://github.com/user-attachments/assets/c5652ecb-3ee8-402a-8a71-4e1059db3ea8)
![그림3](https://github.com/user-attachments/assets/c4d4b78c-3bfe-4d05-8835-f48ba638381d)

[🔗 See more tracking results!](https://github.com/coldlabkaist/MovAl/blob/main/tutorial/Tracking_Result.md)

## Requirement
- CUDA 11.8 (or 12.1, 12.8)
- Supports Windows 11, Ubuntu 22.04 environment.

### Installation Tutorial
[0. Installation](https://github.com/coldlabkaist/MovAl/blob/main/tutorial/0_Installation.md)

### Tutorial on using the MovAl
[1. Create Project](https://github.com/coldlabkaist/MovAl/blob/main/tutorial/1_Create_Project.md)

[2. Preprocess](https://github.com/coldlabkaist/MovAl/blob/main/tutorial/2_Preprocess.md)

[3. Labelary](https://github.com/coldlabkaist/MovAl/blob/main/tutorial/3_Labelary.md)

[4. Pose Estimation](https://github.com/coldlabkaist/MovAl/blob/main/tutorial/4_Pose_Estimation.md)


## News
**25.10.16.** MovAl v1.1.0 released
- Improved Training Stability
- Fixed visibility Bug
- Fixed Labelary key release error
- New Feature: Labeled-Frame Navigation in Labelary
- CSV files from versions 1.0.x may be incompatible.
In such cases, rename the column frame.idx to frame_idx in the CSV file.

**25.09.15.** MovAl v1.0.4 released
- Resolve index shift issue of labelary
- Clean up and standardize config file paths
- Add selective contour generation

**25.08.14.** MovAl v1.0.3 released
- Resolved CPU management issue
- Bug fixed path for txt2csv

**25.08.08.** MovAl v1.0.2 released
- Resolved data split errors
- Enhanced exception handling

**25.08.01.** MovAl v1.0.1 released 
- Improve YOLO inference UI.
- Fix bugs related to video export and skeleton visualization settings.

**25.07.16.** MovAl v1.0.0 released!

## Liscense
MovAl basically follow the MIT license. We allows free use for academic/research purposes, but not for commercial purposes.
We recommends using Cutie and YOLO as part of our pipeline. The usage rights for each part follow the license of each program. 
- Cutie : https://github.com/hkchengrex/Cutie
- YOLO : https://www.ultralytics.com/
