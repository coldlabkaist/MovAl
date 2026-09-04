# MovAl (Move Altogether)
<img width="852" height="769" alt="image" src="https://github.com/user-attachments/assets/adc432b5-40ac-4cb0-a86b-d844879c5b45" />

<br>
<br>

## **Welcome to MovAl!** 
Move Altogether is an integrated pipeline of YOLO (pose) and Cutie (Instance segmentation) for multi animal key point detection. 
This pipeline overcomes the Id-switching problem of existing multi-animal tracking methods and give better key point detection quality. Try it for your multi-instance!

<br>

https://github.com/user-attachments/assets/1bc6e167-f11e-4508-b8ee-5eac7ec70539

[🔗 See more tracking results!](https://github.com/coldlabkaist/MovAl/blob/main/tutorial/Tracking_Result.md)

## Requirement
- CUDA (11.8, 12.1, 12.8)
- Supports Windows 11, Ubuntu 22.04 environment.

### Installation Tutorial
[0. Installation](https://github.com/coldlabkaist/MovAl/blob/main/tutorial/0_Installation.md)

### Tutorial on using the MovAl
[1. Create Project](https://github.com/coldlabkaist/MovAl/blob/main/tutorial/1_Create_Project.md)

[2. Preprocess](https://github.com/coldlabkaist/MovAl/blob/main/tutorial/2_Preprocess.md)

[3. Labelary](https://github.com/coldlabkaist/MovAl/blob/main/tutorial/3_Labelary.md)

[4. Pose Estimation](https://github.com/coldlabkaist/MovAl/blob/main/tutorial/4_Pose_Estimation.md)


## News
[Update Notes!](https://github.com/coldlabkaist/MovAl/blob/main/CHANGELOG.md )

If your local MovAl version is outdated compared to the latest release version on github, you can see a message recommending an update when running the code. You can simply update it using the following command
```
cd (MovAl_folder)
python update_moval.py
# or
python update_moval.py v1.1.1
```
If no version is provided, MovAl updates to the latest release tag. If a version is provided, MovAl checks out that tag.

## Liscense
MovAl basically follow the MIT license. We allows free use for academic/research purposes, but not for commercial purposes.
We recommends using Cutie and YOLO as part of our pipeline. The usage rights for each part follow the license of each program. 
- Cutie : https://github.com/hkchengrex/Cutie
- YOLO : https://www.ultralytics.com/
