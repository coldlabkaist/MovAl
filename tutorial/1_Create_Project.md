# 1. Create Project

## Overview
<img width="1385" height="990" alt="image" src="https://github.com/user-attachments/assets/bdb67d12-a34d-47b6-8d17-c350b94a5f07" />

<br>

## Create Project GUI

Once you’ve installed the program and its dependencies as described in the tutorial, 
running it opens the GUI shown below. 

<img width="1008" height="487" alt="image" src="https://github.com/user-attachments/assets/3456b642-177e-4371-ae08-a859c777f6f5" />

MovAl lets you track multiple animals through its step‑by‑step interface.

To create your first project, click **Step 1: Create Project.**

<img width="944" height="488" alt="image" src="https://github.com/user-attachments/assets/9a44dbce-ef0e-4bae-9fbe-f72ef59fe5c9" />


Enter each field:

**1. Set project info**

- **Project name** : Set name for the new MovAl project. (**important** : Directory names and file names must be in English and contain no spaces.)

- **Animals**: Define the number of visually distinguishable animal types, their ID labels (default: track_N), and the Maximum Detections per ID.

  - If animals have distinct visual features (e.g., different color segmentation), assign them as separate IDs by increasing the number of animals. If multiple animals share the same visual features and do not need individual identity tracking, keep them under the same ID and increase Maximum Detections per ID instead.

**2. Add Videos** : Add source video file to be analysed.

3. Add CSV/TXT Files (Optional) : If pre-defined label datasets are available for the videos, add them and make sure each label file is correctly matched to its corresponding video.

**4. Choose / Create skeleton** : Set key‑point and skeleton template for pose tracking. 
- you can create your own preset through **Skeleton Setting** button


After completing 1-4, click **Create Project** to generate the project.

<br>

## How to create New Skeleton Config

To create new skeleton preset, click **Skeleton Setting** button and open Skeleton Manager GUI. This GUI lets you create or customize any skeleton preset you need.

<img width="1790" height="598" alt="image" src="https://github.com/user-attachments/assets/0c830a3a-08b2-42a3-9fac-4b6255c382a9" />


In **Add keypoint** mode, click anywhere to freely add new keypoints.
Arrange nodes as you like, then set each node’s name and visual settings. Changing the node order affects YOLO-based pose model training.

In **Add skeleton / symmetry** mode, 
Drag between nodes to define skeleton links (left‑click drag, black solid line) or to specify symmetry (right‑click drag, cyan dashed line).
All keypoints and skeleton links can be selected and deleted. Symmetry information is required for YOLO learning.

<br>
## How to manage project
The **Project Manager** allows you to modify an existing MovAl project after it has been created. You can update project options, add or remove videos, manage CSV/TXT labels, edit the project skeleton, and remove large intermediate files generated during processing.

<img width="878" height="501" alt="image" src="https://github.com/user-attachments/assets/afc7f567-caec-4a9b-a31d-0f39a1e81193" />


- Update project options

   - Project-level settings, such as the maximum number of detections allowed for each ID, can be updated here.

- Manage videos and labels

  - Videos can be added, removed, relinked to their original source, or copied into the project directory. Existing CSV/TXT labels can also be added or removed for each video.

- Edit project skeleton

  - The project skeleton can be modified using Edit Project Skeleton.

  - **Caution**: Modifying the skeleton after project creation may make existing data or trained models incompatible with the project. Incorrect changes may cause errors or prevent YOLO-based training and inference from working properly. Edit the skeleton only when necessary and make sure that its node definitions and order remain compatible with the models and datasets being used.

- Compress project directory

  - The Cutie-based pipeline can generate large intermediate files. The Compress Project options allow these files to be removed to reduce the project directory size. Removing these intermediate files does not prevent the project from being opened or used again. However, deleted intermediate results may need to be regenerated during later processing, which can substantially increase execution time.
  - **Before deleting files from runs/,** make sure that the final trained model or checkpoint you want to keep has been backed up.
  - **Before deleting predicts/,** make sure that any prediction results or output files you may need later have been backed up.

<br>

## (Optional) How to use pretrained model

Each model is compatible with the skeleton configuration used during training.
[This link](https://github.com/coldlabkaist/MovAl/blob/main/tutorial/How_to_use_pretrained_tracking_model.md) provides instructions for deploying pretrained models. If you wish to use a model from the distribution, download the appropriate config file from the link and save it in the specified location (MovAl/preset/skeleton). You can then create a project by specifying the file.
If you pasted the file while the "Create Project" window was open, the file may not be reflected in the list of available config files. In this case, reopen the "Create Project" window to select the file normally.

---

<br>
After saving, your project will appear on the main page. To edit it later, load the project.json from the project directory and resume work. 

Now you’re ready for [Step 2: Preprocess](https://github.com/coldlabkaist/MovAl/blob/main/tutorial/2_Preprocess.md).
  
