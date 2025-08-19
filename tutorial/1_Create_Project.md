# 1. Create Project

## Create Project GUI

Once you’ve installed the program and its dependencies as described in the tutorial, 
running it opens the GUI shown below. 

<img width="2124" height="1125" alt="create_project_1" src="https://github.com/user-attachments/assets/5b588156-fd8b-435f-b7d2-eecd6b01fbdc" />
<br><br>

MovAl lets you track multiple animals through its step‑by‑step interface.

To create your first project, click **Step 1: Create Project.**

<img width="2135" height="1117" alt="create_project_2" src="https://github.com/user-attachments/assets/22ccc4a9-229e-4fd1-9dad-126908479924" />  
<br><br>

Enter each field:

**Project name (A)** : name for the new MovAl project. (**important** : Directory names must be in English and contain no spaces.)

**Animals (B)** : number of animals per video and their labels (default name: track_N).

**Raw video (C)** : source video file to be analysed.

**Skeleton preset (D)** : key‑point and skeleton template for pose tracking. (you can create your own preset through **Skeleton Setting** button)

After completing A–D, click Create Project (E) to generate the project.

<br>
You can delete videos or change order of videos through right click menu and drag.
<br><br>
<img width="2169" height="1119" alt="create_project_3" src="https://github.com/user-attachments/assets/1fceed6f-6228-49db-9e40-08e8bac3472a" />

<br><br>

## How to create New Skeleton Config

To create new skeleton preset, click **Skeleton Setting** button and open Skeleton Manager GUI. This GUI lets you create or customize any skeleton preset you need.

<img width="2590" height="874" alt="create_project_4" src="https://github.com/user-attachments/assets/fde691df-2505-4bd3-a6ae-1ee57389913f" />

<br><br>
In **Add keypoint** mode, click anywhere to freely add new keypoints.
Arrange nodes as you like, then set each node’s name and visual settings.

<img width="446" height="446" alt="create_project_5" src="https://github.com/user-attachments/assets/ac247b57-3690-498d-be98-0a5011d3d1eb" />
<img width="2588" height="872" alt="create_project_6" src="https://github.com/user-attachments/assets/79ae8a08-7347-495d-8314-fe4ae33382e6" />

<br><br>

In **Add skeleton / symmetry** mode, 
Drag between nodes to define skeleton links (left‑click drag, black solid line) or to specify symmetry (right‑click drag, cyan dashed line).
All keypoints and skeleton links can be selected and deleted. **Symmetry information** is required for **YOLO learning**.

<img width="649" height="444" alt="create_project_7" src="https://github.com/user-attachments/assets/9c43c5f6-6d8a-4307-9956-9103bb6b9e7a" />


<br><br>
## (Optional) How to include existing labeled data in project
If you’ve created labels with another tool, import them during the Create Project step. 
MovAl supports CSV/TXT files—use the “Additional Tools” on the main page to convert your DLC/SLP data.

<img width="2141" height="1118" alt="create_project_8" src="https://github.com/user-attachments/assets/ace01071-e619-440a-8f98-f39e6738837b" />

<br><br>
In the Create Project tab, add your TXT and CSV label files, 
then use Auto Sort or drag‑and‑drop to place each label beneath its matching video (see panels B and C). 

Once everything is aligned, click Create Project to save.

---

<img width="2145" height="1125" alt="create_project_9" src="https://github.com/user-attachments/assets/647c9b9f-8cf2-4c3a-939e-3e88a935a614" />

<br><br>
After saving, your project will appear on the main page. To edit it later, load the config.yaml from the project directory and resume work. 

Now you’re ready for [Step 2: Preprocess](https://github.com/coldlabkaist/MovAl/blob/main/tutorial/2_Preprocess.md).
  
