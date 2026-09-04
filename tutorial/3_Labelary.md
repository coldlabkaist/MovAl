# Labelary

**Labelary**—combining “Labeling” and “Library”—is a user‑friendly labeling tool for MovAl pipeline!

<img width="1531" height="770" alt="image" src="https://github.com/user-attachments/assets/fb755435-87f6-4d3d-80c1-7f5d7dc971e9" />

From the main page, click Step 3 Labelary to launch Labelary. In the top panel, select the video, data, and visualization mode to start Labelary.


## How to Use Labelary
<img width="2001" height="932" alt="image" src="https://github.com/user-attachments/assets/dba55e84-bf5d-428b-8476-aa5441c12f15" />

### **To label the video**
1. Right-click (or press **Ctrl+A**) on the video to **add a new animal instance**. Use the same menu when you need to replace or delete an instance, change its instance number, or adjust visibility.
2. **Select an instance, then drag** its keypoints to the correct body-part positions. Move through the frame and correct any misplaced points until the skeleton matches the animal.
3. For occluded body parts, place the keypoint at the estimated position first, then **change its visibility** status(right-click or **Ctrl+V**) so the annotation records that the point is hidden.


### **To save the data**
In the bottom-right corner, use the **Save/Export** options to save or export your labeling results:
- Save CSV: Save the current labels as a CSV file. After saving the CSV once, you can use **Ctrl+S** to quickly save subsequent changes to the same file.
- Export TXT: Export the current labels as YOLO-compatible TXT files for training. This may take longer for large datasets. **Each video uses a single TXT label set**, so make sure any existing TXT labels are backed up if needed before exporting again.
- Export Video: Render and save a copy of the video with the current labels overlaid. This can take some time depending on the video length and resolution.

We recommend saving frequently in CSV format while labeling. For training, you can either use the exported TXT labels or train directly from the most recently saved CSV labels.

### **Additional labeling tips**
- You can choose a skeleton color mode from the menu on the right before or during labeling. Use cutie_light or cutie_dark for Cutie-style instance colors, or white / black when a single-color skeleton is easier to see against the current background.
- You can also navigate through the video using the keyboard:
  - **Left / Right Arrow:** Move backward or forward through the video.
  - **A / D:** Alternative shortcuts for moving backward or forward between frames.
- When importing predicted CSV/TXT data, select **Load Inference Result** to load YOLO outputs into Labelary for review and editing.
- You can also export a rendered video with the predicted skeletons overlaid directly from Labelary.


## Automatic Labeling in Labelary

Labelary supports model-assisted labeling so that you can speed up annotation using an existing YOLO pose model from a similar setup.

<img width="1219" height="695" alt="image" src="https://github.com/user-attachments/assets/16de0f25-7f10-4328-928d-1b54d0a050ab" />


**Load a compatible model**: 
If you already have a trained model that matches the current skeleton configuration, click Browse/Load Model to load it.
Once selected, the model is loaded automatically and becomes available for labeling assistance.

**Use Automatic Labeling while browsing frames**: 
If Automatic Labeling is checked, Labelary automatically predicts labels with the loaded model whenever you move to a new frame.

**Use model-assisted labeling manually when needed**:
If Automatic Labeling is not checked, labels are not added automatically when you change frames.
In this case, you can still use the loaded model through the right-click menu:

- Automatic Label Addition
- Automatic Re-labeling

This is useful when you want more manual control over when model-generated labels are applied.

**Refine labels and run Mini Training**: 
After reviewing and adding more labels, you can run Mini Training directly from Labelary.
This is a lightweight training option provided inside the Labelary UI for quick refinement, so we recommend using the standard MovAl training pipeline.

**Notes**:
Mini Training runs in the background, so you can continue labeling while it is running.
However, if Mini Training and Automatic Labeling are used at the same time, they may compete for GPU memory.
If GPU memory is limited, running both simultaneously may cause the process to fail, so use this workflow with caution.


Now you’re ready for [Step 4: Pose Estimation](https://github.com/coldlabkaist/MovAl/blob/main/tutorial/4_Pose_Estimation.md)
