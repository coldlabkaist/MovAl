# 4. Pose Estimation

MovAl GUI supports YOLO‑based pose estimation.
From the main page, click Step 4: Pose Estimation to proceed through the YOLO training process step by step.

<img width="1623" height="938" alt="step4_1" src="https://github.com/user-attachments/assets/ee79cd58-1c54-4400-863a-6d1648bbce27" />

<br>

## Prepare Dataset

<img width="1649" height="705" alt="step4_3" src="https://github.com/user-attachments/assets/75fca8bb-4592-459b-aef1-276088d769bb" />
<br><br>

To begin YOLO training, first split your data. Click Prepare Dataset to see the number of labeled frames for each video and TXT file in your project.

<br>
<img width="1649" height="705" alt="step4_3" src="https://github.com/user-attachments/assets/984f02fd-0a71-48f8-8c59-c39b8e3bad86" />
<br><br>

Select a video from the list, set your train/validation/test split ratios, and choose a visualization method.

Click Run to automatically split the data.

Note: Each project stores only one split—running this will overwrite any existing split.

## Model Training

<img width="1548" height="804" alt="step4_4" src="https://github.com/user-attachments/assets/b2eef5fb-eeb4-45d3-8186-bb34d9fe1682" />
<br><br>

Click Train Model to configure YOLO training options and begin training.

- In the Model section, enable Use Pretrained Model to resume training on new data from an existing checkpoint.

- For multi‑GPU training, specify the device IDs under Training Options.

View training progress in the terminal. All training logs and model checkpoints are saved in your project’s runs folder.

For detailed information, refer to the [YOLO documentation](https://docs.ultralytics.com/ko/modes/train/)

## Inference

<img width="1588" height="688" alt="step4_5" src="https://github.com/user-attachments/assets/39d373eb-c2b2-46ce-9868-0512b732031f" />
<br><br>

Click Pose Estimation to run inference with your trained YOLO model.

- In the Video Selection tab, pick the video and set the Visualization Mode.
Tip: Use the same mode you trained on to maximize accuracy.

- Under Inference Target, check only the objects you want to track.

- In the Visualization tab:

  - Select Show to display results live (not recommended for long runs).

  - Select Save to export results as images or a video (saved to your project’s predicts folder).

Watch inference progress in your terminal. To correct any inference errors or to acquire additional data via inference, 
reload the results in Labelary for editing.

For detailed information, refer to the [YOLO documentation](https://docs.ultralytics.com/ko/modes/predict/)

---

With these steps, you’ll be able to perform accurate body‑part detection using MovAl.

For additional project management guidelines, please refer to the following document: 
[MovAl Project Structure](https://github.com/coldlabkaist/MovAl/blob/main/tutorial/MovAl_Project_Structure.md)
