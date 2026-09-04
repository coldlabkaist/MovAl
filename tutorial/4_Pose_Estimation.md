# 4. Pose Estimation

MovAl GUI supports YOLO‑based pose estimation.
From the main page, click Step 4: Pose Estimation to proceed through the YOLO training process step by step.



## Prepare Dataset

<img width="1799" height="834" alt="image" src="https://github.com/user-attachments/assets/ebe55db7-3172-4f56-854c-cf6ff15623c2" />


Before training a YOLO pose model, first prepare the labeled dataset by splitting it into training, validation, and test sets.
Click Prepare Dataset to view the available labeled data for each video and its corresponding TXT labels in the current project.
- Select the video(s) to include, then set the train / validation / test split ratios and choose the visualization mode that matches the label data.
- For most cases, the default split ratio is recommended. If the total amount of labeled data is limited, you may increase the proportion assigned to the training set so that more labeled frames are used for model learning.
- After confirming the settings, click Run to generate the dataset split automatically.

Note: Each project stores only one dataset split. Running Prepare Dataset again will overwrite the existing split.


## Model Training

<img width="2104" height="928" alt="image" src="https://github.com/user-attachments/assets/a76b1a3b-fa41-41f5-910d-62e00a701d06" />

Click Train Model to configure the YOLO pose training settings and start model training.

**1. Choose the training model**

In the Model section, select the YOLO model architecture or checkpoint to use for training. 
If you want to continue training from an existing checkpoint, enable Use Pretrained Model and select the corresponding .pt file.
You can also use one of the pretrained models distributed with MovAl. Download the model that matches your experimental setup and skeleton configuration, then select it from this window.

**Important**: A pretrained model should use the same skeleton definition and node order as the current project. Models trained with a different skeleton configuration may not work correctly.

**2. Configure training hyperparameters**

Adjust the training parameters according to the size of your dataset and available hardware.

Common settings include:

- epochs: Maximum number of training epochs.
- batch: Number of samples processed at once. Setting batch too high may cause an Out-of-Memory (OOM) error. Choose an appropriate value based on your dataset and available GPU memory. If training fails with an OOM error, reduce the batch value.
- patience: Number of epochs to wait for improvement before early stopping. If patience is set, training may stop automatically before reaching the maximum number of epochs when validation performance no longer improves. Set patience = 0 if you want training to continue for the full configured training schedule without early stopping.
- Other YOLO training parameters can be adjusted as needed.

**3. Select the training device**

Under Training Options, specify the device used for training.
For a single GPU, select or enter the desired GPU device. For multi-GPU training, enter the GPU device IDs that should participate in training. Make sure the selected GPUs have sufficient available memory before starting training.

**4. Start training**

Training progress, loss values, and status messages can be monitored in the terminal while the model is running. Also, the progress is displayed on the main page of the hair.

The best-performing checkpoint is typically stored in `runs/train_.../weights/best.pt`

For detailed information, refer to the [YOLO documentation](https://docs.ultralytics.com/ko/modes/train/)

## Inference

<img width="1873" height="929" alt="image" src="https://github.com/user-attachments/assets/f07189a0-d204-4dd7-bbb5-cb8d980e7be3" />

**1. Select a model**

Click Browse Model and select the trained YOLO model (.pt) you want to use.
For the best performance, use a model trained with the same skeleton configuration and visualization mode as the current project.

**2. Select the inference source**

Under Video Selection, choose the input source:

- video: run inference directly on the selected video. Videos that were not included in the project can also be subject to it.
- image frames: run inference on previously generated image frames.

Tip: Use the same visualization mode used during training whenever possible. For example, if the model was trained on contour images, run inference using the corresponding visualization input.

**3. Select inference targets**

Under Inference Target, check the animal IDs or classes you want to detect.

Only the selected targets will be included in the inference results.

**4. Configure inference and output options**
**Inference Config**

You can adjust the main YOLO inference parameters:

-  imgsz: input image size used for inference.
-  conf: minimum confidence threshold for accepting detections.
-  iou: IoU threshold used during detection filtering.
-  augment: enables augmented inference.
-  half: uses FP16 inference when supported.
-  device: specifies the GPU or CPU device to use.

If detection sensitivity is too low, you may reduce the confidence threshold, but this can also increase false-positive detections.

**Visualization / Save Options**

Use the Visualization section to choose what should be displayed or saved:

-  show tracking result: display inference results live during processing.
   This is useful for quick inspection but is not recommended for long videos, as displaying every frame may slow down processing.
-  save image/video: save rendered frames or videos with predicted skeletons overlaid.
-  run image frames as video: process a sequence of image frames as a video-like input.
-  save result as txt: export predictions in TXT format.
-  save result as csv: export prediction results as CSV.

Additional CSV options allow you to configure the output format and coordinate representation.

**5. Run inference**

After checking the model, input videos, targets, and output settings, click Run Inference.

Inference progress and processing messages can be monitored in the terminal.

Generated prediction files are saved in the project's prediction/output directories according to the selected save options.

For detailed information, refer to the [YOLO documentation](https://docs.ultralytics.com/ko/modes/predict/)

---

With these steps, you’ll be able to perform accurate body‑part detection using MovAl.

For additional project management guidelines, please refer to the following document: 
[MovAl Project Structure](https://github.com/coldlabkaist/MovAl/blob/main/tutorial/MovAl_Project_Structure.md)
