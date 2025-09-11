# How to use pretrained model

For the user's convenient tracking, we provide a pretrained model.

Training a completely new model requires a large number of labels.
If you want to easily complete labeling (around 50-150 labels) and then proceed to training, use the pretrained models below and the appropriate skeleton config files for each model.

**How to Use**
- Pretrained model: Save the file to a location of your choice (recommended: MovAl/weights). Select this model when selecting the model to train in the Pose Estimation step.
- Skeleton config: Save the file to Moval/preset/skeleton. Select the corresponding config file in the Create Project step.
- Training strategy: Take about 50 labels for the video you want to track, fine-tune based on the video, and increase the number of labels as needed based on the tracking quality.

## B6 mouse 6kpt model

This model is specialized for tracking up to six B6 mice in various environments. Details and distribution file links are below.

### Environment
- With contoured video

![environments](https://github.com/user-attachments/assets/792e1a71-4b9c-476d-b51f-6ef2acb41e76)

### Keypoint setting
- Keypoint order : Nose, Body_C, Ear_L, Ear_R, Neck, Tail
<img width="299" height="635" alt="skeleton" src="https://github.com/user-attachments/assets/d758b919-68c5-4ca7-808b-910e0036a1b8" />

### Training result
![val_batch2_pred](https://github.com/user-attachments/assets/ce2cf039-2665-4a7b-929b-5e1ca4f344d5)

### Pretrained model
https://drive.google.com/file/d/1ZIEWHQO5Xp38-M1tddwMzOTtm-a-PxL9/view?usp=sharing
### Skeleton config
https://drive.google.com/file/d/15qZW_F3Y1VE9YQXQMnHeDd7hOYjAvDpN/view?usp=drive_link

---

More models will be updated!
