# Labelary

**Labelary**—combining “Labeling” and “Library”—is a user‑friendly labeling tool for MovAl pipeline!

<img width="2269" height="777" alt="step3_1" src="https://github.com/user-attachments/assets/03ea6837-c56f-4caf-b5aa-bb1dce6ba035" />
<br><br>

From the main page, click Step 3 Labelary to launch Labelary. In the top panel, select the video, data, and visualization mode to start Labelary.

<img width="1266" height="700" alt="step3_2" src="https://github.com/user-attachments/assets/22c6eeae-02c2-4ea4-a8f9-52d2d63dfbf7" />

<br>

## How to Use Labelary
<img width="2059" height="1149" alt="step3_3" src="https://github.com/user-attachments/assets/b4a426cc-7979-4feb-9461-97c02079841c" />
<br>

- Right‑click to add a new instance or access additional tools (panel A).
  
- Select and drag skeleton points to adjust the label shape as needed (panels B/C).
  
- If a body part is occluded, estimate its position, label it, and update its visibility status (panel D).

- On the right side, choose a skeleton color mode: cutie_light and cutie_dark use Cutie’s default palette, while white and black render the skeleton in a single color.

## How to Save/Export Data
<img width="1654" height="917" alt="step3_4" src="https://github.com/user-attachments/assets/93c3c14e-cfe6-4e80-85ff-76c1f9cbb2f6" />
<br><br>
In the bottom‑right corner, use the Save/Export buttons to preserve your work:

- Save CSV: Quickly saves your current labels in CSV format.

- Export TXT: Converts your labels into YOLO‑compatible TXT files (may take longer for large datasets, Each video can contain only one set of TXT files.).

- Export Video: Renders the video with overlaid labels and saves it (this process can be time‑consuming).

We recommend **using CSV saves during the labeling process** and exporting to TXT files before training.

## Additional Tip
When importing CSV/TXT data, select **Load Inference Result** to load YOLO outputs into Labelary for review and editing.

---

Now you’re ready for [Step 4: Pose Estimation](https://github.com/coldlabkaist/MovAl/blob/main/tutorial/4_Pose_Estimation.md)
