# 2.  Preprocess

## Preprocess GUI
With your project loaded, click Preprocess to split the video into individual images and perform segmentation.

**Important**: Before preprocessing, make sure you have completed Dependency **One‑Click Install** or **Cutie Installation** during setup.

<br>

Clicking Preprocess opens the GUI shown below.

<img width="1655" height="938" alt="step2_1" src="https://github.com/user-attachments/assets/fae0c320-913c-479b-b396-03ce940a7a14" />


## Segmentation

<br>

<img width="2225" height="1017" alt="step2_2" src="https://github.com/user-attachments/assets/9f1e2388-5fbf-44e6-8665-fcb950fbe4bc" />

<br><br>

In the **Segment** tab, you can split your video into frames and run segmentation with Cutie.
Clicking **Create Image Frames** will extract frames from all project videos at once and set up the Cutie workspace. You don’t need to use this button to proceed, but without it, the first Cutie launch for each video will take longer.

This process can be time‑consuming. You can monitor progress in the terminal.

<br>
<img width="2056" height="1018" alt="step2_3" src="https://github.com/user-attachments/assets/8bd44ef8-0740-4d01-bafc-c7826d44c31a" />
<br><br>

Select a video from the list at the top and click Run Segmentation to launch Cutie.

- If you encounter a dependency error at this stage, it’s likely due to your NumPy version.
We recommend rerunning the installation steps to recreate your environment.

## How to Use Cutie
Below is a quick overview of how to use Cutie. For more detailed instructions, please refer to the [Cutie documentation](https://github.com/hkchengrex/Cutie).

<img width="1598" height="907" alt="step2_4" src="https://github.com/user-attachments/assets/1a6d25bf-5ca9-44d6-a528-321da8dcb650" />
<br><br>

In the video area (A), left‑click to mark an object and right‑click to mark the background.
To select an object, left‑click on its region—exclude shadows or reflections—and ensure every object in the frame is marked.
In the bottom panel (B), set the object ID (by typing a number or using the up/down arrow keys) to change its color.

<br>
<img width="1598" height="905" alt="step2_5" src="https://github.com/user-attachments/assets/94e151ba-1651-4fb0-9b45-392a88405502" />
<br><br>

Once you’ve segmented the first frame, you can propagate those labels to the next frame.

In the bottom panel (A), set Visualization to **davis** and **Always**.
In the lower‑right panel (B), click **Propagate** to auto‑label the following frame.

**Important: Do not propagate and play the video at the same time**, as this may corrupt your mask data.

<br>
<img width="1614" height="915" alt="step2_6" src="https://github.com/user-attachments/assets/45a2737a-ed07-4e84-b0c1-edb125a5b3e1" />
<br><br>

If unwanted labels appear during propagation, click Stop and return to the frame where the error first occurred.
To prevent the mistake from carrying forward, clear Non‑Permanent Memory (panel B).

In the video area, fix the labels: **first erase any incorrect marks, then label any missing objects.**
Once corrected, click Propagate again to continue. 
Once the issue is resolved, continue to the next step.
If it persists, navigate to the frame where the new error began and repeat the process above.

<br>
<img width="513" height="644" alt="step2_7" src="https://github.com/user-attachments/assets/251f988e-94a5-47ff-b947-43e1eb46ff57" />
<br><br>
There’s no need to save your work in Cutie. When you’re done, close the Cutie window and
switch back to the MovAl tab to review your progress as shown above.

## (Optional) How to Create Contour
<br>
<img width="1098" height="429" alt="step2_8" src="https://github.com/user-attachments/assets/56c297db-9bb7-4397-8b37-35e706dbe4c2" />
<br><br>

To sharpen boundaries between objects, you can convert the segmentation mask into a contour. 
After completing segmentation, click Contour in the previous tab to automatically generate the contoured image.

<br>
The difference between segmentation and contour is as below

<br><br>
<img width="798" height="384" alt="step2_9" src="https://github.com/user-attachments/assets/c72e88b8-29c6-491d-a838-19469ed7a101" />

---

Now you’re ready for [Step 3: Labelary](https://github.com/coldlabkaist/MovAl/blob/main/tutorial/3_Labelary.md)

