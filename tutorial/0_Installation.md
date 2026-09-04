# 0. Installation

## Requirements and Recommendations
- CUDA 11.8 (or 12.1/12.8)
- Supports Windows 11, Ubuntu 22.04 environment (Ubuntu 24.04 may not compatible)
- We recommend using Conda for setting up the environment.
- numpy (>=1.23.0,<2.0) is required to operate Cutie.

## 1. Check Requirements
Check your CUDA version from terminal.
```bash
nvcc --version
```
We recommend to use cuda 11.8, 12.1, 12.8.

**If the command still doesn’t detect the correct CUDA version**, set the system environment variables 
**(System Properties → Environment Variables)** or set them temporarily with the command below.
```bash
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"  # for CUDA 11.8 only
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"  # for CUDA 12.1 only
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"  # for CUDA 12.8 only
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"
```

## 2. Create Virtual Environment
First, create moval venv to avoid conflicts with other programs
```bash
conda create -n moval python=3.9
conda activate moval

cd (folder to download MovAl)
git clone https://github.com/coldlabkaist/MovAl-Move_Altogether.git
```
Additionally, install the required dependencies, including PyTorch. **Check your CUDA version first**
```bash
# for CUDA 11.8 
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# for CUDA 12.1 
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

# for CUDA 12.8
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```  
```bash
conda install "numpy>=1.23.0,<2.0"
cd ./MovAl-Move_Altogether
pip install -r requirements.txt -c constraints.txt
```
Now you can run MovAl UI.
``` bash
python moval.py
```


## 3. One-click installation of Cutie/YOLO dependency
<img width="1498" height="793" alt="image" src="https://github.com/user-attachments/assets/e0ad200b-6540-473b-8384-319948cf430b" />



Click the One Click Install button in the Installation (Cutie/YOLO) tab to easily install dependencies and required models.

If there are any issues, you can install the manual, and when a newer version of the YOLO model is released, you can use the update YOLO button for automatic updates.

## 4. MovAl Update
If your local MovAl version is outdated compared to the latest release version on github, you can see a message recommending an update when running the code. 
You can simply update it using the following command
```bash
cd (MovAl_folder)
python update_moval.py
# or
python update_moval.py v1.2.0
```
If no version is provided, MovAl updates to the latest release tag. If a version is provided, MovAl checks out that tag.

---

**Now you are ready to use MovAl.**

**Troubleshooting**: [Troubleshooting](https://github.com/coldlabkaist/MovAl/blob/main/tutorial/Troubleshooting.md)

Next step: [1. Create Project](https://github.com/coldlabkaist/MovAl/blob/main/tutorial/1_Create_Project.md)
