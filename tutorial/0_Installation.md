# 0. Installation

## Requirements
- CUDA 11.8 or 12.1 (Both versions are compatible) 
- Supports Windows environment (planned to support Linux environment)
- We recommend using Conda for setting up the environment.
- numpy (>=1.23.0,<2.0) is required to operate Cutie.

## 1. Check Requirements
Check your CUDA version from terminal.
```bash
nvcc --version
```
You must use CUDA 11.8 or 12.1 to install pytorch 1.2.1. 

If you are unable to check the appropriate CUDA version using the above command even though you have installed the appropriate CUDA version, 
consider setting the system environment variables through the computer settings or temporarily setting the environment variables using the command below.
```bash
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"  # for CUDA 11.8
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"  # for CUDA 12.1
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"
```

## 2. Create Virtual Environment
First, create moval venv to avoid conflicts with other programs
```bash
conda create -n moval python=3.9
conda activate moval
git clone https://github.com/coldlabkaist/MovAl-Move_Altogether.git
```
Additionally, install the required dependencies, including PyTorch.
```bash
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia  # for CUDA 11.8, Window
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia  # for CUDA 12.1, Window
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 -c pytorch  # for Linux

conda install "numpy>=1.23.0,<2.0"
pip install -r requirements.txt -c constraints.txt
```
Now you can run MovAl UI.
``` bash
python moval.py
```

## 3. One-click installation of Cutie/YOLO dependency
<img width="884" height="602" alt="image" src="https://github.com/user-attachments/assets/cc934834-acac-4529-86a8-26d190afcfba" />

Click the One Click Install button in the Installation (Cutie/YOLO) tab to easily install dependencies and required models.
<img width="558" height="263" alt="image" src="https://github.com/user-attachments/assets/803a8381-deb7-4c56-a3e0-f65798ba8333" />


---

**Now you are ready to use MovAl.**
