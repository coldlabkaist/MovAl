# Troubleshooting (Common Errors)

## 1. OmegaConf / dependency error on startup

**Symptom:**
Import/initialization fails with messages related to omegaconf or version conflicts.

**Cause:**
An incompatible NumPy version is installed, which breaks Cutie’s dependency.

**Fix:** 
Recreate the moval virtual environment exactly as described in the MovAl installation tutorial.

**Tips:** 
Do not arbitrarily upgrade NumPy; stick to the versions pinned by the tutorial. 
Avoid mixing conda and pip unless the tutorial explicitly instructs it.

## 2. Error During Cutie Oneclick Installation

**Symptom:** Errors occur during Cutie installation, especially in the model download process.

**Cause:** 
In many cases, these issues are related to PyTorch dependencies. Please review the following points:

- Check for fbgemm.dll errors.
(See the solution section below.)

- Verify your Ubuntu version.
If you are using Ubuntu 24.04, be aware that due to Torch compatibility issues, MovAl does not support Ubuntu 24.04.

- Confirm Torch and CUDA functionality in your environment.
If PyTorch cannot be successfully imported, MovAl will also fail to run.

```
conda activate moval
python - <<PY
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
```


## 3. OSError: [WinError 182] ... fbgemm.dll (Windows / PyTorch)

**Symptom:** 
Runtime error loading fbgemm.dll (or one of its dependencies) when launching a Torch-based program:

OSError: [WinError 182] The operating system cannot run %1.
Error loading ".../torch/lib/fbgemm.dll" or one of its dependencies.

**Cause:** 
Missing MSVC C++ runtime and/or OpenMP runtime required by PyTorch on Windows.

**Fix:** 
Install Visual Studio (or Build Tools for Visual Studio) with the “Desktop development with C++” workload.
This provides the required MSVC runtime for Torch.

If the error persists, install the Intel OpenMP runtime:
```pip install intel-openmp```

Restart your shell (or reboot) and try again.


## Other Common Issues
**Cutie runs too slowly**

- When running Cutie, the terminal should explicitly display that CUDA is being used.
If this message does not appear, segmentation may be running on CPU instead of GPU.
- In this case, please recheck your dependency configuration and ensure CUDA is correctly enabled.
- **Notes:**
Ensure your PyTorch build matches your Python version and CUDA/driver setup.
