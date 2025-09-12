# Troubleshooting (Common Errors)

## 1. OmegaConf / dependency error on startup

Symptom: 
Import/initialization fails with messages related to omegaconf or version conflicts.

Cause: 
An incompatible NumPy version is installed, which breaks Cutie’s dependency.

Fix: 
Recreate the moval virtual environment exactly as described in the MovAl installation tutorial.

Tips: 
Do not arbitrarily upgrade NumPy; stick to the versions pinned by the tutorial. 
Avoid mixing conda and pip unless the tutorial explicitly instructs it.


## 2. OSError: [WinError 182] ... fbgemm.dll (Windows / PyTorch)

Symptom: 
Runtime error loading fbgemm.dll (or one of its dependencies) when launching a Torch-based program:

OSError: [WinError 182] The operating system cannot run %1.
Error loading ".../torch/lib/fbgemm.dll" or one of its dependencies.

Cause: 
Missing MSVC C++ runtime and/or OpenMP runtime required by PyTorch on Windows.

Fix: 
Install Visual Studio (or Build Tools for Visual Studio) with the “Desktop development with C++” workload.
This provides the required MSVC runtime for Torch.

If the error persists, install the Intel OpenMP runtime:
```pip install intel-openmp```

Restart your shell (or reboot) and try again.

Notes

Ensure your PyTorch build matches your Python version and CUDA/driver setup.

If you recently changed CUDA or drivers, reinstalling a matching PyTorch build can help.
