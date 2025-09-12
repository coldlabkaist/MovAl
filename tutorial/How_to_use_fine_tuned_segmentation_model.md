# Mouse-Tuned Cutie Models (for MovAl/Cutie)

The base segmentation model handles general object detection well, but it is not optimized for mice. For more stable ID tracking and robust labeling under occlusion, we provide fine-tuned models tailored to mouse datasets.

## Quick Start (Inference)

1. Download models (.pth) from your link.

2. Copy files into:

```MovAl/Cutie/weights```


3. Select a weight file in:

```MovAl/Cutie/cutie/config/gui_config.yaml```

4. Change model before run segmentation

around line ~12

weights: weights/<target_model_name>.pth

**base model example: cutie-base-mega.pth**


5. Run Cutie; it will use the specified weights.


## Available Models & Recommended Use
### [B6 Pretrained model](https://drive.google.com/file/d/1UT4HRKQLJiURKt5voLBWRCZ5Wio8NUtB/view?usp=sharing)

- **B6_pretrained_hc_setting.pth**

  - Environment: Home cage · Strain: B6 · Mice: up to 6

- **B6_pretrained_OF_setting.pth**

  - Environment: Open field · Strain: B6 · Mice: up to 4

- **B6_pretrained_generalized.pth**

  - Environment: Generalized · Strain: B6 · Mice: up to 6

Tip: Run a small labeling test with each model and select the one that best matches your recording setup.

## Optional: Fine-Tuning Guide

Prereqs: GPU required; training time is non-trivial.

Goal: Adapt the model for your exact recording environment.

1. Collect a Small Labeled Set

- Use the Cutie dialog to label multiple contiguous sequences of frames.

- Include both non-occluded and occluded frames.

- Aim for several sets, each with tens of frames (e.g., 10-30).

2. Prepare the Directory Layout

- Keep images (JPG) and masks (PNG) in separate top-level folders.

- For each contiguous sequence, use the same subfolder name under both images/ and masks/.

- Within each sequence, frame numbers must match (only the file extension differs).

Example
```
/path/to/dataset/
├─ images/                  # JPG only
│  ├─ seq_0001/
│  │  ├─ 000000.jpg
│  │  ├─ 000001.jpg
│  │  └─ ...
│  └─ seq_0002/
│     ├─ 000100.jpg
│     └─ ...
└─ masks/                   # PNG only
   ├─ seq_0001/
   │  ├─ 000000.png
   │  ├─ 000001.png
   │  └─ ...
   └─ seq_0002/
      ├─ 000100.png
      └─ ...

# Key rules: images/<SEQ>/<FRAME>.jpg ↔ masks/<SAME_SEQ>/<SAME_FRAME>.png

```

- JPGs and PNGs are grouped separately at the top level.

- Sequence names and frame indices must align exactly.

3. Edit Training Config

- Edit: ```MovAl/Cutie/cutie/config/data/base.yaml```

- If present, remove the old dataset block (example):
```
# main_training:
#   datasets:
#     - DAVIS
#     - YouTubeVO
```

- Replace with the following final config (example):
```
vos_datasets:
  mouse:
    image_directory: /path/to/dataset/images
    mask_directory:  /path/to/dataset/masks
    frame_interval: 1
    subset: null
    empty_masks: null
    multiplier: 1

main_training:
  datasets: [mouse]
  num_iterations: 287500
  lr_schedule_steps: [10000]
```

4. Activate Environment
- ```conda activate moval```

5. Set GPU & Threads

- example : Linux/macOS (GPUs 0,1):
```
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4
```

- example : Windows PowerShell (GPUs 0,1):
```

$env:CUDA_VISIBLE_DEVICES="0,1"
$env:OMP_NUM_THREADS="4"
```

6. Launch Training
```
cd <MovAl directory>/Cutie

torchrun --standalone --nnodes=1 --nproc_per_node=2 ^
         --master_port=29500 ^
         cutie/train.py
# For a single GPU, set --nproc_per_node=1.
```

- If the port is taken, change --master_port to another number.

7. Use the Trained Weights

- Point gui_config.yaml to your new checkpoint:

- ```weights: weights/<your_finetuned_model>.pth```

- The default output path is saved in cutie's output folder.

**Tips & Caveats**

Masks must align with images in size and pixel registration.

If you hit OOM:
Lower batch size or resolution, reduce dataset size, or decrease nproc_per_node.

## Next Step: Speed Up Proofreading

Once tracking is stable, consider the Cutie Labeling Assist Panel Add-On for:

Selective play, Selective re-propagation, single-frame propagation, and bulk ID reassignment to accelerate proofreading.

See: https://github.com/coldlabkaist/Labeling-Assist-Panel-Injection-Code
