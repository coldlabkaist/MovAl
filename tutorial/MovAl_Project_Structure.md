# MovAl Project Structure

Each MovAl project is saved as a self-contained project folder. The project folder stores the project configuration, copied or linked video information, labeling data, preprocessing outputs, YOLO training runs, and inference results.

To reopen a project, load the `project.json` file in the project root. Older projects may still contain `config.yaml`; MovAl treats this as a legacy project file and converts it to the current `project.json` format when needed.

## Overview

```text
my_project/
|-- project.json
|-- raw_videos/
|   `-- video_01.mp4
|-- skeleton/
|   `-- project_skeleton.yaml
|-- frames/
|   `-- video_01/
|       |-- images/
|       |-- masks/
|       `-- visualization/
|           |-- davis/
|           `-- contour/
|-- labels/
|   `-- video_01/
|       |-- csv/
|       `-- txt/
|-- runs/
|   |-- training_config.yaml
|   |-- dataset/
|   |-- train_YYMMDD_HHMMSS/
|   |-- mini_label_exports/
|   `-- mini_datasets/
|-- predicts/
|   `-- run_YYMMDD_HHMMSS/
`-- outputs/
```

## Key Files

- `project.json`: The main project file. It stores the project title, animal IDs, maximum skeletons per ID, video records, skeleton preset name, project-local skeleton data, and UI state such as the last selected video or display mode.

- `skeleton/project_skeleton.yaml`: A project-local copy of the skeleton used by Labelary and YOLO training. Editing the project skeleton updates this file and the skeleton data in `project.json`/`training_config.yaml` simultaneously.

- `runs/training_config.yaml`: The YOLO dataset config generated from the project settings. It points YOLO to the current dataset split under `runs/dataset/` and includes keypoint names, skeleton count, flip index, and animal class names.

## Folders

- `raw_videos/`: Stores local copies of videos when "Copy videos into project raw_videos" is enabled. If videos were added as external links, this folder may be empty. If an external source is missing later, use Project Manager to relink the video or copy it into the project.

- `frames/<video_name>/images/`: Extracted image frames from the source video. These are created during preprocessing or when an image-based workflow needs frame files. If the original video exists, this file can be recreated through a program.

- `frames/<video_name>/masks/`: Cutie segmentation masks. These are important labeling assets and should usually be kept. If the original video exists, this file can be recreated through the program, but additional time is required for GPU computation and proofreading.

- `frames/<video_name>/visualization/davis/`: DAVIS-style visualization frames generated from masks. These are useful as a Labelary display mode. If masks exist, this file can be recreated through a program.

- `frames/<video_name>/visualization/contour/`: Contour visualization frames generated from masks. These are useful for separating nearby animals and can be used as a Labelary or YOLO training display mode. If masks exist, this file can be recreated through a program.

- `labels/<video_name>/csv/`: CSV label files for the video. MovAl can store multiple CSV files here, so this is the best place to keep edited label snapshots during review.

- `labels/<video_name>/txt/`: YOLO-format TXT labels for the video. These are used for YOLO training. A video normally has one active TXT label set for training.

- `outputs/`: Videos exported from Labelary, such as videos with skeleton overlays.

- `predicts/`: YOLO inference results. Each inference creates a timestamped run folder such as `run_260901_231256/`.

- `runs/`: YOLO training and dataset-preparation workspace. Standard training runs, mini-training runs, generated datasets, logs, and checkpoints are stored here.

## Runs Folder

The `runs/` folder can become large because it stores both temporary dataset copies and model checkpoints.

Common subfolders and files:

- `runs/dataset/`: The current train/val/test split created by Step 4 Prepare Dataset. Running Prepare Dataset again replaces this split.

- `runs/dataset/train/images/`, `runs/dataset/val/images/`, `runs/dataset/test/images/`: Images copied or extracted for YOLO training.

- `runs/dataset/train/labels/`, `runs/dataset/val/labels/`, `runs/dataset/test/labels/`: YOLO TXT labels matched to the dataset images.

- `runs/train_YYMMDD_HHMMSS/`: A full YOLO training run.

- `runs/train_YYMMDD_HHMMSS/weights/best.pt`: The best checkpoint from that training run. Use this for inference or further fine-tuning.

- `runs/train_YYMMDD_HHMMSS/weights/last.pt`: The final checkpoint from that training run.

- `runs/train_YYMMDD_HHMMSS/results.csv` and `results.png`: Training metrics and summary plots.

- `runs/train_YYMMDD_HHMMSS/args.yaml`: YOLO training arguments used for the run.

- `runs/mini_label_exports/<video_name>/txt_snapshot_YYMMDD_HHMMSS/`: A TXT snapshot exported from the labels currently loaded in Labelary before mini training.

- `runs/mini_datasets/mini_training_dataset_YYMMDD_HHMMSS/`: The temporary train/val/test dataset created for Labelary mini training.

- `runs/mini_training_YYMMDD_HHMMSS/`: A mini-training YOLO run created from Labelary.

## Predicts Folder

The `predicts/` folder stores pose-estimation outputs from Step 4.

Common output layout:

```text
predicts/
`-- run_YYMMDD_HHMMSS/
    |-- video_01.csv
    |-- video_01.interpolated.csv
    |-- skeletal_output.mp4
    |-- input_videos/
    |   `-- video_01.mp4
    `-- video_01/
        `-- video_01.avi
```

- `video_01.csv`: Converted inference results in MovAl CSV format.

- `video_01.interpolated.csv`: Optional interpolated CSV output when "Raw + interpolated CSV" is selected.

- `video_01/`: YOLO media output folder for that source.

- `input_videos/`: Temporary or saved videos generated when image frames are run through YOLO as a video.

- `skeletal_output.mp4`: A rendered pose-estimation result video when video export is enabled.

If TXT output is enabled during inference, TXT files may also appear inside the run folder. If CSV-only output is selected, MovAl may remove intermediate TXT files after CSV conversion.

## What to Keep

Keep these files and folders when archiving or moving a project:

- `project.json`
- `skeleton/project_skeleton.yaml`
- `raw_videos/` if the project uses copied videos
- `labels/`
- `frames/<video_name>/masks/`
- important model checkpoints such as `runs/*/weights/best.pt`
- desired predicted files

Generated files that can usually be rebuilt or cleaned up:

- `frames/<video_name>/images/`
- `frames/<video_name>/visualization/davis/`
- `frames/<video_name>/visualization/contour/`
- `runs/dataset/`
- temporary mini-training datasets under `runs/mini_datasets/`
- inference previews under `predicts/`

Use Project Manager's Compress Project tool when you want to reduce project size. It keeps raw videos, labels, masks, and project configuration files, while allowing generated images, visualizations, training outputs, and prediction outputs to be removed.
