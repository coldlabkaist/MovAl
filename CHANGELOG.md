## MovAl v1.2.x

**26.06.01.** MovAl v1.2.1-dev updates
- Add project-level per-ID skeleton limit configuration and improve multi-instance labeling/review flows.
- Improve Labelary frame alignment with delay-based sync and corrected CSV index saving.
- Standardize inference TXT loading to zero-based indexing and improve frame matching behavior.
- Improve Labelary safety when reloading or closing with unsaved changes.
- Reorganize inference outputs by run root and support CSV-only inference export without keeping TXT files.
- Add a YOLO-only updater in Installation Manager and share YOLO model download helpers.

**26.05.19.** MovAl v1.2.0 released
- Add video-based pipeline support and improve inference workflow convenience.
- Add background video export in Labelary and improve Labelary playback performance.
- Improve mini-training data split threading and stabilize task/compression handling.
- Improve project compatibility for legacy YAML conversion and missing skeleton preset/config cases.
- Revise Project Manager, preprocessing, training/additional tools, and general UI/theme behavior.
- Fix Labelary video playback, CSV-save shortcut conflicts, and selection sync edge cases.

## MovAl v1.1.x

**26.04.15** MovAl v1.1.4 released
- 1.1.4 remember the last opened project config.
- Improve Labelary detail settings behavior
- Add update assistant

**26.04.08.** MovAl v1.1.3 released
- Add “Select All” option for video frames in inference

**26.04.09.** MovAl v1.1.2 released
- Fix frame misalignment issue in Labelary

**26.04.08.** MovAl v1.1.1 released
- Auto-labeling assistant update
- Improved labeling convenience

**25.10.16.** MovAl v1.1.0 released
- Improved Training Stability
- Fixed visibility Bug
- Fixed Labelary key release error
- New Feature: Labeled-Frame Navigation in Labelary
- CSV files from versions 1.0.x may be incompatible.

**In such cases, rename the column frame.idx to frame_idx in the CSV file.**



## MovAl v1.0.x

**25.09.15.** MovAl v1.0.4 released
- Resolve index shift issue of labelary
- Clean up and standardize config file paths
- Add selective contour generation

**25.08.14.** MovAl v1.0.3 released
- Resolved CPU management issue
- Bug fixed path for txt2csv

**25.08.08.** MovAl v1.0.2 released
- Resolved data split errors
- Enhanced exception handling

**25.08.01.** MovAl v1.0.1 released 
- Improve YOLO inference UI.
- Fix bugs related to video export and skeleton visualization settings.

**25.07.16.** MovAl v1.0.0 released!
