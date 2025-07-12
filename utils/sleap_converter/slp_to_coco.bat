@echo off
echo Running SLP to COCO GUI...
set SCRIPT_DIR=%~dp0
call conda activate sleap
python "%SCRIPT_DIR%slp_to_coco.py"
pause