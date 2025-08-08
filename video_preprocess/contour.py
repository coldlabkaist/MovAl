import os
import cv2
import numpy as np

def ContouredVideoProduction(
    output_video_name: str,
    segmented_frames: list,
    masks: list,
    fps: int = 30,
    output_dir: str = None,
    progress_callback=None
):

    low_threshold = 1
    high_threshold = 256

    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(segmented_frames)):
        try:
            original_image = cv2.imread(segmented_frames[i])
            mask = cv2.imread(masks[i])

            edges = cv2.Canny(mask, low_threshold, high_threshold)
            colored_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            contoured_image = np.where(colored_edges > 0, (255, 255, 255), original_image).astype(np.uint8)
            base_name  = os.path.splitext(os.path.basename(segmented_frames[i]))[0]
            """save_path  = os.path.join(output_dir,
                                    f"{output_video_name}_{base_name[2:]}.jpg")"""
            save_path = os.path.join(output_dir, f"{base_name}.jpg")

            cv2.imwrite(save_path, contoured_image)
            
            if progress_callback:
                progress_callback(i + 1)
        except IndexError:
            print(f"Unable to load appropriate frame information of {output_video_name}. Please check again whether the segmentation process has completed.")
            return

    print(f"{len(segmented_frames)} contour images are saved to "
            f"'{output_dir}'")