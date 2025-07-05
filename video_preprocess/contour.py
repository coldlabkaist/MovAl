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

    if output_dir is None:
        output_video_dir = os.path.join("Videos", "Contoured Videos")
    else:
        output_video_dir = output_dir

    if not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir)

    output_video_prefix = "Contoured_"
    output_video_path = os.path.join(output_video_dir, output_video_prefix + output_video_name + ".mp4")

    if len(segmented_frames) == 0:
        print(f"[{output_video_name}] No segmented frames available.")
        return

    frame = cv2.imread(segmented_frames[0])
    if frame is None:
        print(f"[{output_video_name}] Failed to read first frame: {segmented_frames[0]}")
        return
    height, width, _ = frame.shape 

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    low_threshold = 1
    high_threshold = 256

    for i in range(len(segmented_frames)):
        original_image = cv2.imread(segmented_frames[i])
        mask = cv2.imread(masks[i])
        if original_image is None or mask is None:
            print(f"[{output_video_name}] Error reading frame or mask at index {i}.")
            continue

        edges = cv2.Canny(mask, low_threshold, high_threshold)
        colored_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        contoured_image = np.where(colored_edges > 0, (255, 255, 255), original_image).astype(np.uint8)
        video_writer.write(contoured_image)

        if progress_callback:
            progress_callback(i + 1)

    video_writer.release()
    cv2.destroyAllWindows()
    print(f"[{output_video_name}] Video saved at: {output_video_path}")
