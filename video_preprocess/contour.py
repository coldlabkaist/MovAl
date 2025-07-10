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
    # 안전장치 여러개 추가 필요 0708 (폴더생성도)
    for i in range(len(segmented_frames)):
        original_image = cv2.imread(segmented_frames[i])
        mask = cv2.imread(masks[i])

        edges = cv2.Canny(mask, low_threshold, high_threshold)
        colored_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        contoured_image = np.where(colored_edges > 0, (255, 255, 255), original_image).astype(np.uint8)
        base_name  = os.path.splitext(os.path.basename(segmented_frames[i]))[0]
        #print(output_dir, output_video_name)
        save_path  = os.path.join(output_dir,
                                f"{output_video_name}_{base_name[2:]}.jpg")

        #print(save_path)
        
        cv2.imwrite(save_path, contoured_image)
        
        if progress_callback:
            progress_callback(i + 1)

    print(f"[{len(segmented_frames)} 장의 컨투어 이미지가 "
      f"'{output_dir}'에 저장되었습니다.")