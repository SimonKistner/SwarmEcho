import cv2
from PIL import Image
import os

def main():
    input_vid = "docs/Preview_vid/eval_update_000572_20260418_151951_20260418_151952_20260418_151952.mp4"
    output_gif = "docs/Preview_vid/demo_2x.gif"
    
    if not os.path.exists(input_vid):
        print(f"Error: {input_vid} not found.")
        return
        
    print(f"Opening {input_vid}...")
    cap = cv2.VideoCapture(input_vid)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 30
        
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # To get 2x speed, we can literally push all frames but halve the duration, 
        # or we skip every other frame and maintain the same duration. 
        # Since GIFs get huge, skipping every other frame is much better.
        if count % 2 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize for optimal GitHub README rendering (600px width)
            h, w = frame_rgb.shape[:2]
            target_w = 600
            target_h = int(h * (target_w / w))
            frame_resized = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
            
            frames.append(Image.fromarray(frame_resized))
            
        count += 1

    cap.release()
    
    if not frames:
        print("No frames extracted.")
        return
        
    # Standard frame duration is 1000ms / fps. we skipped a frame so it's 2 frames per step
    # but we want 2x real speed. Wait:
    # 1x speed: show all frames at (1000/fps) ms.
    # 2x speed: skip half frames, show remaining at (1000/fps) ms -> 2x subjective speed, same framerate.
    duration_ms = int(1000 / fps)
    
    print(f"Saving {len(frames)} frames to {output_gif} at {duration_ms}ms per frame...")
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0
    )
    print("Done!")

if __name__ == "__main__":
    main()
