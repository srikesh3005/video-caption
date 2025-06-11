import cv2
from transformers import pipeline
from PIL import Image


pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

def frame_extract(video_path, num_frames=6):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb_image))
    cap.release()
    return frames

def caption_video(video_path):
    frames = frame_extract(video_path, num_frames=6)
    captions = []
    for i, frame in enumerate(frames):
        print(f"Captioning frame {i+1}...")
        result = pipe(frame)
        if isinstance(result, list) and 'generated_text' in result[0]:
            captions.append(result[0]['generated_text'])
        else:
            captions.append("No caption")
    final_caption = max(set(captions), key=captions.count)
    return final_caption, captions

if __name__ == "__main__":
    video_path = "/ml projects/akai/video/10.avi"
    final_caption, all_captions = caption_video(video_path)

    print("Final Caption")
    print(final_caption)

    print("\nAll Captions")
    for i, cap in enumerate(all_captions, 1):
        print(f"Frame {i}: {cap}")
