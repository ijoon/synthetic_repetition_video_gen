import numpy as np
import cv2


def read_video(video_filename, width=224, height=224):
    """Read video from file."""
    cap = cv2.VideoCapture(video_filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    if cap.isOpened():
        while True:
            success, frame_bgr = cap.read()
            if not success:
                break
            frame_rgb = frame_bgr#cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            #   frame_rgb = cv2.resize(frame_rgb, (width, height))
            frames.append(frame_rgb)
    frames = np.asarray(frames)
    return frames, fps

def augment_video(file, seq, num_repeat):
    frames, fps = read_video(file)
    # frames = frames[::2]
    num_frames = len(frames)
    start_idx = np.random.randint(100, num_frames-seq)
    repeat_start = np.random.randint(0, 10)
    repeat_range = np.random.randint(10, 15)
    aug_frames = frames[start_idx:start_idx+repeat_start]
    clip = frames[start_idx+repeat_start:start_idx+repeat_start+repeat_range]
    print('# of clip frames:', len(clip))
    for _ in range(num_repeat):
        aug_frames = np.append(aug_frames, clip, axis=0)
        clip = clip[::-1, :, :, :]
    aug_frames = np.append(aug_frames, clip, axis=0)
    s = start_idx + repeat_start + repeat_range# - len(clip)
    e = s+seq-len(aug_frames)
    print(s, e)
    # exit(1)
    aug_frames = np.append(aug_frames, frames[s:e], axis=0)
    print('return frames:', len(aug_frames))
    return aug_frames, fps

file_name = 'test.mp4'
frames, fps = augment_video(file_name, 128, 6)

wait_milli = int(1000. / fps)
while True:
    for img in frames:
        cv2.imshow('aug_video', img)
        cv2.waitKey(wait_milli)
    if cv2.waitKey(0) == ord('q'):
        break
