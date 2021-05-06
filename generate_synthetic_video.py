import itertools
import random

import cv2
import numpy as np


def generate_repeat_with_various_speed(array,
                                       repeat_speed_x_range=(0.2, 0.7),
                                       repeat_num_range=(2,5),
                                       concat_result=True):
    
    repeat_num = random.randint(*repeat_num_range)

    print(repeat_num)
    results = []
    for _ in range(repeat_num):
        speed_x = random.uniform(*repeat_speed_x_range)

        arr_len = len(array)
        target_len = round(arr_len * (1./speed_x))
        linspace = np.around(
            np.linspace(0, arr_len-1, target_len)).astype(np.int64)

        new_array = []
        for i in linspace:
            new_array.append(array[i])
        
        repeat_array = new_array + new_array[::-1]
        results.append(repeat_array)

    if concat_result:
        return concat_lists(results)

    return results


def concat_lists(lists):
    return list(itertools.chain(*lists))


def generate_repeat_video(video_path,
                          clip_sec_range=(0.5, 2),
                          padding_sec_range=(0, 2),
                          repeat_num_range=(2, 5),
                          repeat_speed_x_range=(0.5, 2),
                          output_fps=30,
                          video_size_wh=(224, 224),
                          video_num=3):

    """
    1. 영상 읽기

    2. 클립 추출
      - 클립 추출은 초단위로
      - 패딩 고려

    3. 반복 영상 생성
      - 

    """

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    clip_frame_size = round(random.uniform(*clip_sec_range) * fps)
    padding_frame_size_s = round(random.uniform(*padding_sec_range) * fps)
    padding_frame_size_e = round(random.uniform(*padding_sec_range) * fps)
    frame_size = padding_frame_size_s + clip_frame_size + padding_frame_size_e

    if frame_size > frame_count:
        raise Exception('framesize is too large')

    start_frame = random.randint(0, frame_count-frame_size)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    if cap.isOpened():
        for _ in range(frame_size):
            success, frame = cap.read()
            if not success:
                break

            # frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, video_size_wh)
            frames.append(frame)

    padding_s = frames[:padding_frame_size_s]
    clip = frames[padding_frame_size_s:padding_frame_size_s+clip_frame_size]
    padding_e = frames[padding_frame_size_s+clip_frame_size:]

    clip_repeat = generate_repeat_with_various_speed(
        array=clip,
        repeat_speed_x_range=repeat_speed_x_range,
        repeat_num_range=repeat_num_range,
        concat_result=True)

    result = concat_lists([padding_s, clip_repeat, clip, padding_e])

    for frame in result:
        cv2.imshow('result', frame)
        cv2.waitKey(30)

if __name__ == '__main__':
    generate_repeat_video(
        video_path='run.mp4',
        clip_sec_range=(0.1, 1),
        padding_sec_range=(0, 2),
        repeat_num_range=(2, 3),
        repeat_speed_x_range=(0.8, 1.2),
        output_fps=30,
        video_size_wh=(854, 480),
        video_num=3
    )
