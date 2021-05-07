import itertools
import os
import random

import cv2
import numpy as np


def generate_repeat_with_various_speed(array,
                                       repeat_speed_x_range=(0.2, 0.7),
                                       repeat_num_range=(2,5),
                                       concat_result=True):
    
    repeat_num = random.randint(*repeat_num_range)

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
                          generate_num=3,
                          output_dir='result/'):

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
    
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    error_count = 0
    for i in range(generate_num):
        clip_frame_size = round(random.uniform(*clip_sec_range) * fps)
        padding_frame_size_s = round(random.uniform(*padding_sec_range) * fps)
        padding_frame_size_e = round(random.uniform(*padding_sec_range) * fps)
        frame_size = padding_frame_size_s+clip_frame_size+padding_frame_size_e

        if frame_size > frame_count:
            error_count += 1
            print('{}: frame size is too large. (count {})'.format(
                os.path.basename(video_path), error_count))
                
            if error_count >= 10:
                break
        
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

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        video_out_path = os.path.join(output_dir, f'{video_name}_{i}.mp4')
        out = cv2.VideoWriter(video_out_path, fourcc, output_fps, video_size_wh)

        for frame in result:
            out.write(frame)

        out.release()

if __name__ == '__main__':
    generate_repeat_video(
        video_path='sample_video/run.mp4',
        clip_sec_range=(1, 1.5),
        padding_sec_range=(0, 1),
        repeat_num_range=(2, 5),
        repeat_speed_x_range=(0.7, 1.3),
        output_fps=30,
        video_size_wh=(854, 480),
        generate_num=3,
        output_dir='result/'
    )
