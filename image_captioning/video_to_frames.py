import os
import uuid
from pathlib import Path

import cv2
import numpy as np

from global_env import TEMP_DIRECTORY_PATH, SAVING_FRAMES_PER_SECOND


def format_timedelta(td):
    """Служебная функция для классного форматирования объектов timedelta (например, 00:00:20.05)
    исключая микросекунды и сохраняя миллисекунды"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return result + ".00".replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")


def get_saving_frames_durations(cap, saving_fps):
    """Функция, которая возвращает интервалы, из которых нужно брать изображение"""
    s = []
    # получаем продолжительность клипа, разделив количество кадров на количество кадров в секунду
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    # используйте np.arange () для выполнения шагов с плавающей запятой
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s


def create_temp_directory_with_frames(video_file):
    uid = uuid.uuid1()
    dirname_str = TEMP_DIRECTORY_PATH + str(uid) + '-opencv'
    dirname = Path(dirname_str)
    # создаем папку по названию видео файла
    if not os.path.isdir(dirname_str):
        dirname.mkdir(exist_ok=True)
    # создаем папку по названию видео файла
    # читать видео файл
    capture = cv2.VideoCapture(video_file)
    # получить FPS видео
    fps = capture.get(cv2.CAP_PROP_FPS)
    # если SAVING_FRAMES_PER_SECOND выше видео FPS, то установите его на FPS (как максимум)
    saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
    # получить список длительностей для сохранения
    saving_frames_durations = get_saving_frames_durations(capture, saving_frames_per_second)
    # запускаем цикл
    count = 0
    frame_insert_count = 1
    while True:
        is_read, frame = capture.read()
        if not is_read:
            # выйти из цикла, если нет фреймов для чтения
            break
        # получаем продолжительность, разделив количество кадров на FPS
        frame_duration = count / fps
        try:
            # получить самую раннюю продолжительность для сохранения
            closest_duration = saving_frames_durations[0]
        except IndexError:
            # список пуст, все кадры длительности сохранены
            break
        if frame_duration >= closest_duration:
            # если ближайшая длительность меньше или равна длительности кадра,
            # затем сохраняем фрейм
            # frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
            # only_file_name = filename.split("/")[-1]
            frame_name = f"frame_{frame_insert_count}.jpg"
            frame_insert_count += 1

            cv2.imwrite(os.path.join(dirname_str, frame_name), frame)
            # удалить точку продолжительности из списка, так как эта точка длительности уже сохранена
            try:
                saving_frames_durations.pop(0)
            except IndexError:
                pass
        # увеличить количество кадров
        count += 1
    return dirname_str
