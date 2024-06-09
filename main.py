import os.path
import shutil

import pandas as pd

from automatic_speech_recognition import speech_recognition
from global_env import DATASET_PATH, START_INTERVAL, END_INTERVAL, CPU_COUNT, TEMP_DIRECTORY_PATH, ETL_MODE, EtlMode, \
    CACHE_ENABLED
from image_captioning import image_caption as image_captioning
import time
import global_context

import multiprocessing as mp

from image_captioning.utils import get_image_hash
from image_captioning.video_to_frames import create_temp_directory_with_frames

invalided_links = []

dataframe_data = []
dataset = pd.read_csv(DATASET_PATH).copy()
dataset = dataset.sort_values('link')

# Собираем мапу с хешами и обработанными записями для кэша
DATASET_WITH_HASH_PATH = ''
current_dataset_with_hash = pd.DataFrame([])  # pd.read_csv(DATASET_WITH_HASH_PATH)
hashWithRowMap = {}

for _, row_with_hash in current_dataset_with_hash.iterrows():
    row_hash = row_with_hash['hash']
    hashWithRowMap[row_hash] = row_with_hash.to_dict()


# texts = []


def enrich(row, index):
    start_time_enrich = time.time()

    link = row['link']
    tags = row['description']
    video_hash = None

    enrich_result = {
        'index': index,
        'link': link,
        'tags': tags
    }

    if ETL_MODE != EtlMode.ONLY_SPEECH:
        directory_name = create_temp_directory_with_frames(link)
        first_file_name = os.path.join(directory_name, 'frame_1.jpg')
        video_hash = get_image_hash(first_file_name)
        image_captioning_result = try_to_get_description_by_hash(directory_name)
        if image_captioning_result is None:
            image_captioning_result = image_captioning.get_video_caption(directory_name,
                                                                         model=global_context.image_captioner,
                                                                         image_processor=global_context.image_processor,
                                                                         tokenizer=global_context.image_tokenizer)
        enrich_result.update(image_captioning_result)
        # удаляем temp директорию
        shutil.rmtree(directory_name)
        # Прихраниваем результат в кэш
        hashWithRowMap[video_hash] = enrich_result

    if ETL_MODE != EtlMode.ONLY_CAPTIONING:
        speech_result = speech_recognition.transcribe_video(link)
        enrich_result.update(speech_result)

    print('Вот такая строчка: ', index)
    print("--- Вот за столько секунд обработали %s видео ---" % (time.time() - start_time_enrich))
    enrich_result['hash'] = video_hash
    return enrich_result


def try_to_get_description_by_hash(video_hash):
    video_from_hash_map = hashWithRowMap.get(video_hash)
    if video_from_hash_map is not None:
        print('Найдено видео по хэшу', video_hash)
        return video_from_hash_map


def add_row_to_invalid_links(row, index):
    failed_result = row
    failed_result['index'] = index
    invalided_links.append(row)


def try_to_enrich(row, index):
    try:
        return enrich(row, index)
    except Exception as inst:
        print(type(inst))
        print(inst.args)
        print(inst)
        add_row_to_invalid_links(row, index)
        return {}


def enrich_task(start_index, end_index):
    global dataset
    enrich_results = []
    for index_id in range(start_index, end_index):
        row = dataset.iloc[index_id]
        enrich_result = try_to_enrich(row, index_id)
        if enrich_result is not None:
            enrich_results.append(enrich_result)
    return enrich_results


def filter_na_values(row):
    if row.get('link') is None:
        return False
    return True


def get_task_cursors():
    task_cursors_internal = []
    batch_size = (END_INTERVAL - START_INTERVAL) / CPU_COUNT
    temp_start = START_INTERVAL
    while temp_start < END_INTERVAL:
        i = int(temp_start)
        j = int(min(i + batch_size, END_INTERVAL))

        task_cursors_internal.append((i, j))
        temp_start += batch_size
    return task_cursors_internal


if __name__ == '__main__':
    start_time = time.time()
    task_cursors = get_task_cursors()

    with mp.Pool(CPU_COUNT) as thread_pool:
        dataframe_data = thread_pool.starmap(enrich_task, task_cursors)
        dataframe_data = [chuck_data for chunk in dataframe_data for chuck_data in chunk]
        dataframe_data = filter(filter_na_values, dataframe_data)

        DATASET_TEMP_PATH = TEMP_DIRECTORY_PATH + 'test_' + str(START_INTERVAL) + '.csv'
        DATASET_FAIL_PATH = TEMP_DIRECTORY_PATH + 'fail_' + str(START_INTERVAL) + '.csv'

        dataframe_for_output = pd.DataFrame(dataframe_data)
        dataframe_for_output.to_csv(DATASET_TEMP_PATH, index=False)

        dataframe_with_failed_data_output = pd.DataFrame(invalided_links)
        dataframe_with_failed_data_output.to_csv(DATASET_FAIL_PATH, index=False)

    evaluation_time = (time.time() - start_time) / 60

    print("--- Работа пайпа в минутах %s ---" % str(evaluation_time))
