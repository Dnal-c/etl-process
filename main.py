import multiprocessing as mp
import os.path
import shutil
import time

import pandas as pd

import global_context
from automatic_speech_recognition import speech_recognition
from global_env import DATASET_PATH, START_INTERVAL, END_INTERVAL, CPU_COUNT, TEMP_DIRECTORY_PATH, ETL_MODE, EtlMode
from image_captioning import image_caption as image_captioning
from image_captioning.utils import get_image_hash
from image_captioning.video_to_frames import create_temp_directory_with_frames

invalided_links = []

dataframe_data = []
dataset = pd.read_csv(DATASET_PATH).copy()
dataset = dataset.sort_values('link')

# Собираем мапу с хешами и обработанными записями для кэша
dataset_with_hash_path = ''
current_dataset_with_hash = pd.DataFrame([])  # pd.read_csv(dataset_with_hash_path)
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
        print("")
        speech_text = speech_recognition.recognize_speech(link)
        enrich_result['text'] = speech_text

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


def get_file_prefix():
    match ETL_MODE:
        case ETL_MODE.ONLY_SPEECH:
            return 'speech_'
        case ETL_MODE.ONLY_CAPTIONING:
            return 'captioning_'
        case _:
            return 'full_data_'


def get_temp_file_full_name():
    file_prefix = get_file_prefix()
    success_file_path = TEMP_DIRECTORY_PATH + file_prefix + str(START_INTERVAL) + '-' + str(END_INTERVAL) + '.csv'
    fail_file_path = TEMP_DIRECTORY_PATH + 'fail_' + str(START_INTERVAL) + '-' + str(END_INTERVAL) + '.csv'
    return success_file_path, fail_file_path


if __name__ == '__main__':
    start_time = time.time()

    dataframe_data = enrich_task(START_INTERVAL, END_INTERVAL)
    dataframe_data = filter(filter_na_values, dataframe_data)

    full_file_names = get_temp_file_full_name()
    success_file_name = full_file_names[0]
    fail_file_name = full_file_names[1]

    dataframe_for_output = pd.DataFrame(dataframe_data)
    dataframe_for_output.to_csv(success_file_name, index=False)

    dataframe_with_failed_data_output = pd.DataFrame(invalided_links)
    dataframe_with_failed_data_output.to_csv(fail_file_name, index=False)

    evaluation_time = (time.time() - start_time) / 60

    print("--- Работа пайпа в минутах %s ---" % str(evaluation_time))
