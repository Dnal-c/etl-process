import pandas as pd

from automatic_speech_recognition import speech_recognition
from image_captioning import main as image_captioning
import time

invalided_links = {
    'link': [],
    'row_number': []
}

links = []
descriptions_ru = []
short_descriptions_ru = []
descriptions_en = []
short_descriptions_en = []
tags = []
texts = []


def enrich(row):
    start_time = time.time()
    link = row['link']
    result_caption = {}  # image_captioning.get_video_caption(link)
    result_caption['text'] = speech_recognition.transcribe_video(link)
    print("--- Вот за столько секунд обработали %s видео ---" % (time.time() - start_time))
    return result_caption


def try_to_enrich(row, index):
    link = row['link']
    try:
        result = enrich(row)
        links.append(link)
        tags.append(row['description'])
        # descriptions_ru.append(result['description_ru'])
        # short_descriptions_ru.append(result['short_description_ru'])
        # descriptions_en.append(result['description_en'])
        # short_descriptions_en.append(result['short_description_en'])
        texts.append(result['text'])
    except Exception as inst:
        print('УПАЛИИИ')
        print(type(inst))
        print(inst.args)
        print(inst)
        invalided_links['link'].append(link)
        invalided_links['row_number'].append(index)


if __name__ == '__main__':
    DATASET_PATH = './notebooks/2024_400k.csv'
    dataset = pd.read_csv(DATASET_PATH).copy()
    dataset = dataset.sort_values('link')

    batch = 500
    start = 0
    end = 10000
    for i in range(start, end, batch):
        j = 0
        while j < batch:
            current_number = i + j
            row = dataset.iloc[current_number]
            try_to_enrich(row, current_number)
            print(current_number)
            j += 1
        DATASET_TEMP_PATH = './/test_' + str(i) + '.csv'
        DATASET_FAIL_PATH = './/fail_' + str(i) + '.csv'

        dataframe_for_output = pd.DataFrame(data={
            'link': links,
            'tags': tags,
            # 'description_ru': descriptions_ru,
            # 'short_description_ru': short_descriptions_ru,
            # 'description_en': descriptions_en,
            # 'short_description_en': short_descriptions_en,
            'text': texts
        })

        dataframe_for_output.to_csv(DATASET_TEMP_PATH, index=False)

        invalid_rows_dataframe = pd.DataFrame(data={
            'link': invalided_links['link'],
            'row_number': invalided_links['row_number'],
        })

        invalid_rows_dataframe.to_csv(DATASET_FAIL_PATH, index=False)

        links = []
        # descriptions_ru = []
        # short_descriptions_ru = []
        # descriptions_en = []
        # short_descriptions_en = []
        tags = []
        texts = []
