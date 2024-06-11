import os
import random
import time

import global_context
import translate.translator as translate
from .utils import load_image


# функция инференса
def get_caption_by_image(model, image_processor, tokenizer, image_path):
    image = load_image(image_path)
    # предобработка
    img = image_processor(image, return_tensors="pt").to(global_context.DEVICE)
    # генерируем описание
    output = model.generate(**img)
    # декодим вывод
    return tokenizer.batch_decode(output, skip_special_tokens=True)[0]


def get_video_caption(directory_name, model, image_processor, tokenizer):
    # объявим массив, в который будем складывать результаты предсказаний по фреймам
    full_english_descriptions = []
    start_time_desc = time.time()
    for dirname, _, filenames in os.walk(directory_name):
        for filename in filenames:
            full_name = os.path.join(dirname, filename)
            file_english_description = get_caption_by_image(model, image_processor, tokenizer, full_name)
            full_english_descriptions.append(file_english_description)

    # избавляемся от явных дублей
    full_english_descriptions = list(set(full_english_descriptions))
    full_russian_descriptions = translate.translate_frames_caption(full_english_descriptions)
    descriptions_length = len(full_russian_descriptions) - 1

    random_frame_number = random.randint(0, descriptions_length)

    random_russian_description = full_russian_descriptions[random_frame_number]
    random_english_description = full_english_descriptions[random_frame_number]

    full_description_en = ' '.join(full_english_descriptions)
    full_description_ru = ' '.join(full_russian_descriptions)
    print("--- Работа процесса в секундах %s ---" % str(time.time() - start_time_desc))

    return {
        'description_ru': full_description_ru,
        'short_description_ru': random_russian_description,
        'description_en': full_description_en,
        'short_description_en': random_english_description
    }
