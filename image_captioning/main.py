import os
import shutil
import random
import time

import torch

import translate
from .video_to_frames import create_temp_directory_with_frames
from .utils import load_image
from transformers import GPT2TokenizerFast, ViTImageProcessor, pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
# модель кодировщика, которая обрабатывает изображение и возвращает его фичи

# encoder_model = "WinKawaks/vit-small-patch16-224"
# encoder_model = "google/vit-base-patch16-224"
# encoder_model = "google/vit-base-patch16-224-in21k"
encoder_model = "microsoft/swin-base-patch4-window7-224-in22k"

# модель декодера, которая обрабатывает элементы изображения и генерирует текст подписи

# decoder_model = "bert-base-uncased"
# decoder_model = "prajjwal1/bert-tiny"
decoder_model = "gpt2"

# Инициализируем токенайзер
# tokenizer = AutoTokenizer.from_pretrained(decoder_model)
tokenizer = GPT2TokenizerFast.from_pretrained(decoder_model)
# tokenizer = BertTokenizerFast.from_pretrained(decoder_model)
# Загружаем обработчик изображений
image_processor = ViTImageProcessor.from_pretrained(encoder_model)

# Используя pipeline api
image_captioner = pipeline("image-to-text", model="Abdou/vit-swin-base-224-gpt2-image-captioning")
image_captioner.model = image_captioner.model.to(device)


# функция инференса
def get_caption_by_image(model, image_processor, tokenizer, image_path):
    image = load_image(image_path)
    # предобработка
    img = image_processor(image, return_tensors="pt").to(device)
    # генерируем описание
    output = model.generate(**img)
    # декодим вывод
    return tokenizer.batch_decode(output, skip_special_tokens=True)[0]


# функция для получения краткого описания
def get_short_description(string):
    i = 255
    if len(string) > 255:
        while i > 0:
            if len(string) > 255:
                if string[i] not in [' ', '.', ',']:
                    i -= 1
                else:
                    break
    return string[:i]


def get_video_caption(video_file):
    directory_name = create_temp_directory_with_frames(video_file)
    # объявим массив, в который будем складывать результаты предсказаний по фреймам
    full_english_descriptions = []
    for dirname, _, filenames in os.walk(directory_name):
        for filename in filenames:
            full_name = os.path.join(dirname, filename)
            file_english_description = get_caption_by_image(image_captioner.model, image_processor, tokenizer, full_name)
            full_english_descriptions.append(file_english_description)
    # удаляем temp директорию
    shutil.rmtree(directory_name)

    # избавляемся от явных дублей
    full_english_descriptions = list(set(full_english_descriptions))

    start_time = time.time()
    full_russian_descriptions = translate.translate_frames_caption(full_english_descriptions)
    print("--- Перевод %s записи ---" % (time.time() - start_time))
    descriptions_length = len(full_russian_descriptions) - 1

    random_frame_number = random.randint(0, descriptions_length)

    random_russian_description = full_russian_descriptions[random_frame_number]
    random_english_description = full_english_descriptions[random_frame_number]

    full_description_en = ' '.join(full_english_descriptions)
    full_description_ru = ' '.join(full_russian_descriptions)

    return {
        'description_ru': full_description_ru,
        'short_description_ru': random_russian_description,
        'description_en': full_description_en,
        'short_description_en': random_english_description
    }
