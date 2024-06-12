from enum import Enum

import torch

CPU_COUNT = 2
START_INTERVAL = 50000
END_INTERVAL = 100000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Если запуск идет с CPU, то необходимо float32 или float16, на GPU можно пробовать int8, float16, float32
WHISPER_COMPUTE_TYPE = "int8" if DEVICE == "gpu" else "float32"

BASE_DIR_PATH = '/root/projects/etl-process/'
MODEL_CAPTION_PATH = BASE_DIR_PATH + '/models/captioning'  # нужен для локальной работы с image captioning
MODEL_TRANSLATOR_PATH = BASE_DIR_PATH + '/models/translator'  # нужен для локальной работы с automatic speech recogni
DATASET_PATH = BASE_DIR_PATH + 'source_dataset.csv'  # путь до исходной выборки

TEMP_DIRECTORY_PATH = '/root/projects/temp/'  # Параметр, отвечающий за то, куда будем складывать файлы

SAVING_FRAMES_PER_SECOND = .25  # Параметр, отвечающий за то, сколько фреймов с видео резать за секунду для обработки


class EtlMode(Enum):
    FULL = 1
    ONLY_SPEECH = 2  # совет, запускать только с GPU иначе будет 1 предсказание занимает в районе минуты
    ONLY_CAPTIONING = 3


ETL_MODE = EtlMode.ONLY_SPEECH
CACHE_ENABLED = True # нужно ли подрубать кэш в рантайме для оптимизации работы модели
