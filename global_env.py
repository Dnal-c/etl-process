import torch
from enum import Enum

CPU_COUNT = 6
START_INTERVAL = 10000
END_INTERVAL = 11000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_COMPUTE_TYPE = "int8"

LOCAL_MODELS_PATH = '/Users/dandi/PycharmProjects/etl-process/models/'
MODEL_CAPTION_PATH = LOCAL_MODELS_PATH + 'captioning'
MODEL_TRANSLATOR_PATH = LOCAL_MODELS_PATH + 'translator'
DATASET_PATH = './notebooks/2024_400k.csv'

# Параметр, отвечающий за то, куда будем складывать файлы
TEMP_DIRECTORY_PATH = '/Users/dandi/Documents/hack/temp/'
# Параметр, отвечающий за то, сколько фреймов с видео резать за секунду
SAVING_FRAMES_PER_SECOND = .25


class EtlMode(Enum):
    FULL = 1
    ONLY_SPEECH = 2
    ONLY_CAPTIONING = 3


ETL_MODE = EtlMode.ONLY_CAPTIONING
CACHE_ENABLED = True
