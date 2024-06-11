import os
import urllib.parse as parse

import imagehash
import requests
from PIL import Image


# функция, определяющая, является ли строка URL-адресом или нет
def is_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False


# фукнция загрузки изображения
def load_image(image_path):
    if is_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)


def get_image_hash(image_path):
    current_image = Image.open(image_path)
    image_hash_temp = imagehash.average_hash(current_image)
    return image_hash_temp


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
