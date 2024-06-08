import urllib.parse as parse
import os
from PIL import Image
import requests


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
