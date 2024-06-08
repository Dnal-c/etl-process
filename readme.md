# Команда "Клан Д". ETL-процессы
В данном репозитории содержится пайпы для наполнения данными хранилища, по которому будет осуществлять поиск видеозаписей

Перед начало разбора кода просьба ознакомиться с описанием того, какую логику пресследовали экстракте данных с видео

- Ознакомление и анализ полученной выборки - [Здесь будет ссылка](#some-title-1)
- Выработка алгоритма для обогащения данных - [Здесь будет ссылка](#some-title-1)
- Объединение результатов работы моделей - [Здесь будет ссылка](#some-title-1)
- Объединенные пайпы в .ipynb формате, необходим был, чтобы выполнять код с GPU на Kaggle - [Здесь будет ссылка](#some-title-1)

Также хотелось бы зарезюмировать, что в конечном итоге паралелить пайп не получилось на скорую руку (за N дней < 2 недель). Пробовали варианты для pandarallel и thread-пулы, но в конечном итоге каждый раз появлялся неочевидный боттлнек, который не успели расшить.


Подробнее по структуре пакетов и описание отдельных файлов:

<pre>
├── automatic_speech_recognition - содержит код для распознования текста с видео
├── data_union - содержит код для объединения результатов работы модели
├── image_captioning - содержит код для распознавания текста с видео
├── notebooks - содержит jupiter-тетрадки, опубликованные в Kaggle и описанные выше
├── translate - код для перевода текста на русский 
├── main.py - первый пайплан для сбора данных в обогащенные csv
├── main_two.py - второй пайплайн для публикации обогащенных данных с csv в хранилище
├── requirements.txt - зависимости
└── test_data.csv - пример того, как выглядят обогащенные данные
</pre>


# Инструкция для работы локально
### В первую очередь необходимо поменять конфигурацию

#### Из какой ветки запускаться
- если хотите парсить все сразу - ветка **main**
- если собирать только речь с видео - ветка **speech**
- если осуществлять обогащение по контенту с видео - **ветка caption**

#### Системные требования для старта помимо установленной cuda
- не забыть проверить наличие ffmpeg на ПК. Нужно для whisper
- поставить через pip все зависимости из requirements.txt в ту среду, откуда будете запускать скрипт

#### Конфигурация параметров
1. В файле main.py в рутовой директории необходимо изменить следующее
   - DATASET_PATH - выставить корректный путь до исходного датасета из ТЗ
   - DATASET_TEMP_PATH - в первую часть до str(i) проставить префикс с директорией и названием файла, в которые будут сохраняться обогащенные фреймы
   - DATASET_FAIL_PATH - аналогично DATASET_TEMP_PATH, только префикс для хранения инфы о файлах, которые упали
   - batch - с помощью параметра задаем размер батча для одного фрейма. То есть, сколько записей собираем в 1 файл
   - start - с какой записи исходного фрейма начинаем сборку
   - end - на какой записи исходного фрейма заканчиваем сборку
2. Файл image_captioning/video_to_frames.py
   - BASE_PATH - указать путь для созданной temp директории, куда будут подкладываться временные папки с картинками
3. Файл automatic_speech_recognition/speech_recognition.py
   - compute_type - если запуск идет с CPU (без cuda) - выставляем float32, иначе float16
4. Файл data_union/data_union.py
   - speech_path - указать директорию, где лежат чанки от ImageCaptioning
   - captioning_path - указать директорию, где лежат чанки от ASR
   - output_captioning_path - указать название файла, куда сложится результат агрегации