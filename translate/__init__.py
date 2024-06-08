from googletrans import Translator

translator = Translator()

def translate_frames_caption(source_texts):
    translated_texts = []
    for res in source_texts:
        text = translator.translate(res, src='en', dest='ru')
        translated_texts.append(text.text)
    return translated_texts