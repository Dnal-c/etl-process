import translators as ts
from googletrans import Translator

from global_context import translator_tokenizer, translator_model

translator = Translator()
##ts.preaccelerate_and_speedtest()


def translate_frames_caption(source_texts):
    try:
        return translate_frames_caption_by_google(source_texts)
    except Exception as inst:
        print('Уперлись в rate-limiter google')
        try:
            return translate_frames_caption_by_open_ai(source_texts)
        except Exception as inst:
            print('Проблемы на OpenAI')
            print(type(inst))
            print(inst.args)
            print(inst)
            return translate_frames_caption_by_model(source_texts)


def translate_frames_caption_by_google(source_texts):
    translated_texts = []
    for source_text in source_texts:
        text = translator.translate(source_text, src='en', dest='ru')
        translated_texts.append(text.text)
    return translated_texts


def translate_frames_caption_by_open_ai(source_texts):
    translated_texts = []
    for source_text in source_texts:
        text = ts.translate_text(source_text, from_language='en', to_language='ru') #
        translated_texts.append(text)
    return translated_texts


def translate_frames_caption_by_model(source_texts):
    translated_texts = []
    for source_text in source_texts:
        inputs = translator_tokenizer(source_text, return_tensors="pt")
        output = translator_model.generate(**inputs, max_new_tokens=1000)
        out_text = translator_tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        translated_texts.append(out_text)
    return translated_texts
