from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, GPT2TokenizerFast, ViTImageProcessor

'''
Загрузка модели для перевода текста 
'''
translate_model_name = "Helsinki-NLP/opus-mt-en-ru"
translate_tokenizer = AutoTokenizer.from_pretrained(translate_model_name)
translate_model = AutoModelForSeq2SeqLM.from_pretrained(translate_model_name)

translate_tokenizer.save_pretrained('./translator')
translate_model.save_pretrained('./translator')

'''
Загрузка модели для Image Captioning
'''
# Используя pipeline api
image_captioner = pipeline("image-to-text", model="Abdou/vit-swin-base-224-gpt2-image-captioning")
image_captioner.save_pretrained('./captioning')
# encoder_model = "WinKawaks/vit-small-patch16-224"
# encoder_model = "google/vit-base-patch16-224"
# encoder_model = "google/vit-base-patch16-224-in21k"
image_captioning_encoder_model = "microsoft/swin-base-patch4-window7-224-in22k"

# модель декодера, которая обрабатывает элементы изображения и генерирует текст подписи
# decoder_model = "bert-base-uncased"
# decoder_model = "prajjwal1/bert-tiny"
image_captioning_decoder_model = "gpt2"

# Инициализируем токенайзер
# tokenizer = AutoTokenizer.from_pretrained(decoder_model)
image_captioning_tokenizer = GPT2TokenizerFast.from_pretrained(image_captioning_decoder_model)
image_captioning_tokenizer.save_pretrained('./captioning')
# tokenizer = BertTokenizerFast.from_pretrained(decoder_model)
# Загружаем обработчик изображений
image_captioning_processor = ViTImageProcessor.from_pretrained(image_captioning_encoder_model)
image_captioning_processor.save_pretrained('./captioning')
