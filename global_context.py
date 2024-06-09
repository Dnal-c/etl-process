import torch
import pandas as pd
from transformers import GPT2TokenizerFast, ViTImageProcessor, AutoTokenizer, AutoModelForSeq2SeqLM, \
    VisionEncoderDecoderModel

from global_env import MODEL_CAPTION_PATH, DEVICE, MODEL_TRANSLATOR_PATH

image_captioner = VisionEncoderDecoderModel.from_pretrained(MODEL_CAPTION_PATH).to(DEVICE)
image_captioner_encoder_model = "microsoft/swin-base-patch4-window7-224-in22k"
image_captioner_decore_model = "gpt2"
image_tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_CAPTION_PATH)
image_processor = ViTImageProcessor.from_pretrained(MODEL_CAPTION_PATH)

translator_tokenizer = AutoTokenizer.from_pretrained(MODEL_TRANSLATOR_PATH)
translator_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_TRANSLATOR_PATH).to(DEVICE)

