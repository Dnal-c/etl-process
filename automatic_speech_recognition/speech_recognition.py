import torch
import whisperx

device = "cuda" if torch.cuda.is_available() else "cpu"

compute_type = "float32"
whisper_model = whisperx.load_model("large-v2", device=device, compute_type=compute_type)

def transcribe_video(input_video):
    batch_size = 32

    audio = whisperx.load_audio(input_video)
    result = whisper_model.transcribe(audio, batch_size=batch_size, language="ru")

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    segments = result['segments']
    texts = []
    for seg in segments:
        texts.append(seg['text'])
    return ' '.join(texts)

