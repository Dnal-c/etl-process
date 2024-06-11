import whisperx

from global_env import DEVICE, WHISPER_COMPUTE_TYPE

whisper_model = whisperx.load_model("large-v2", device=DEVICE, compute_type=WHISPER_COMPUTE_TYPE)


def recognize_speech(input_video):
    batch_size = 32

    audio = whisperx.load_audio(input_video)
    result = whisper_model.transcribe(audio, batch_size=batch_size, language="ru")

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
    result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE,
                            return_char_alignments=False)

    segments = result['segments']
    texts = []
    for seg in segments:
        texts.append(seg['text'])
    return ' '.join(texts)
