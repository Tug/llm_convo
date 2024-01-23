import io
import os
import tempfile
import queue
import functools
import logging

from pydub import AudioSegment
import speech_recognition as sr
#import whisper
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf

@functools.cache
def get_whisper_model(size: str = "large"):
    logging.info(f"Loading whisper {size}")
    model = WhisperForConditionalGeneration.from_pretrained("pierreguillou/whisper-medium-french")
    return model
    # return whisper.load_model(size)

@functools.cache
def get_whisper_processor():
    processor = WhisperProcessor.from_pretrained("pierreguillou/whisper-medium-french")#, language="french")
    return processor

class WhisperMicrophone:
    def __init__(self):
        self.audio_model = get_whisper_model()
        self.processor = get_whisper_processor()
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 500
        self.recognizer.pause_threshold = 0.8
        self.recognizer.dynamic_energy_threshold = False

    def get_transcription(self) -> str:
        with sr.Microphone(sample_rate=16000) as source:
            logging.info("Waiting for mic...")
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = os.path.join(tmp, "mic.wav")
                audio = self.recognizer.listen(source)
                data, sample_rate = sf.read(io.BytesIO(audio.get_wav_data()))
                print(data)
                #audio_clip = AudioSegment.from_file(data)
                # audio_clip.export(tmp_path, format="wav")
                input_features = self.processor(
                    data, sampling_rate=16000, return_tensors="pt"
                ).input_features
                predicted_ids = self.audio_model.generate(input_features)#, language='french')
                # Decode token ids to text
                transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
                # result = self.audio_model.transcribe(tmp_path, language="english")
            # predicted_text = result["text"]
            predicted_text = transcription[0]
        return predicted_text


class _TwilioSource(sr.AudioSource):
    def __init__(self, stream):
        self.stream = stream
        self.CHUNK = 1024
        self.SAMPLE_RATE = 8000
        self.SAMPLE_WIDTH = 2

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class _QueueStream:
    def __init__(self):
        self.q = queue.Queue(maxsize=-1)

    def read(self, chunk: int) -> bytes:
        return self.q.get()

    def write(self, chunk: bytes):
        self.q.put(chunk)


class WhisperTwilioStream:
    def __init__(self):
        self.audio_model = get_whisper_model()
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.pause_threshold = 1.5
        self.recognizer.dynamic_energy_threshold = False
        self.stream = None

    def get_transcription(self) -> str:
        self.stream = _QueueStream()
        with _TwilioSource(self.stream) as source:
            logging.info("Waiting for twilio caller...")
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = os.path.join(tmp, "mic.wav")
                audio = self.recognizer.listen(source)
                data = io.BytesIO(audio.get_wav_data())
                audio_clip = AudioSegment.from_file(data)
                audio_clip.export(tmp_path, format="wav")
                result = self.audio_model.transcribe(tmp_path, language="english")
        predicted_text = result["text"]
        self.stream = None
        return predicted_text
