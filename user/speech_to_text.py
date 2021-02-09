import time

import deepspeech
import keyboard
import numpy as np
import pyaudio

# DeepSpeech parameters
MODEL_FILE_PATH = \
    r"C:\Users\alexander.zap\PycharmProjects\task-chatbot\resources\deepspeech-model\deepspeech-0.8.1-models.pbmm"
SCORER_FILE_PATH = \
    r"C:\Users\alexander.zap\PycharmProjects\task-chatbot\resources\deepspeech-model\deepspeech-0.8.1-models.scorer"
BEAM_WIDTH = 500
LM_ALPHA = 0.75
LM_BETA = 1.85

# Make DeepSpeech Model
model = deepspeech.Model(MODEL_FILE_PATH)
model.enableExternalScorer(SCORER_FILE_PATH)
model.setScorerAlphaBeta(LM_ALPHA, LM_BETA)
model.setBeamWidth(BEAM_WIDTH)

# PyAudio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_SIZE = 1024

transcribed_text = ''


def speech_to_text():
    global transcribed_text

    # Create a Streaming session
    ds_stream = model.createStream()
    audio = pyaudio.PyAudio()

    # Encapsulate DeepSpeech audio feeding into a callback for PyAudio
    def process_audio(in_data, frame_count, time_info, status):
        global transcribed_text
        data16 = np.frombuffer(in_data, dtype=np.int16)
        ds_stream.feedAudioContent(data16)
        intermediate_text = ds_stream.intermediateDecode()
        if intermediate_text != transcribed_text:
            transcribed_text = intermediate_text
        return in_data, pyaudio.paContinue

    # Feed audio to DeepSpeech in a callback to PyAudio
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
        stream_callback=process_audio
    )

    print('Please start speaking, when done press "#" ...')
    stream.start_stream()

    keyboard.on_press_key("#", lambda _: end_stream())

    def end_stream():
        global transcribed_text
        # PyAudio
        stream.stop_stream()

    while stream.is_active():
        time.sleep(0.2)

    return transcribed_text
