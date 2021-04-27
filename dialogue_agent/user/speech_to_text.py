import deepspeech
import numpy as np
import pyaudio


class SpeechToText:

    def __init__(self, model_file_path, scorer_file_path, beam_width, lm_alpha, lm_beta):
        # Make DeepSpeech Model
        self.model = deepspeech.Model(model_file_path)
        self.model.enableExternalScorer(scorer_file_path)
        self.model.setScorerAlphaBeta(lm_alpha, lm_beta)
        self.model.setBeamWidth(beam_width)

        # PyAudio parameters
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk_size = 1024

        self.transcribed_text = ''

        self.audio = pyaudio.PyAudio()
        self.audio_stream = None
        self.deepspeech_stream = None

    def reset(self):
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()

        if self.audio:
            self.audio.terminate()

        if self.deepspeech_stream:
            self.deepspeech_stream.finishStream()

        self.audio = pyaudio.PyAudio()
        self.deepspeech_stream = None
        self.audio_stream = None

    def start_speech_to_text(self):
        # Create a Streaming session
        self.deepspeech_stream = self.model.createStream()

        # Encapsulate DeepSpeech audio feeding into a callback for PyAudio
        def process_audio(in_data, frame_count, time_info, status):
            data16 = np.frombuffer(in_data, dtype=np.int16)
            self.deepspeech_stream.feedAudioContent(data16)
            intermediate_text = self.deepspeech_stream.intermediateDecode()
            self.transcribed_text = intermediate_text
            return in_data, pyaudio.paContinue

        # Feed audio to DeepSpeech in a callback to PyAudio
        self.audio_stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=process_audio
        )

        self.audio_stream.start_stream()
