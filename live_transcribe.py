import pyaudio
import numpy as np
import whisper
import threading
import queue
import sys
import time
import wave
from datetime import datetime

class LiveTranscriber:
    def __init__(self):
        self.model = whisper.load_model("base")
        self.audio_queue = queue.Queue()
        self.keep_running = True
        self.full_recording = []
        
        # Audio settings
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024 * 4
        self.audio = pyaudio.PyAudio()
        
        # Create timestamp for the recording file
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.wav_filename = f"recording_{self.timestamp}.wav"
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_data)
        self.full_recording.append(audio_data.copy())  # Store for complete transcription
        return (in_data, pyaudio.paContinue)
    
    def save_recording(self):
        print(f"\nSaving full recording to {self.wav_filename}...")
        full_audio = np.concatenate(self.full_recording)
        
        # Convert float32 to int16 for WAV file
        full_audio_int16 = (full_audio * 32767).astype(np.int16)
        
        with wave.open(self.wav_filename, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(self.RATE)
            wf.writeframes(full_audio_int16.tobytes())
    
    def transcribe_full_recording(self):
        print("\nTranscribing complete recording...")
        full_audio = np.concatenate(self.full_recording)
        result = self.model.transcribe(full_audio, language="en")
        
        # Save full transcription to file
        transcript_filename = f"transcript_{self.timestamp}.txt"
        with open(transcript_filename, 'w') as f:
            f.write(result["text"])
        print(f"\nFull transcription saved to {transcript_filename}")
        print("\nComplete transcription:")
        print(result["text"])
    
    def process_audio(self):
        while self.keep_running:
            # Collect ~2 seconds of audio data
            audio_data = []
            for _ in range(int(self.RATE * 2 / self.CHUNK)):
                if not self.keep_running:
                    break
                try:
                    data = self.audio_queue.get(timeout=2.0)
                    audio_data.append(data)
                except queue.Empty:
                    continue
            
            if len(audio_data) > 0:
                # Convert audio data to the format Whisper expects
                audio_data = np.concatenate(audio_data, axis=0)
                
                # Transcribe
                try:
                    result = self.model.transcribe(audio_data, language="en")
                    if result["text"].strip():
                        print(f"\r{result['text']}", flush=True)
                except Exception as e:
                    print(f"\rError transcribing: {e}", flush=True)
    
    def start(self):
        print("Starting live transcription... (Press Ctrl+C to stop)")
        print("Recording will be saved and transcribed in full when you stop.")
        
        # Start audio stream
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self.audio_callback
        )
        
        # Start processing thread
        process_thread = threading.Thread(target=self.process_audio)
        process_thread.start()
        
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping transcription...")
            self.keep_running = False
            stream.stop_stream()
            stream.close()
            self.audio.terminate()
            process_thread.join()
            
            # Save and transcribe the complete recording
            self.save_recording()
            self.transcribe_full_recording()

if __name__ == "__main__":
    transcriber = LiveTranscriber()
    transcriber.start()
