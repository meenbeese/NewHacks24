import threading
from datetime import datetime, timedelta
from queue import Queue
from time import sleep

import numpy as np
import speech_recognition as sr
import torch
import whisper
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

recognizer = sr.Recognizer()
audio_model = whisper.load_model("base")

data_queue = Queue()
phrase_time = None
transcription_active = False


def record_callback(_, audio: sr.AudioData):
    if transcription_active:
        data = audio.get_raw_data()
        data_queue.put(data)


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def connect():
    print("Client connected")
    emit('message', {'text': 'Connected to server!'})


@socketio.on('disconnect')
def disconnect():
    global transcription_active
    print("Client disconnected")
    transcription_active = False


def transcription_loop():
    global phrase_time, transcription_active

    while transcription_active:
        try:
            now = datetime.utcnow()
            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=3):
                    phrase_complete = True
                phrase_time = now

                audio_data = []
                while not data_queue.empty():
                    audio_data.append(data_queue.get())
                audio_data = b''.join(audio_data)

                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                print(f"Transcribed text: {text}")
                socketio.emit('transcription_update', {'text': text, 'phrase_complete': phrase_complete})

            else:
                sleep(0.25)
        except Exception as e:
            print(f"Error in transcription loop: {str(e)}")
            continue


@socketio.on('start_transcription')
def start_transcription(data):
    global transcription_active
    transcription_active = True

    with sr.Microphone() as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source)
        recognizer.listen_in_background(source, record_callback, phrase_time_limit=2)

        threading.Thread(target=transcription_loop, daemon=True).start()

        print(f"Started transcription with interval: {data.get('interval', 'not specified')}")


@socketio.on('stop_transcription')
def stop_transcription():
    global transcription_active
    print("Stopping transcription")
    transcription_active = False


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
