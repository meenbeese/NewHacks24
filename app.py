from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import numpy as np
import speech_recognition as sr
import whisper
import torch
from datetime import datetime, timedelta
from queue import Queue
from time import sleep

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app)

recognizer = sr.Recognizer()
audio_model = whisper.load_model("base")

data_queue = Queue()
phrase_time = None

def record_callback(_, audio: sr.AudioData):
    data = audio.get_raw_data()
    data_queue.put(data)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def connect():
    emit('message', {'text': 'Connected to server!'})

@socketio.on('start_transcription')
def start_transcription():
    global phrase_time
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        recognizer.listen_in_background(source, record_callback, phrase_time_limit=2)

        while True:
            try:
                now = datetime.utcnow()
                if not data_queue.empty():
                    phrase_complete = False
                    if phrase_time and now - phrase_time > timedelta(seconds=3):
                        phrase_complete = True
                    phrase_time = now

                    audio_data = b''.join(data_queue.queue)
                    data_queue.queue.clear()

                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                    text = result['text'].strip()

                    emit('transcription_update', {'text': text, 'phrase_complete': phrase_complete})

                else:
                    sleep(0.25)
            except KeyboardInterrupt:
                break

if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
