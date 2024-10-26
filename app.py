from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import torch
import torch.nn as nn
import joblib

app = Flask(__name__)

class ScamClassifier(nn.Module):
    def __init__(self, input_dim):
        super(ScamClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

vectorizer = joblib.load('tfidf_vectorizer.pkl')
input_dim = len(vectorizer.get_feature_names_out())

model = ScamClassifier(input_dim=input_dim)
model.load_state_dict(torch.load('scam_classifier.pth'))
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/speech_to_text', methods=['POST'])
def speech_to_text():
    data = request.get_json()
    interval = int(data.get('interval', 5))
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio = recognizer.listen(source, timeout=interval)
        try:
            text = recognizer.recognize_google(audio)
            text_vector = vectorizer.transform([text]).toarray()
            text_tensor = torch.tensor(text_vector, dtype=torch.float32)
            prediction = model(text_tensor).item()
            classification = 'Scam' if prediction > 0.5 else 'Not Scam'
            return jsonify({'text': text, 'classification': classification})
        except sr.UnknownValueError:
            return jsonify({'error': "Sorry, I could not understand the audio."})
        except sr.RequestError:
            return jsonify({'error': "Could not request results; check your network connection."})

if __name__ == '__main__':
    app.run(debug=True)
