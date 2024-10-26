import speech_recognition as sr

recognizer = sr.Recognizer()

with sr.Microphone() as source:
    print("Please say something:")
    audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google_cloud(audio)
        print("You said: " + text)
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
    except sr.RequestError:
        print("Could not request results; check your network connection.")