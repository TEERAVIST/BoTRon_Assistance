import speech_recognition as sr

r = sr.Recognizer()

with sr.Microphone() as source:
    print("Please")
    audio = r.listen(source)
    text = r.recognize_google(audio, language='th-Th')
    print("you",text)
