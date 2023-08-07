import os
import speech_recognition as sr

# Initialize the recognizer
recognizer = sr.Recognizer()

# Load the PocketSphinx acoustic model and dictionary
acoustic_model_path = os.path.join(sr.__path__[0], "pocketsphinx-data", "en-US")
dictionary_path = os.path.join(acoustic_model_path, "cmudict-en-us.dict")
language_model_path = os.path.join(acoustic_model_path, "en-us.lm.bin")

# Recognize speech from microphone input
with sr.Microphone() as source:
    print("Say something:")
    audio = recognizer.listen(source)

try:
    recognized_text = recognizer.recognize_sphinx(audio, dictionary_path=dictionary_path, language_model_path=language_model_path)
    print("You said:", recognized_text)
except sr.UnknownValueError:
    print("Sorry, could not understand audio.")
except sr.RequestError as e:
    print(f"Error with the request: {e}")

