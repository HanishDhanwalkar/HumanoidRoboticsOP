import speech_recognition as sr
import pyttsx3

class VoiceModel:
    def __init__(self, trigger_word):
        self.trigger_word = trigger_word

    def listen(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=10)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            print("sorry, could not recognise")
        except sr.RequestError as e:
            print("Error while connecting to Google Speech Recognition service; {0}".format(e))

    def speak(self, text):
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

    def listen_and_speak(self):
        while True:
            text = self.listen()
            if self.trigger_word in text:
                print("Found Trigger word ****************************")
                self.speak("Hello, how can I help you?")

while(1):
    print("speak now (say 'robo' to trigger the bot)")
    a1 = VoiceModel("Robo")

    a1.listen_and_speak()