import speech_recognition as sr

LANG_CODES = {
    "english": "en-IN",
    "hindi": "hi-IN",
    "marathi": "mr-IN"
}

class STT:
    def __init__(self, lang="english"):
        self.recognizer = sr.Recognizer()
        self.set_language(lang)

    def set_language(self, lang):
        self.lang = lang
        self.code = LANG_CODES.get(lang, "en-IN")

    def listen(self):
        try:
            with sr.Microphone() as src:
                self.recognizer.adjust_for_ambient_noise(src, duration=0.5)
                print("🎤 Listening...")
                audio = self.recognizer.listen(src, timeout=6)
            text = self.recognizer.recognize_google(audio, language=self.code)
            print("🗣️ USER:", text)
            return text.lower()
        except:
            return None

    def typed(self, prompt):
        ans = input(prompt).strip().lower()
        print("🗣️ USER:", ans)
        return ans
