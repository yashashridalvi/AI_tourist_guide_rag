import io
import pygame
from gtts import gTTS

LANG_MAP = {
    "english": "en",
    "hindi": "hi",
    "marathi": "mr"
}

pygame.mixer.init()

class TTS:
    def __init__(self, lang="english"):
        self.set_language(lang)

    def set_language(self, lang):
        self.lang = lang if lang in LANG_MAP else "english"
        self.code = LANG_MAP[self.lang]

    def speak(self, text):
        if not text:
            return

        print(f"\n🔊 BOT: {text}")

        try:
            mp3 = io.BytesIO()
            gTTS(text=text, lang=self.code).write_to_fp(mp3)
            mp3.seek(0)

            pygame.mixer.music.load(mp3)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

        except Exception as e:
            print("[TTS ERROR]", e)

