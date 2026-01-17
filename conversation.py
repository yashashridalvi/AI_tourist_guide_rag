from rag import CityDatabase
from llm import RAGLLM
from speech_to_text import STT
from text_to_speech import TTS

# ------------------- City Aliases -------------------
CITY_ALIASES = {
    "मुंबई": "mumbai",
    "पुणे": "pune",
    "दिल्ली": "delhi",
    "जयपुर": "jaipur",
    "कोलकाता": "kolkata",
    "बेंगलुरु": "bengaluru",
    "bombay": "mumbai",
    "poona": "pune"
}

# ------------------- Number Normalization -------------------
NUMBER_MAP = {
    # Hindi
    "एक": "1", "दो": "2", "तीन": "3",
    # Marathi
    "एक": "1", "दोन": "2", "तीन": "3",
    # English
    "one": "1", "two": "2", "three": "3"
}

# ------------------- Multilingual Prompts -------------------
PROMPTS = {
    "english": {
        "welcome": "WELCOME TO AI TOURIST GUIDE",
        "select_lang": "Select language: 1 English, 2 Hindi, 3 Marathi",
        "select_mode": "Select mode: 1 Voice, 2 Text",
        "ask_city": "Which city do you want?",
        "ask_info": "Choose info: 1 Transport  2 Hotels  3 Places",
        "more_same": "Do you want more information for this city? yes or no",
        "another_city": "Do you want information for another city? yes or no",
        "city_not_found": "City not found.",
        "goodbye": "Goodbye!"
    },

    "hindi": {
        "welcome": "AI टूरिस्ट गाइड में आपका स्वागत है",
        "select_lang": "भाषा चुनें: 1 अंग्रेज़ी, 2 हिंदी, 3 मराठी",
        "select_mode": "मोड चुनें: 1 बोलकर, 2 टाइपिंग",
        "ask_city": "आप कौन सा शहर जानना चाहते हैं?",
        "ask_info": "जानकारी चुनें: 1 परिवहन  2 होटल  3 दर्शनीय स्थल",
        "more_same": "क्या आप इसी शहर की और जानकारी चाहते हैं? हाँ या नहीं",
        "another_city": "क्या आप किसी और शहर की जानकारी चाहते हैं? हाँ या नहीं",
        "city_not_found": "शहर नहीं मिला।",
        "goodbye": "धन्यवाद! अलविदा!"
    },

    "marathi": {
        "welcome": "AI टुरिस्ट गाईड मध्ये आपले स्वागत आहे",
        "select_lang": "भाषा निवडा: 1 इंग्रजी, 2 हिंदी, 3 मराठी",
        "select_mode": "मोड निवडा: 1 बोलून, 2 टायपिंग",
        "ask_city": "आपण कोणते शहर जाणून घ्यायचे आहे?",
        "ask_info": "माहिती निवडा: 1 वाहतूक  2 हॉटेल  3 पर्यटन स्थळे",
        "more_same": "या शहराची अजून माहिती हवी आहे का? हो किंवा नाही",
        "another_city": "दुसऱ्या शहराची माहिती हवी आहे का? हो किंवा नाही",
        "city_not_found": "शहर सापडले नाही.",
        "goodbye": "धन्यवाद! पुन्हा भेटूया!"
    }
}


# ------------------- Conversation Class -------------------
class Conversation:
    def __init__(self, db_path):
        self.db = CityDatabase(db_path)
        self.llm = RAGLLM()
        self.lang = "english"
        self.mode = "voice"
        self.tts = TTS(self.lang)
        self.stt = STT(self.lang)

    # -------- Normalize helpers --------
    def normalize_city(self, text):
        if not text:
            return None
        t = text.strip().lower()
        return CITY_ALIASES.get(t, t)

    def normalize_number(self, text):
        if not text:
            return ""
        t = text.strip().lower()
        return NUMBER_MAP.get(t, t)

    # -------- Ask Function --------
    def ask(self, msg):
        if self.mode == "voice":
            self.tts.speak(msg)
            res = self.stt.listen()
            return res if res else self.stt.typed("Type: ")
        else:
            print("BOT:", msg)
            return input("You: ")

    # -------- Start --------
    def start(self):
        print("\n🔊 BOT:", PROMPTS["english"]["welcome"])
        self.tts.speak("Select language: 1) English  2) Hindi  3) Marathi")
        c = input("Choose language: ")

        self.lang = {"2": "hindi", "3": "marathi"}.get(c, "english")
        self.tts.set_language(self.lang)
        self.stt.set_language(self.lang)

        self.tts.speak("Choose mode of conversation: 1) Voice  2) Text")
        self.mode = "voice" if input("Choose mode: ") == "1" else "text"

        self.tts.speak(PROMPTS[self.lang]["welcome"])
        self.loop()

    # -------- Main Loop --------
    def loop(self):
        while True:
            # Ask City
            city_name = self.ask(PROMPTS[self.lang]["ask_city"])
            city_key = self.normalize_city(city_name)

            city = self.db.find_city(city_key)
            if not city:
                self.tts.speak(PROMPTS[self.lang]["city_not_found"])
                continue

            # ---- Same city info loop ----
            while True:
                sec_raw = self.ask(PROMPTS[self.lang]["ask_info"])
                sec = self.normalize_number(sec_raw)

                if sec == "1":
                    topic = "transport"
                elif sec == "2":
                    topic = "hotels"
                elif sec == "3":
                    topic = "places"
                else:
                    self.tts.speak(PROMPTS[self.lang]["ask_info"])
                    continue

                # --- RAG + LLM Answer ---
                answer = self.llm.generate(city, topic, self.lang)

                if self.mode == "voice":
                    self.tts.speak(answer)
                else:
                    print("\nBOT:", answer)

                # Ask more for same city
                more = self.ask(PROMPTS[self.lang]["more_same"]).lower()
                if not more.startswith(("y", "हो", "हाँ")):
                    break

            # Ask another city
            again = self.ask(PROMPTS[self.lang]["another_city"]).lower()
            if not again.startswith(("y", "हो", "हाँ")):
                self.tts.speak(PROMPTS[self.lang]["goodbye"])
                break
