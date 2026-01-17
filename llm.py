import ollama

class RAGLLM:
    def __init__(self, model="tinyllama"):
        self.model = model

    def generate(self, city_obj, topic, lang):
        # Build clean factual context
        city = city_obj["city"]
        state = city_obj["state"]

        if topic == "transport":
            data = city_obj["transport"]
            context = f"""
City: {city}, {state}
Transport:
Bus: {data['bus']}
Train: {data['train']}
Flight: {data['flight']}
"""
        elif topic == "hotels":
            h = city_obj["hotels"]
            context = f"""
City: {city}, {state}
Hotels:
Budget: {", ".join(h["budget"])}
Mid-range: {", ".join(h["mid_range"])}
Premium: {", ".join(h["premium"])}
"""
        else:
            places = ", ".join(city_obj["must_visit"])
            context = f"""
City: {city}, {state}
Must visit places: {places}
"""

        # Force correct language
        lang_instruction = {
            "english": "Answer in clear English.",
            "hindi": "उत्तर केवल शुद्ध हिंदी में दें। किसी अन्य भाषा का प्रयोग न करें।",
            "marathi": "उत्तर फक्त शुद्ध मराठीत द्या. इतर भाषा वापरू नका."
        }[lang]

        prompt = f"""
You are a tourist guide.
Use ONLY the information below.
Do NOT add extra facts.
{lang_instruction}

Information:
{context}

User wants: {topic}

Answer:
"""

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"].strip()
