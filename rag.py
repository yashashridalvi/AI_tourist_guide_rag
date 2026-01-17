import json
from pathlib import Path
from difflib import get_close_matches

class CityDatabase:
    def __init__(self, path="database.json"):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError("database.json not found")

        with p.open("r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.map = {c["city"].lower(): c for c in self.data}

    def find_city(self, name):
        if not name:
            return None

        name = name.lower().strip()

        if name in self.map:
            return self.map[name]

        matches = get_close_matches(name, self.map.keys(), n=1, cutoff=0.6)
        return self.map[matches[0]] if matches else None
