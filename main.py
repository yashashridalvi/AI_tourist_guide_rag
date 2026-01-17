from conversation import Conversation

def main():
    conv = Conversation(db_path="database.json")
    conv.start()

if __name__ == "__main__":
    main()