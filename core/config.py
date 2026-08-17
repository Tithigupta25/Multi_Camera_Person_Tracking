import json
import os


CONFIG_FILE = "config/config.json"
CLASSES_FILE = "config/classes.txt"


def load_config():
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"{CONFIG_FILE} not found.")

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_classes():
    if not os.path.exists(CLASSES_FILE):
        raise FileNotFoundError(f"{CLASSES_FILE} not found.")

    with open(CLASSES_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]