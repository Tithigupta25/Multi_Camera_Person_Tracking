import json
import threading
from datetime import datetime


log_lock = threading.Lock()


def log_event(log_file, event):
    event["ts"] = datetime.now().isoformat(timespec="seconds")

    with log_lock:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")