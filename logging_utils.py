from datetime import datetime

def log_event(component: str, message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{component}] {message}"
    print(log_entry)
    with open("events.log", "a", encoding="utf-8") as f:  # <-- FIXED encoding here
        f.write(log_entry + "\n")
