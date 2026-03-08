import time
last_activity = 0.0

def speak():
    global last_activity
    last_activity = time.time()

if __name__ == "__main__":
    def main():
        global last_activity
        last_activity = -100.0
        speak()
        print(f"last_activity in main: {last_activity}")
    main()
