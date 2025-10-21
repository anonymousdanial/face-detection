import subprocess
import time

def continuously_curl():
    while True:
        try:
            result = subprocess.run(["curl", "localhost:8000/faces"], capture_output=True, text=True)
            print(result.stdout)
        except Exception as e:
            print(f"Error occurred: {e}")
        time.sleep(0.1)  # Delay between requests (adjust as needed)

if __name__ == "__main__":
    continuously_curl()
