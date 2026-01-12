# python -m venv .venv

import tkinter as tk
from datetime import datetime
import requests
import random
import io
try:
    from PIL import Image, ImageTk
    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False

url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 48.4118, # 52.52,  # Berlin
    "longitude": 8.6628, # 13.41,
    "current": "temperature_2m,wind_speed_10m",
    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m"
}

response = requests.get(url, params=params)
data = response.json()
# print(data["current"])

city = "Horb am Neckar"
user_name = "Manfred"

# 30 nice greeting texts
greetings = [
    "Have a wonderful day ahead!",
    "Wishing you a bright and joyful day!",
    "Hope your day is full of little wins!",
    "May today bring you peace and progress.",
    "Sending a smile your way — enjoy the day!",
    "Make today amazing — one step at a time.",
    "Here's to a productive and pleasant day!",
    "Embrace the day with a grateful heart.",
    "Good vibes and smooth sailing today!",
    "Today is a great day to try something new.",
    "May your coffee be strong and your day be sweet.",
    "Shine bright — the day is yours!",
    "Take it easy and enjoy the small moments.",
    "Wishing you clarity and calm today.",
    "Let curiosity lead you to something wonderful.",
    "Keep smiling — good things are coming.",
    "May your day be productive and kind.",
    "Stay positive and make today count.",
    "A fresh day, a fresh start — enjoy it!",
    "Be kind to yourself today and always.",
    "Find a reason to celebrate today.",
    "May your day be filled with creativity.",
    "Small steps today, big progress tomorrow.",
    "Breathe deeply — you've got this.",
    "Hope today brings you clarity and energy.",
    "Make time for something that makes you smile.",
    "Wishing you balance, joy, and focus today.",
    "Take on the day with confidence and grace.",
    "Let today surprise you in a good way!",
    "May your day be gentle and rewarding."
]

# 30 random images (using picsum.photos seeded images)
image_urls = [f"https://picsum.photos/seed/img{i}/800/400" for i in range(1, 31)]

greeting_text = random.choice(greetings)
selected_image_url = random.choice(image_urls)

def update_time():
    now = datetime.now().strftime("%A, %d %B %Y %H:%M:%S")
    time_label.config(text=now)
    root.after(1000, update_time)

root = tk.Tk()
root.title(f"Welcome, {user_name}!")
root.geometry("800x480")
root.configure(bg="#e6f2ff")


# Attempt to load and display a random image (if Pillow available)
if _PIL_AVAILABLE:
    try:
        resp = requests.get(selected_image_url, timeout=6)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
        img.thumbnail((760, 240))
        photo = ImageTk.PhotoImage(img)
        img_label = tk.Label(root, image=photo, bg="#e6f2ff")
        img_label.image = photo # type: ignore
        img_label.pack(pady=8)
    except Exception as _e:
        print("Could not load image:", _e)
else:
    print("Pillow not available; image will not be displayed.")

salutation = tk.Label(root, text=f"Hello {user_name}!", font=("Arial", 16, "bold"), bg="#e6f2ff")
salutation.pack(pady=6)
greeting = tk.Label(root, text=greeting_text, font=("Arial", 16, "bold"), bg="#e6f2ff", fg="#006400")
greeting.pack(pady=6)

time_label = tk.Label(root, font=("Arial", 12), bg="#e6f2ff")
time_label.pack(pady=5)
update_time()

weather_temp = f"Current temperature in {city}: {data['current']['temperature_2m']}°C"
weather_wind_speed = f"Wind speed: {data['current']['wind_speed_10m']} km/h"
weather_label = tk.Label(root, text=weather_temp, font=("Arial", 12), bg="#e6f2ff")
weather_label.pack(pady=5)
weather_label = tk.Label(root, text=weather_wind_speed, font=("Arial", 12), bg="#e6f2ff")
weather_label.pack(pady=5)

root.mainloop()