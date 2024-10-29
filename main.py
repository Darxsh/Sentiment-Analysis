import tkinter as tk
from tkinter import ttk, scrolledtext
from textblob import TextBlob
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

class SentimentAnalyzer:
    def __init__(self, master):
        self.master = master
        master.title("Comprehensive Sentiment Analysis Tool")
        master.geometry("1200x600")
        master.configure(bg='#f0f0f0')

        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.load_emojis()
        self.setup_face_detection()
        self.create_widgets()

    def load_emojis(self):
        emoji_dir = "emojis"
        self.emoji_images = {
            "Positive": ImageTk.PhotoImage(Image.open(os.path.join(emoji_dir, "positive.png")).resize((50, 50))),
            "Negative": ImageTk.PhotoImage(Image.open(os.path.join(emoji_dir, "negative.png")).resize((50, 50))),
            "Neutral": ImageTk.PhotoImage(Image.open(os.path.join(emoji_dir, "neutral.png")).resize((50, 50)))
        }

    def setup_face_detection(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.cap = cv2.VideoCapture(0)

    def create_widgets(self):
        title_label = ttk.Label(self.master, text="Chaitanya Patil Sentiment Analysis Tool with text and face analysis", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=10)

        left_frame = ttk.Frame(self.master)
        left_frame.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.BOTH, expand=True)

        input_label = ttk.Label(left_frame, text="Enter text for analysis:")
        input_label.pack(anchor='w')

        self.input_text = scrolledtext.ScrolledText(left_frame, height=5, wrap=tk.WORD)
        self.input_text.pack(fill=tk.X, expand=True)

        analyze_button = ttk.Button(left_frame, text="Analyze Text", command=self.analyze_text)
        analyze_button.pack(pady=10)

        result_label = ttk.Label(left_frame, text="Text Analysis Result:")
        result_label.pack(anchor='w')

        self.result_text = scrolledtext.ScrolledText(left_frame, height=3, wrap=tk.WORD, state='disabled')
        self.result_text.pack(fill=tk.X, expand=True)

        self.text_emoji_label = ttk.Label(left_frame)
        self.text_emoji_label.pack(pady=10)

        right_frame = ttk.Frame(self.master)
        right_frame.pack(side=tk.RIGHT, padx=20, pady=20, fill=tk.BOTH, expand=True)

        self.video_frame = ttk.Label(right_frame)
        self.video_frame.pack()

        video_result_label = ttk.Label(right_frame, text="Video Analysis Result:")
        video_result_label.pack(anchor='w')

        self.video_result_text = scrolledtext.ScrolledText(right_frame, height=3, wrap=tk.WORD, state='disabled')
        self.video_result_text.pack(fill=tk.X, expand=True)

        self.update_video()

    def analyze_text_sentiment(self, text):
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        if sentiment > 0:
            return 'Positive', sentiment
        elif sentiment < 0:
            return 'Negative', sentiment
        else:
            return 'Neutral', sentiment

    def analyze_text(self):
        text = self.input_text.get("1.0", tk.END).strip()
        if text:
            sentiment, score = self.analyze_text_sentiment(text)
            result = f"Sentiment: {sentiment}\nScore: {score:.2f}"
            self.result_text.config(state='normal')
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, result)
            self.result_text.config(state='disabled')
            self.text_emoji_label.config(image=self.emoji_images[sentiment])
        else:
            self.result_text.config(state='normal')
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, "Please enter some text to analyze.")
            self.result_text.config(state='disabled')
            self.text_emoji_label.config(image='')

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                roi = gray[y:y + h, x:x + w]
                roi = cv2.resize(roi, (64, 64))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = self.emotion_model.predict(roi)[0]
                emotion_idx = preds.argmax()
                emotion = self.emotions[emotion_idx]

                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                self.video_result_text.config(state='normal')
                self.video_result_text.delete("1.0", tk.END)
                self.video_result_text.insert(tk.END, f"Detected Emotion: {emotion}")
                self.video_result_text.config(state='disabled')

            img = Image.fromarray(image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)

        self.video_frame.after(10, self.update_video)

def main():
    root = tk.Tk()
    app = SentimentAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main()