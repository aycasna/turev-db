import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk
import sounddevice as sd
import wavio
import numpy as np
import librosa
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# Load the models
audio_model = tf.keras.models.load_model('Models/audio_best_model.keras')
spectrogram_model = tf.keras.models.load_model('Models/spectrogram_best_model.keras')

# Define emotion labels
emotion_labels = {0: 'Angry', 1: 'Calm', 2: 'Happy', 3: 'Sad'}

# Record audio
def record_audio(filename, duration=3, fs=44100):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    wavio.write(filename, recording, fs, sampwidth=2)
    print("Recording finished")

# Predict emotion from MFCC features
def predict_emotion_from_audio_model(audio_file_path, model):
    y, sr = librosa.load(audio_file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.mean(mfcc.T, axis=0)
    input_data = np.expand_dims(mfcc, axis=0)
    predictions = model.predict(input_data)
    predicted_class = np.argmax(predictions[0])
    return emotion_labels[predicted_class], predictions[0][predicted_class]

# Preprocess audio for spectrogram model
def preprocess_audio(audio_path, sr=22050, duration=2, n_mels=128, image_size=(600, 400)):
    audio, _ = librosa.load(audio_path, sr=sr, duration=duration, mono=True)
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram = cv2.resize(spectrogram, dsize=image_size[::-1], interpolation=cv2.INTER_CUBIC)
    spectrogram_rgb = np.stack((spectrogram,) * 3, axis=-1)
    return spectrogram_rgb[:, :, :3]

# Normalize spectrogram
def normalize_spectrogram(spectrogram):
    normalized_spectrogram = ((spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram)) * 255).astype(np.uint8)
    return normalized_spectrogram

# Predict emotion from spectrogram
def predict_emotion_from_spectrogram_model(audio_path, model):
    spectrogram = preprocess_audio(audio_path)
    spectrogram = normalize_spectrogram(spectrogram)
    predictions = model.predict(np.expand_dims(spectrogram, axis=0))
    predicted_label = emotion_labels[np.argmax(predictions)]
    return predicted_label, np.max(predictions), spectrogram

# Tkinter GUI
class EmotionRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Recognizer")

        # Configure styles
        style = ttk.Style()
        style.configure('TButton', font=('Helvetica', 16), padding=10)
        style.configure('TLabel', font=('Helvetica', 16), padding=10)
        
        # Configure the main frame
        main_frame = ttk.Frame(root, padding=20)
        main_frame.pack(expand=True, fill=tk.BOTH)

        # Add title label
        title_label = ttk.Label(main_frame, text="Emotion Recognizer", font=("Helvetica", 24, "bold"))
        title_label.pack(pady=(0, 20))

        # Add record button
        self.record_button = ttk.Button(main_frame, text="Record Voice", command=self.record_voice, style='TButton')
        self.record_button.pack(pady=20)

        # Add result label
        self.result_label = ttk.Label(main_frame, text="", style='TLabel')
        self.result_label.pack(pady=20)

        # Add show spectrogram button
        self.show_spectrogram_button = ttk.Button(main_frame, text="Show Spectrogram", command=self.show_spectrogram, style='TButton')
        self.show_spectrogram_button.pack(pady=20)
        self.show_spectrogram_button.state(['disabled'])

        # Add save button
        self.save_button = ttk.Button(main_frame, text="Save Result", command=self.save_result, style='TButton')
        self.save_button.pack(pady=20)
        self.save_button.state(['disabled'])

        # Initialize spectrogram
        self.spectrogram = None
    
    def record_voice(self):
        self.result_label.config(text="Recording...")
        audio_path = "user_voice.wav"
        record_audio(audio_path)
        self.result_label.config(text="Recording finished. Predicting...")

        try:
            emotion_audio_model, confidence_audio_model = predict_emotion_from_audio_model(audio_path, audio_model)
            emotion_spectrogram_model, confidence_spectrogram_model, self.spectrogram = predict_emotion_from_spectrogram_model(audio_path, spectrogram_model)
            
            result_text = (
                f"Audio Model Prediction: {emotion_audio_model} (Confidence: {confidence_audio_model * 100:.2f}%)\n"
                f"Spectrogram Model Prediction: {emotion_spectrogram_model} (Confidence: {confidence_spectrogram_model * 100:.2f}%)"
            )
            
            self.result_label.config(text=result_text)
            self.save_button.state(['!disabled'])
            self.show_spectrogram_button.state(['!disabled'])
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.result_label.config(text="")
            self.save_button.state(['disabled'])
            self.show_spectrogram_button.state(['disabled'])

    def save_result(self):
        result_text = self.result_label.cget("text")
        if result_text and self.spectrogram is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
            if file_path:
                with open(file_path, 'w') as file:
                    file.write(result_text)
                
                # Save the spectrogram image
                spectrogram_image_path = file_path.replace('.txt', '_spectrogram.png')
                plt.imsave(spectrogram_image_path, self.spectrogram)
                
                messagebox.showinfo("Saved", "Result and spectrogram saved successfully!")
        else:
            messagebox.showwarning("Warning", "No result or spectrogram to save!")

    def show_spectrogram(self):
        if self.spectrogram is not None:
            plt.imshow(self.spectrogram)
            plt.axis('off')  # Turn off axis labels
            plt.show()
        else:
            messagebox.showwarning("Warning", "No spectrogram to display!")

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionRecognizerApp(root)
    root.mainloop()
