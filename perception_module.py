import speech_recognition as sr
import whisper
import os
from transformers import pipeline
from deepface import DeepFace
import cv2
import threading

class PerceptionModule:

    def __init__(self):
        self.model = whisper.load_model("medium")
        self.recognizer = sr.Recognizer()

    def speech_to_text_whisper(self, audio):
        """
        Converts recorded audio into text using OpenAI's Whisper model.
        
        - Loads the Whisper model (`medium`).
        - Saves the recorded audio as a temporary WAV file.
        - Uses Whisper to transcribe the saved audio.
        - Deletes the temporary file after transcription.
        
        Args:
            audio (sr.AudioData): The recorded audio data.
        
        Returns:
            str: The transcribed text.
        """
        
        temp_filename = "temp.wav"
        with open(temp_filename, "wb") as f:
            f.write(audio.get_wav_data())
        
        result = self.model.transcribe(temp_filename)
        
        os.remove(temp_filename)
        return result["text"]

    def record_audio(self):
        cap = cv2.VideoCapture(0)
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("Recording... Speak now!")
            audio = self.recognizer.listen(source)
            print("Recording complete.")
            return audio

    def get_emotion_scores(text):
        classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
        return max(classifier(text)[0], key=lambda x: x['score'])["label"]

    class EmotionDetectionThread(threading.Thread):
        def __init__(self):
            super().__init__()
            self._stop_event = threading.Event()
            self.cap = None
            self.recorded = []

        def recognize_emotion(self, frame):
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                return result[0]['dominant_emotion']
            except Exception as e:
                print(f"Emotion detection failed: {e}")
                return "unknown"

        def run(self):
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Webcam not accessible.")
                return

            while not self._stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break

                # Detect emotion
                emotion = self.recognize_emotion(frame)
                self.recorded.append(emotion)
                # Display emotion label
                cv2.putText(frame, f'Emotion: {emotion}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Webcam - Emotion Detection', frame)

            self.cap.release()
            cv2.destroyAllWindows()

        def stop(self):
            self._stop_event.set()

    def most_common(lst, bias_str, bias=.5):
        return max(set(lst), key=lambda x: lst.count(x)* (1+ (bias if x is bias_str else 0)))

    hf_to_deepface_emotion_map = {
        'anger': 'angry',
        'disgust': 'disgust',
        'fear': 'fear',
        'joy': 'happy',
        'neutral': 'neutral',
        'sadness': 'sad',
        'surprise': 'surprise'
    }

    def percieve(self):
        emotion_thread = self.EmotionDetectionThread()
        emotion_thread.start()
        audio = self.record_audio()
        emotion_thread.stop()
        emotion_thread.join() 
        
        visual_emotions = emotion_thread.recorded
        try:
            transcript = self.speech_to_text_whisper(audio)
        except Exception as e:
            print("Whisper failed:", e)

        emotional_tone = self.get_emotion_scores(transcript)
        print("visual emotions:", visual_emotions)
        print("Emotional tone:", emotional_tone)
        print("Most likely emotion (combined):", PerceptionModule.most_common(visual_emotions, PerceptionModule.hf_to_deepface_emotion_map[emotional_tone], bias = .7))
        print("Transcript:", transcript)

if __name__ == "__main__":
    PerceptionModule().percieve()