import speech_recognition as sr
import whisper
import os
from transformers import pipeline
from deepface import DeepFace
import cv2
import threading
import logging
import platform
from transformers.utils import logging as hf_logging

logging.getLogger("transformers").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()
class PerceptionModule:

    def __init__(self):
        self.model = whisper.load_model("medium")
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 2.0

    def play_beep(self, sound_type="start"):
        """Generate a beep sound to indicate recording status.
        
        Args:
            sound_type: Type of sound to play - "start" for recording start, "end" for recording end
        """
        try:
            # On macOS
            if platform.system() == 'Darwin':
                if sound_type == "start":
                    os.system("afplay /System/Library/Sounds/Funk.aiff")
                else:  # end sound
                    os.system("afplay /System/Library/Sounds/Blow.aiff")
            # On Windows
            elif platform.system() == 'Windows':
                try:
                    import winsound
                    if sound_type == "start":
                        winsound.Beep(1000, 200)  # 1000 Hz for 200 ms
                    else:  # end sound
                        winsound.Beep(800, 200)   # 800 Hz for 200 ms
                except ImportError:
                    if sound_type == "start":
                        print("\n🎙️ RECORDING NOW 🎙️\n")
                    else:
                        print("\n🛑 RECORDING STOPPED 🛑\n")
            # On Linux or other systems
            else:
                # Use print as fallback
                if sound_type == "start":
                    print("\n🎙️ RECORDING NOW 🎙️\n")
                else:
                    print("\n🛑 RECORDING STOPPED 🛑\n")
        except Exception as e:
            print(f"Could not generate beep: {e}")
            if sound_type == "start":
                print("\n🎙️ RECORDING NOW 🎙️\n")
            else:
                print("\n🛑 RECORDING STOPPED 🛑\n")
            
    def play_beep_async(self, sound_type="start"):
        """Play beep sound in a separate thread to not block recording.
        
        Args:
            sound_type: Type of sound to play - "start" for recording start, "end" for recording end
        """
        beep_thread = threading.Thread(target=self.play_beep, args=(sound_type,))
        beep_thread.daemon = True  # Thread will exit when main program exits
        beep_thread.start()

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

    def percieve_combined(self):
        emotion_thread = self.EmotionDetectionThread()
        emotion_thread.start()
        audio = self.record_audio()
        emotion_thread.stop()
        emotion_thread.join()

        visual_emotions = emotion_thread.recorded
        transcript = self.speech_to_text_whisper(audio)
        emotional_tone = self.get_emotion_scores(transcript)
        bias_str = self.hf_to_deepface_emotion_map.get(emotional_tone, "neutral")
        dominant_emotion = self.most_common(visual_emotions, bias_str=bias_str, bias=0.7)

        return transcript, dominant_emotion

    def record_audio(self):
        with sr.Microphone() as source:
            # Do all preparation first
            self.recognizer.adjust_for_ambient_noise(source)
            
            # Play start beep in background thread at the same time recording starts
            self.play_beep_async("start")
            
            # Start recording immediately
            audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=None)
            
            # Play end beep to indicate recording is complete
            self.play_beep_async("end")
            
            return audio

    def get_emotion_scores(self, text):
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

            self.cap.release()
            cv2.destroyAllWindows()

        def stop(self):
            self._stop_event.set()

    def most_common(self, lst, bias_str, bias=.5):
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