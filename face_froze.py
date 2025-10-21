import logging
from pathlib import Path
import face_recognition
import cv2
import numpy as np
import json
import tempfile
import shutil
import os
import random
from recommendations2 import recommender
from emotions import EmotionDetector
import Danial
from flask import Flask, request, jsonify

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

database = Danial.customers()
# Make DB directory configurable via environment for easier testing/deployment
DB_DIR = Path(os.getenv("FACES_DIR", "Faces"))  # Directory to store/load files
DB_FILE_NAME = os.getenv("FACES_FILE", "face_encodings.json")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "40.0"))
class FaceRecognizer:
    def __init__(self):
        self.video_capture = None
        self.known_face_encodings = []
        self.known_face_names = []
        self.db_last_modified = 0
        self.emotion_detector = EmotionDetector()  # Initialize emotion detector
        self._initialize_camera()
        self._load_face_database()  # Only load once at startup
    
    def _initialize_camera(self):
        """Initialize camera once and keep it open"""
        if self.video_capture is None:
            self.video_capture = cv2.VideoCapture(0)
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # Warm up the camera
            for _ in range(5):
                self.video_capture.read()
    
    def _get_db_modified_time(self):
        """Get the modification time of the face encodings database file"""
        try:
            database_file = DB_DIR / DB_FILE_NAME
            if not database_file.exists():
                return 0
            return database_file.stat().st_mtime
        except (ValueError, OSError):
            return 0
    
    def _load_face_database(self):
        """Load face database once at startup"""
        database_file = DB_DIR / DB_FILE_NAME

        if not DB_DIR.exists():
            DB_DIR.mkdir(parents=True, exist_ok=True)
            self.known_face_encodings = []
            self.known_face_names = []
            return

        if not database_file.exists():
            # initialize empty json database
            with open(database_file, 'w', encoding='utf-8') as fh:
                json.dump({}, fh)
            self.known_face_encodings = []
            self.known_face_names = []
            return

        try:
            logger.info("Loading face database (json)...")
            with open(database_file, 'r', encoding='utf-8') as fh:
                data = json.load(fh)

            # data is dict: face_id -> list of encodings (each encoding is a list)
            encodings = []
            names = []
            for face_id, enc_list in data.items():
                # ensure list
                if not isinstance(enc_list, list):
                    continue
                for enc in enc_list:
                    try:
                        arr = np.array(enc, dtype=np.float64)
                        encodings.append(arr)
                        names.append(face_id)
                    except Exception:
                        logger.warning(f"Invalid encoding for {face_id}, skipping")

            self.known_face_encodings = encodings
            self.known_face_names = names
            logger.info(f"Loaded {len(self.known_face_names)} known face encodings (from {len(data)} ids)")
        except Exception as e:
            logger.warning(f"Failed to load face database: {e}")
            self.known_face_encodings = []
            self.known_face_names = []
    
    def save_new_customer(self, face_encoding):
        """Save new customer and update in-memory database"""
        # Ensure DB_DIR exists
        DB_DIR.mkdir(parents=True, exist_ok=True)
        database_file = DB_DIR / DB_FILE_NAME

        try:
            # Generate a random unique 16-digit integer string for the face id
            import secrets

            def gen_faceid() -> str:
                # 16-digit decimal integer
                return ''.join(str(secrets.randbelow(10)) for _ in range(16))

            # Ensure uniqueness against DB via database.fetch_customer_by_faceid
            attempts = 0
            face_id = None
            while attempts < 20:
                attempts += 1
                candidate = gen_faceid()
                try:
                    existing = database.fetch_customer_by_faceid(candidate)
                except Exception:
                    existing = None
                if existing:
                    continue
                face_id = candidate
                break

            if face_id is None:
                raise RuntimeError("Unable to generate unique face id")

            # Load existing JSON db
            if database_file.exists():
                try:
                    with open(database_file, 'r', encoding='utf-8') as fh:
                        data = json.load(fh)
                except Exception:
                    data = {}
            else:
                data = {}

            # Add new face id with one encoding (store as list)
            data[face_id] = [face_encoding.tolist()]

            # Write atomically
            fd, tmp_path = tempfile.mkstemp(prefix='faces-', suffix='.json', dir=str(DB_DIR))
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as tmpf:
                    json.dump(data, tmpf, ensure_ascii=False)
                shutil.move(tmp_path, str(database_file))
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

            # Insert into DB with explicit CustomerFaceID; add_customer will compute the numeric CustomerID
            inserted_id = database.add_customer(CustomerID=None, CustomerFaceID=face_id, Name=None)

            new_name = face_id

            # Update in-memory database
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(new_name)

            logger.info(f'New customer saved: {new_name} (ID={inserted_id})')
            return new_name
        except Exception as e:
            logger.exception('Failed to save new customer')
            raise

    def add_encoding_for_face(self, face_id, face_encoding):
        """Append an encoding to an existing face_id in the JSON DB and update memory"""
        DB_DIR.mkdir(parents=True, exist_ok=True)
        database_file = DB_DIR / DB_FILE_NAME
        try:
            if database_file.exists():
                try:
                    with open(database_file, 'r', encoding='utf-8') as fh:
                        data = json.load(fh)
                except Exception:
                    data = {}
            else:
                data = {}

            enc_list = data.get(face_id, [])
            enc_list.append(face_encoding.tolist())
            data[face_id] = enc_list

            # atomic write
            fd, tmp_path = tempfile.mkstemp(prefix='faces-', suffix='.json', dir=str(DB_DIR))
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as tmpf:
                    json.dump(data, tmpf, ensure_ascii=False)
                shutil.move(tmp_path, str(database_file))
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

            # Update in-memory
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(face_id)
            logger.info(f'Appended new encoding for face {face_id} (now {len(enc_list)} encodings)')
            return True
        except Exception:
            logger.exception(f'Failed to append encoding for {face_id}')
            return False
    
    def recognize_faces(self):
        """Fast face recognition with emotion detection"""
        # No longer reloads database every call
        # Capture frame
        ret, frame = self.video_capture.read()
        if not ret:
            logger.error("Failed to capture frame from camera.")
            return []
        
        # Process frame
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        
        if len(face_locations) == 0:
            return []
        
        # Get face encodings
        try:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=1)
        except Exception as e:
            logger.exception(f"Error encoding face: {e}")
            return []
        
        face_names_with_emotions = []
        
        for i, face_encoding in enumerate(face_encodings):
            # Get face location for this encoding
            top, right, bottom, left = face_locations[i]
            
            # Extract face region for emotion detection (from original frame)
            face_roi = frame[top*2:bottom*2, left*2:right*2]
            
            # Predict emotion if face region is valid
            emotion = "Unknown"
            emotion_confidence = 0.0
            if face_roi.size > 0 and min(face_roi.shape[:2]) > 30:
                try:
                    emotion, emotion_confidence = self.emotion_detector.predict_emotion(face_roi)
                except Exception as e:
                    logger.exception(f"Error detecting emotion: {e}")
                    emotion = "Unknown"
                    emotion_confidence = 0.0
            
            # Recognize the person
            if len(self.known_face_encodings) == 0:
                # Save the first detected face as a new customer
                new_name = self.save_new_customer(face_encoding)
                person_name = f"{new_name} (added)"
            else:
                # Compare against every stored encoding (including multiple arrays per person)
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = int(np.argmin(face_distances))
                best_match_distance = float(face_distances[best_match_index])
                confidence = (1 - best_match_distance) * 100

                # Heuristics:
                # - confidence < 40% -> create new entry
                # - 55% <= confidence < 85% -> append this encoding for the matched face id
                # - confidence >= 85% -> recognized
                if confidence < CONFIDENCE_THRESHOLD:
                    # create new entry
                    new_name = self.save_new_customer(face_encoding)
                    person_name = f"{new_name} (new)"
                elif 55.0 <= confidence < 85.0:
                    # Append a second numpy array for that face (use the matched face id)
                    raw_name = self.known_face_names[best_match_index]
                    try:
                        appended = self.add_encoding_for_face(raw_name, face_encoding)
                        if appended:
                            person_name = f"{raw_name} (updated, {confidence:.1f}%)"
                        else:
                            person_name = f"{raw_name} ({confidence:.1f}%)"
                    except Exception:
                        person_name = f"{raw_name} ({confidence:.1f}%)"
                else:
                    # high confidence: recognized
                    raw_name = self.known_face_names[best_match_index]
                    display_name = "Unknown"
                    try:
                        if isinstance(raw_name, str) and raw_name.startswith('customer_'):
                            cid_part = raw_name.split('_', 1)[1]
                            cid = int(cid_part)
                            row = database.fetch_customer_by_id(cid)
                            if row:
                                display_name = getattr(row, 'Name', None) or (row[2] if len(row) > 2 else str(row))
                    except Exception as e:
                        logger.exception(f"Error fetching name from DB for {raw_name}: {e}")

                    person_name = f"{display_name} ({confidence:.1f}%)"
            
            # Combine person name with emotion
            if emotion != "Unknown" and emotion_confidence > 0.6:
                full_name = f"{person_name} - {emotion} ({emotion_confidence:.0%})"
            else:
                full_name = person_name
            
            face_names_with_emotions.append(full_name)
        
        # Draw rectangles and labels with emotions
        for (top, right, bottom, left), name_with_emotion in zip(face_locations, face_names_with_emotions):
            # Scale back up face locations since the frame we detected in was scaled to 1/2 size
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            
            # Choose color based on emotion
            color = self._get_emotion_color(name_with_emotion)
            
            # Draw face rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Calculate text size to create proper background
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            thickness = 1
            
            # Split text into lines if too long
            text_lines = self._split_text_for_display(name_with_emotion, max_width=right-left-10)
            
            # Calculate total height needed for all text lines
            line_height = cv2.getTextSize("Ay", font, font_scale, thickness)[0][1] + 5
            total_height = len(text_lines) * line_height + 10
            
            # Draw background rectangle for text
            cv2.rectangle(frame, (left, bottom), (right, bottom + total_height), color, cv2.FILLED)
            
            # Draw each line of text
            for i, line in enumerate(text_lines):
                y_position = bottom + (i + 1) * line_height
                cv2.putText(frame, line, (left + 6, y_position), font, font_scale, (255, 255, 255), thickness)
            # Optionally show the video for local debugging
            # cv2.imshow('Video', frame)
            # cv2.waitKey(1)
        
        # print(f"Current seen faces: {face_names_with_emotions}")
        return face_names_with_emotions
    
    def _get_emotion_color(self, name_with_emotion):
        """Get color based on detected emotion"""
        if "Happy" in name_with_emotion:
            return (0, 255, 0)  # Green
        elif "Sad" in name_with_emotion:
            return (255, 0, 0)  # Blue
        elif "Angry" in name_with_emotion:
            return (0, 0, 255)  # Red
        elif "Surprise" in name_with_emotion:
            return (255, 255, 0)  # Cyan
        elif "Fear" in name_with_emotion:
            return (0, 165, 255)  # Orange
        elif "Disgust" in name_with_emotion:
            return (128, 0, 128)  # Purple
        else:
            return (0, 0, 255)  # Default red for unknown/neutral
    
    def _split_text_for_display(self, text, max_width):
        """Split long text into multiple lines for better display"""
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            # Rough estimate: each character is about 8 pixels wide
            if len(test_line) * 8 <= max_width or not current_line:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        # Limit to 3 lines maximum
        if len(lines) > 3:
            lines = lines[:2] + [lines[2][:15] + "..."]
        
        return lines
    
    def close(self):
        """Clean up resources"""
        if self.video_capture:
            self.video_capture.release()
        if hasattr(self.emotion_detector, 'cleanup'):
            self.emotion_detector.cleanup()
        cv2.destroyAllWindows()
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.close()

# Global instance for easy access
_face_recognizer = None

def get_face_recognizer():
    """Get or create the global face recognizer instance"""
    global _face_recognizer
    if _face_recognizer is None:
        _face_recognizer = FaceRecognizer()
    return _face_recognizer

def recognize_faces():
    """Fast face recognition function with emotions - maintains compatibility"""
    recognizer = get_face_recognizer()
    return recognizer.recognize_faces()

def cleanup_face_recognizer():
    """Call this when your application shuts down"""
    global _face_recognizer
    if _face_recognizer:
        _face_recognizer.close()
        _face_recognizer = None



app = Flask(__name__)

@app.route('/faces', methods=['GET'])
def faces():
    faces_with_emotions = recognize_faces()
    # Return JSON for easier consumption by clients
    return jsonify({"detected": faces_with_emotions})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)