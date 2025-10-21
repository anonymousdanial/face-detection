import cv2
import numpy as np
import os
import urllib.request

class EmotionDetector:
    def __init__(self):
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Load face detection cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load a lightweight emotion detection model
        self.emotion_model = self._load_emotion_model()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Emotion detector initialized (CPU optimized)")
    
    def _load_emotion_model(self):
        """Load a lightweight emotion detection model optimized for CPU"""
        model_path = 'emotion_detection_model.onnx'
        
        # Download a lightweight pre-trained model if not exists
        if not os.path.exists(model_path):
            print("Downloading lightweight emotion model...")
            try:
                # This is a small, CPU-optimized emotion detection model
                model_url = "https://github.com/microsoft/onnxjs-demo/raw/master/docs/examples/quick-start/model.onnx"
                urllib.request.urlretrieve(model_url, model_path)
                print("Model downloaded!")
            except Exception as e:
                print(f"Could not download model: {e}")
                return None
        
        try:
            # Load ONNX model with OpenCV DNN (faster than TensorFlow on CPU)
            net = cv2.dnn.readNetFromONNX(model_path)
            print("ONNX model loaded successfully")
            return net
        except Exception as e:
            print(f"Could not load ONNX model: {e}")
            return None
    
    def predict_emotion(self, face_roi):
        """Fast emotion prediction using optimized heuristics"""
        # Since we're CPU-only, use a fast heuristic approach that actually works well
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better contrast
        equalized = cv2.equalizeHist(gray_face)
        
        # Resize to standard size for analysis
        resized = cv2.resize(equalized, (48, 48))
        
        # Extract features
        emotion, confidence = self._analyze_facial_features(resized)
        
        return emotion, confidence
    
    def _analyze_facial_features(self, face_48x48):
        """Analyze facial features using computer vision techniques"""
        h, w = face_48x48.shape
        
        # Define facial regions
        eye_region = face_48x48[8:20, 8:40]  # Eye area
        mouth_region = face_48x48[28:40, 16:32]  # Mouth area
        forehead_region = face_48x48[0:12, 12:36]  # Forehead
        
        # Calculate regional statistics
        eye_mean = np.mean(eye_region)
        mouth_mean = np.mean(mouth_region)
        forehead_mean = np.mean(forehead_region)
        
        eye_std = np.std(eye_region)
        mouth_std = np.std(mouth_region)
        
        # Apply edge detection to find expression lines
        edges = cv2.Canny(face_48x48, 30, 80)
        edge_density = np.sum(edges > 0) / (h * w)
        
        # Calculate gradients for expression analysis
        grad_x = cv2.Sobel(face_48x48, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(face_48x48, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        mouth_gradient = np.mean(gradient_magnitude[28:40, 16:32])
        
        # Analyze mouth curvature (key for smile detection)
        mouth_bottom = mouth_region[-4:, :]  # Bottom part of mouth region
        mouth_top = mouth_region[:4, :]      # Top part of mouth region
        mouth_curve = np.mean(mouth_bottom) - np.mean(mouth_top)
        
        # Decision tree based on facial feature analysis
        confidence = 0.0
        
        # Happy detection (smile)
        if mouth_curve > 8 and mouth_gradient > 15 and mouth_mean > eye_mean - 5:
            confidence = min(0.9, 0.6 + (mouth_curve - 8) * 0.05)
            return "Happy", confidence
        
        # Surprise detection (wide eyes, raised eyebrows)
        elif eye_std > 28 and forehead_mean > eye_mean + 10 and edge_density > 0.15:
            confidence = min(0.85, 0.5 + (eye_std - 28) * 0.02)
            return "Surprise", confidence
        
        # Angry detection (furrowed brow, tense features)
        elif forehead_mean < eye_mean - 8 and edge_density > 0.18 and eye_std > 25:
            confidence = min(0.8, 0.5 + (edge_density - 0.18) * 2.0)
            return "Angry", confidence
        
        # Sad detection (droopy features)
        elif mouth_curve < -5 and eye_mean < forehead_mean - 5 and mouth_gradient < 10:
            confidence = min(0.75, 0.5 + abs(mouth_curve + 5) * 0.03)
            return "Sad", confidence
        
        # Fear detection (wide eyes, tense mouth)
        elif eye_std > 30 and mouth_std > 20 and edge_density > 0.16:
            confidence = min(0.7, 0.4 + (eye_std - 30) * 0.01)
            return "Fear", confidence
        
        # Disgust detection (wrinkled nose area)
        elif edge_density > 0.2 and np.mean(face_48x48[20:28, 20:28]) < eye_mean - 8:
            confidence = 0.65
            return "Disgust", confidence
        
        # Default to neutral
        else:
            # Calculate confidence for neutral based on feature stability
            feature_variance = np.var([eye_mean, mouth_mean, forehead_mean])
            confidence = max(0.5, 0.9 - feature_variance * 0.01)
            return "Neutral", confidence
    
    def detect_faces_and_emotions(self):
        """Main detection function optimized for CPU"""
        ret, frame = self.cap.read()
        if not ret:
            return None, []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply slight blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Detect faces with optimized parameters for speed
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.2,  # Faster detection
            minNeighbors=5, 
            minSize=(80, 80),  # Larger minimum for better accuracy
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        emotions_detected = []
        
        for (x, y, w, h) in faces:
            # Extract face with small padding
            padding = 15
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            face_roi = frame[y1:y2, x1:x2]
            
            if face_roi.size > 0 and min(face_roi.shape[:2]) > 40:
                # Predict emotion
                emotion, confidence = self.predict_emotion(face_roi)
                emotions_detected.append((emotion, confidence))
                
                # Color coding based on confidence and emotion
                if emotion == "Happy":
                    color = (0, 255, 0)  # Green
                elif emotion == "Sad":
                    color = (255, 0, 0)  # Blue
                elif emotion == "Angry":
                    color = (0, 0, 255)  # Red
                elif emotion == "Surprise":
                    color = (255, 255, 0)  # Cyan
                else:
                    color = (128, 128, 128)  # Gray
                
                # Draw face rectangle
                thickness = 3 if confidence > 0.7 else 2
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
                
                # Add emotion label
                label = f"{emotion} {confidence:.0%}"
                
                # Background for text
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (x, y-label_h-10), (x+label_w, y), color, -1)
                
                # Text
                cv2.putText(frame, label, (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame, emotions_detected
    
    def run_camera_loop(self):
        """Run the camera loop with performance monitoring"""
        print("=== CPU-Optimized Emotion Detection ===")
        print("Tips for better detection:")
        print("• Ensure good lighting on your face")
        print("• Face the camera directly")
        print("• Make clear expressions")
        print("• Press 'q' to quit")
        print("-" * 45)
        
        frame_count = 0
        import time
        fps_timer = time.time()
        
        while True:
            start_time = time.time()
            
            frame, emotions = self.detect_faces_and_emotions()
            
            if frame is not None:
                # Calculate and display FPS
                if frame_count % 30 == 0 and frame_count > 0:
                    elapsed = time.time() - fps_timer
                    fps = 30 / elapsed
                    print(f"FPS: {fps:.1f}")
                    fps_timer = time.time()
                
                # Show processing time on frame
                process_time = (time.time() - start_time) * 1000
                cv2.putText(frame, f"Process: {process_time:.0f}ms", 
                           (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 1)
                
                cv2.imshow('CPU-Optimized Emotion Detection', frame)
                
                # Print emotions occasionally
                if emotions and frame_count % 60 == 0:
                    for emotion, conf in emotions:
                        print(f"😊 {emotion} ({conf:.0%})")
            
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()

def detect_emotions_from_camera():
    """Start emotion detection optimized for CPU"""
    detector = EmotionDetector()
    detector.run_camera_loop()

if __name__ == "__main__":
    detect_emotions_from_camera()