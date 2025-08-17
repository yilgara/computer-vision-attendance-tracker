import os, csv, pickle, datetime, time, logging
import numpy as np
import pandas as pd
from deepface import DeepFace
import streamlit as st

class AttendanceTracker:
    def __init__(self):
        self.employees_folder = "employees"
        self.csv_file = "attendance.csv"
        self.embeddings_file = "face_embeddings.pkl"
        self.known_embeddings = []
        self.known_names = []
        self.last_seen = {}
        self.entry_exit_buffer = 30
        
        # Face recognition models
        self.detection_backend = 'retinaface'
        self.recognition_model = 'Facenet512'
        self.similarity_threshold = 0.6
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize session state
        self.init_session_state()
        
        # Create necessary directories
        self.init_directories()
        
        # Initialize CSV
        self.init_csv()
        self.load_embeddings()
    
    def init_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'tracker_initialized' not in st.session_state:
            st.session_state.tracker_initialized = False
        if 'embeddings_loaded' not in st.session_state:
            st.session_state.embeddings_loaded = False
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False
        if 'attendance_logs' not in st.session_state:
            st.session_state.attendance_logs = []
        
    
    def init_directories(self):
        """Create necessary directories"""
        os.makedirs(self.employees_folder, exist_ok=True)
    
    def init_csv(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Employee', 'Date', 'Time', 'Action', 'Confidence'])
    
    def extract_face_embedding(self, image_array):
        """Extract face embedding from image array using DeepFace"""
        try:
            # Save image temporarily
            temp_path = "temp_image.jpg"
            cv2.imwrite(temp_path, image_array)
            
            # Extract embedding
            embedding = DeepFace.represent(
                img_path=temp_path,
                model_name=self.recognition_model,
                detector_backend=self.detection_backend,
                enforce_detection=False
            )
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if embedding and len(embedding) > 0:
                return np.array(embedding[0]["embedding"])
            else:
                return None
                
        except Exception as e:
            if os.path.exists("temp_image.jpg"):
                os.remove("temp_image.jpg")
            st.error(f"Error extracting embedding: {e}")
            return None
    
    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def recognize_face_from_embedding(self, face_embedding):
        """Recognize face from embedding using cosine similarity"""
        if not self.known_embeddings or face_embedding is None:
            return "Unknown", 0.0
        
        best_match = "Unknown"
        best_similarity = 0.0
        
        for i, known_embedding in enumerate(self.known_embeddings):
            similarity = self.cosine_similarity(face_embedding, known_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = self.known_names[i]
        
        confidence = best_similarity * 100
        
        if best_similarity >= self.similarity_threshold:
            return best_match, confidence
        else:
            return "Unknown", confidence
    
    def log_attendance(self, employee_name, action="ENTRY", confidence=0.0):
        """Log attendance to CSV file"""
        current_time = datetime.datetime.now()
        date_str = current_time.strftime("%Y-%m-%d")
        time_str = current_time.strftime("%H:%M:%S")
        
        # Check if this employee was seen recently
        current_timestamp = time.time()
        employee_key = f"{employee_name}_{action}"
        
        if employee_key in self.last_seen:
            if current_timestamp - self.last_seen[employee_key] < self.entry_exit_buffer:
                return False
        
        self.last_seen[employee_key] = current_timestamp
        
        try:
            with open(self.csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([employee_name, date_str, time_str, action, f"{confidence:.2f}"])
            
            # Add to session state for real-time display
            st.session_state.attendance_logs.append({
                'Employee': employee_name,
                'Date': date_str,
                'Time': time_str,
                'Action': action,
                'Confidence': f"{confidence:.2f}"
            })
            
            return True
        except Exception as e:
            st.error(f"Error logging attendance: {e}")
            return False
    
    def determine_action(self, employee_name):
        """Determine if this is an entry or exit based on recent history"""
        try:
            if os.path.exists(self.csv_file):
                df = pd.read_csv(self.csv_file)
                today = datetime.datetime.now().strftime("%Y-%m-%d")
                
                # Filter for today and this employee
                today_logs = df[(df['Employee'] == employee_name) & (df['Date'] == today)]
                
                if not today_logs.empty:
                    last_action = today_logs.iloc[-1]['Action']
                    return "EXIT" if last_action == "ENTRY" else "ENTRY"
                else:
                    return "ENTRY"
            else:
                return "ENTRY"
        except Exception as e:
            return "ENTRY"
    
    def load_embeddings(self):
        """Load face embeddings from file"""
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_embeddings = data['embeddings']
                    self.known_names = data['names']
                st.session_state.embeddings_loaded = True
                return True
            except Exception as e:
                st.error(f"Error loading embeddings: {e}")
                return False
        return False
    
    def save_embeddings(self):
        """Save face embeddings to file"""
        try:
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump({
                    'embeddings': self.known_embeddings,
                    'names': self.known_names
                }, f)
            return True
        except Exception as e:
            st.error(f"Error saving embeddings: {e}")
            return False



def get_tracker():
    """Get or create tracker instance using Streamlit session state"""
    if 'attendance_tracker' not in st.session_state:
        st.session_state.attendance_tracker = AttendanceTracker()
    return st.session_state.attendance_tracker
