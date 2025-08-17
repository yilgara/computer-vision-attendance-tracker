import streamlit as st
import cv2
import numpy as np
import os
import csv
import datetime
import time
import pickle
import logging
import pandas as pd
from deepface import DeepFace
import warnings
import tempfile
import shutil
from PIL import Image
import io
import zipfile
import base64
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Employee Attendance System",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitAttendanceTracker:
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
        st.session_state.attendance_tracker = StreamlitAttendanceTracker()
    return st.session_state.attendance_tracker


def main():
    """Main Streamlit application"""
    st.title("Employee Attendance Tracking System")
    st.markdown("---")
    
    # Initialize tracker
    tracker = get_tracker()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Page", [
        "🏠 Dashboard",
        "👤 Manage Employees", 
        "📷 Live Recognition",
        "📊 Attendance Reports",
        "⚙️ Settings"
    ])
    
    if page == "🏠 Dashboard":
        show_dashboard(tracker)
    elif page == "👤 Manage Employees":
        show_employee_management(tracker)
    elif page == "📷 Live Recognition":
        show_live_recognition(tracker)
    elif page == "📊 Attendance Reports":
        show_reports(tracker)
    elif page == "⚙️ Settings":
        show_settings(tracker)

def show_dashboard(tracker):
    """Show dashboard with system overview"""
    st.header("System Dashboard")
    
    
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        num_employees = len(set(tracker.known_names)) if tracker.known_names else 0
        st.metric("Registered Employees", num_employees)
    
    with col2:
        if os.path.exists(tracker.csv_file):
            df = pd.read_csv(tracker.csv_file)
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            today_entries = len(df[df['Date'] == today])
        else:
            today_entries = 0
        st.metric("Today's Entries", today_entries)
    
    with col3:
        embeddings_count = len(tracker.known_embeddings)
        st.metric("Face Embeddings", embeddings_count)
    
    with col4:
        system_status = "✅ Ready" if tracker.known_embeddings else "⚠️ Setup Required"
        st.metric("System Status", system_status)
    
    # Recent activity
    st.subheader("Recent Activity")
    if os.path.exists(tracker.csv_file):
        df = pd.read_csv(tracker.csv_file)
        if not df.empty:
            recent_df = df.tail(10).sort_values('Date', ascending=False)
            st.dataframe(recent_df, use_container_width=True)
        else:
            st.info("No attendance records yet.")
    else:
        st.info("No attendance records yet.")

def show_employee_management(tracker):
    """Show employee management interface"""
    st.header("Employee Management")
    
    tab1, tab2, tab3 = st.tabs(["Add Employee", "View Employees", "Bulk Upload"])
    
    with tab1:
        st.subheader("Add New Employee")

        with st.form("add_employee_form", clear_on_submit=True):
            employee_name = st.text_input("Employee Name", placeholder="Enter full name")
            uploaded_files = st.file_uploader(
                "Upload Employee Photos", 
                type=['png', 'jpg', 'jpeg'], 
                accept_multiple_files=True,
                help="Upload multiple clear photos of the employee for better recognition"
            )
        
            if st.form_submit_button("Add Employee", type="primary"):
                
                if employee_name and uploaded_files:
                  
                    # Create employee folder
                    employee_folder = os.path.join(tracker.employees_folder, employee_name)
                    os.makedirs(employee_folder, exist_ok=True)
                    
                    # Save uploaded files
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    embeddings_to_add = []
                    names_to_add = []
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        # Save file
                        file_path = os.path.join(employee_folder, uploaded_file.name)
                        image = Image.open(uploaded_file)
                        image_array = np.array(image)
                        
                        if len(image_array.shape) == 3:
                            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                        
                        cv2.imwrite(file_path, image_array)
                        
                        # Extract embedding
                        status_text.text(f"Processing {uploaded_file.name}...")
                        embedding = tracker.extract_face_embedding(image_array)
                        
                        if embedding is not None:
                            embeddings_to_add.append(embedding)
                            names_to_add.append(employee_name)
                            status_text.text(f"✅ Processed {uploaded_file.name}")
                        else:
                            status_text.text(f"⚠️ No face found in {uploaded_file.name}")
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    # Add to tracker
                  
                    tracker.known_embeddings.extend(embeddings_to_add)
                    tracker.known_names.extend(names_to_add)
                    tracker.save_embeddings()


                    success_container = st.empty()
                    success_container.success(f"Added {employee_name} with {len(embeddings_to_add)} face embeddings!")
                    
                    # Clean up progress indicators
                    status_text.empty()
                    progress_bar.empty()
                    
                    # Clear success message after 3 seconds
                    time.sleep(3)
                    success_container.empty()
              
                else:
                    st.error("Please provide employee name and photos.")
    
    with tab2:
        st.subheader("Registered Employees")
        
        # Load embeddings if not loaded
        if not tracker.known_embeddings:
            tracker.load_embeddings()
        
        if tracker.known_names:
            # Get unique employee names and their photo counts
            unique_names = list(set(tracker.known_names))
            employee_data = []
            
            for name in unique_names:
                count = tracker.known_names.count(name)
                folder_path = os.path.join(tracker.employees_folder, name)
                photos_count = 0
                if os.path.exists(folder_path):
                    photos_count = len([f for f in os.listdir(folder_path) 
                                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                
                employee_data.append({
                    'Name': name,
                    'Face Embeddings': count,
                    'Photos': photos_count
                })
            
            df = pd.DataFrame(employee_data)
            st.dataframe(df, use_container_width=True)
            
            # Employee deletion
            st.subheader("Remove Employee")
            employee_to_remove = st.selectbox("Select employee to remove:", unique_names)
            
            if st.button("Remove Employee", type="secondary"):
                if employee_to_remove:
                    # Remove from embeddings
                    indices_to_remove = [i for i, name in enumerate(tracker.known_names) 
                                       if name == employee_to_remove]
                    
                    for index in sorted(indices_to_remove, reverse=True):
                        del tracker.known_embeddings[index]
                        del tracker.known_names[index]
                    
                    # Remove folder
                    folder_path = os.path.join(tracker.employees_folder, employee_to_remove)
                    if os.path.exists(folder_path):
                        shutil.rmtree(folder_path)
                    
                    tracker.save_embeddings()
                    st.success(f"Removed {employee_to_remove}")
                    st.rerun()
        else:
            st.info("No employees registered yet.")
    
    with tab3:
        st.subheader("Bulk Upload")
        st.info("Upload a ZIP file containing folders named after employees with their photos inside.")
        
        zip_file = st.file_uploader("Upload ZIP file", type=['zip'])
        
        if zip_file and st.button("Process ZIP file"):
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract ZIP
                with zipfile.ZipFile(zip_file) as zf:
                    zf.extractall(temp_dir)
                
                # Process folders
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                folders = [d for d in os.listdir(temp_dir) 
                          if os.path.isdir(os.path.join(temp_dir, d))]
                
                for i, folder_name in enumerate(folders):
                    status_text.text(f"Processing {folder_name}...")
                    
                    source_folder = os.path.join(temp_dir, folder_name)
                    dest_folder = os.path.join(tracker.employees_folder, folder_name)
                    os.makedirs(dest_folder, exist_ok=True)
                    
                    # Copy files and create embeddings
                    for file_name in os.listdir(source_folder):
                        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            source_path = os.path.join(source_folder, file_name)
                            dest_path = os.path.join(dest_folder, file_name)
                            shutil.copy2(source_path, dest_path)
                            
                            # Create embedding
                            image = cv2.imread(source_path)
                            if image is not None:
                                embedding = tracker.extract_face_embedding(image)
                                if embedding is not None:
                                    tracker.known_embeddings.append(embedding)
                                    tracker.known_names.append(folder_name)
                    
                    progress_bar.progress((i + 1) / len(folders))
                
                tracker.save_embeddings()
                st.success(f"Processed {len(folders)} employees from ZIP file!")
                status_text.empty()
                progress_bar.empty()

def show_live_recognition(tracker):
    """Show live face recognition interface"""
    st.header("Live Face Recognition")
    
    # Load embeddings if not loaded
    if not tracker.known_embeddings:
        if not tracker.load_embeddings():
            st.warning("No trained face embeddings found. Please add employees first.")
            return
    
    st.info(f"System ready with {len(set(tracker.known_names))} registered employees")
    
    # Mode selection
    st.subheader("Recognition Mode")
    mode = st.radio(
        "Select Recognition Mode:",
        options=["Manual Mode", "Automatic Mode", "DIO"],
        help="Manual: Take picture manually and approve logging. Automatic: Continuous detection with auto-logging."
    )
    
    if mode == "Manual Mode":
        show_manual_recognition(tracker)
    elif mode == "DIO":
        show_complete_diagnostic(tracker)
    else:
        show_automatic_recognition(tracker)
    
    # Manual entry option
    st.subheader("Manual Entry")
    with st.expander("Manual Attendance Entry"):
        if tracker.known_names:
            unique_names = list(set(tracker.known_names))
            selected_employee = st.selectbox("Select Employee:", unique_names)
            action = st.selectbox("Action:", ["ENTRY", "EXIT"])
            
            if st.button("Log Manual Entry"):
                success = tracker.log_attendance(selected_employee, action, 100.0)
                if success:
                    st.success(f"Manually logged {action} for {selected_employee}!")
                else:
                    st.warning("Entry was too recent, skipped logging.")




def show_manual_recognition(tracker):
    """Manual recognition mode - existing functionality"""
    st.subheader("📷 Manual Camera Input")
    st.info("Take a picture when ready, then manually approve attendance logging.")
    
    # Use Streamlit's camera input
    picture = st.camera_input("Take a picture for face recognition")
    
    if picture is not None:
        # Convert to OpenCV format
        image = Image.open(picture)
        image_array = np.array(image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Process image
        with st.spinner("Processing image..."):
            try:
                # Detect faces
                temp_path = "temp_camera.jpg"
                cv2.imwrite(temp_path, image_array)
                
                faces = DeepFace.extract_faces(
                    img_path=temp_path,
                    detector_backend=tracker.detection_backend,
                    enforce_detection=False
                )
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                if faces:
                    st.success(f"Found {len(faces)} face(s)")
                    
                    # Process each face
                    for i, face in enumerate(faces):
                        if isinstance(face, dict):
                            face_array = (face['face'] * 255).astype(np.uint8)
                        else:
                            face_array = (face * 255).astype(np.uint8)
                        
                        # Get embedding
                        embedding = tracker.extract_face_embedding(face_array)
                        
                        if embedding is not None:
                            # Recognize face
                            name, confidence = tracker.recognize_face_from_embedding(embedding)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.image(face_array, caption=f"Face {i+1}", width=200)
                            
                            with col2:
                                st.write(f"**Recognition Result:**")
                                st.write(f"Name: {name}")
                                st.write(f"Confidence: {confidence:.1f}%")
                                
                                if name != "Unknown" and confidence > (tracker.similarity_threshold * 100):
                                    action = tracker.determine_action(name)
                                    
                                    if st.button(f"Log {action} for {name}", key=f"log_{i}"):
                                        success = tracker.log_attendance(name, action, confidence)
                                        if success:
                                            st.success(f"Logged {action} for {name}!")
                                        else:
                                            st.warning("Entry was too recent, skipped logging.")
                                else:
                                    st.warning("Confidence too low or unknown person")
                        else:
                            st.error(f"Could not process face {i+1}")
                else:
                    st.warning("No faces detected in the image")
                    
            except Exception as e:
                st.error(f"Error processing image: {e}")

def show_live_recognition(tracker):
    """Show live face recognition interface"""
    st.header("Live Face Recognition")
    
    # Load embeddings if not loaded
    if not tracker.known_embeddings:
        if not tracker.load_embeddings():
            st.warning("No trained face embeddings found. Please add employees first.")
            return
    
    st.info(f"System ready with {len(set(tracker.known_names))} registered employees")
    
    # Mode selection
    st.subheader("Recognition Mode")
    mode = st.radio(
        "Select Recognition Mode:",
        options=["Manual Mode", "Automatic Mode"],
        help="Manual: Take picture manually and approve logging. Automatic: Continuous detection with auto-logging."
    )
    
    if mode == "Manual Mode":
        show_manual_recognition(tracker)
    else:
        show_automatic_recognition(tracker)

def show_manual_recognition(tracker):
    """Manual recognition mode - existing functionality"""
    st.subheader("📷 Manual Camera Input")
    st.info("Take a picture when ready, then manually approve attendance logging.")
    
    # Use Streamlit's camera input
    picture = st.camera_input("Take a picture for face recognition")
    
    if picture is not None:
        # Convert to OpenCV format
        image = Image.open(picture)
        image_array = np.array(image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Process image
        with st.spinner("Processing image..."):
            try:
                # Detect faces
                temp_path = "temp_camera.jpg"
                cv2.imwrite(temp_path, image_array)
                
                faces = DeepFace.extract_faces(
                    img_path=temp_path,
                    detector_backend=tracker.detection_backend,
                    enforce_detection=False
                )
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                if faces:
                    st.success(f"Found {len(faces)} face(s)")
                    
                    # Process each face
                    for i, face in enumerate(faces):
                        if isinstance(face, dict):
                            face_array = (face['face'] * 255).astype(np.uint8)
                        else:
                            face_array = (face * 255).astype(np.uint8)
                        
                        # Get embedding
                        embedding = tracker.extract_face_embedding(face_array)
                        
                        if embedding is not None:
                            # Recognize face
                            name, confidence = tracker.recognize_face_from_embedding(embedding)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.image(face_array, caption=f"Face {i+1}", width=200)
                            
                            with col2:
                                st.write(f"**Recognition Result:**")
                                st.write(f"Name: {name}")
                                st.write(f"Confidence: {confidence:.1f}%")
                                
                                if name != "Unknown" and confidence > (tracker.similarity_threshold * 100):
                                    action = tracker.determine_action(name)
                                    
                                    if st.button(f"Log {action} for {name}", key=f"log_{i}"):
                                        success = tracker.log_attendance(name, action, confidence)
                                        if success:
                                            st.success(f"Logged {action} for {name}!")
                                        else:
                                            st.warning("Entry was too recent, skipped logging.")
                                else:
                                    st.warning("Confidence too low or unknown person")
                        else:
                            st.error(f"Could not process face {i+1}")
                else:
                    st.warning("No faces detected in the image")
                    
            except Exception as e:
                st.error(f"Error processing image: {e}")

def get_camera_html(detection_interval):
    """Generate HTML/JS for automatic camera capture"""
    return f"""
    <div id="camera-container">
        <video id="camera-feed" autoplay playsinline style="width: 100%; max-width: 640px; height: auto; border-radius: 10px;"></video>
        <canvas id="capture-canvas" style="display: none;"></canvas>
        <div id="camera-status" style="margin-top: 10px; padding: 10px; background: #f0f2f6; border-radius: 5px;">
            <p id="status-text">Initializing camera...</p>
        </div>
    </div>

    <script>
    let video = document.getElementById('camera-feed');
    let canvas = document.getElementById('capture-canvas');
    let context = canvas.getContext('2d');
    let statusText = document.getElementById('status-text');
    let isCapturing = false;

    // Initialize camera
    async function initCamera() {{
        try {{
            const constraints = {{
                video: {{
                    width: {{ ideal: 640 }},
                    height: {{ ideal: 480 }},
                    facingMode: 'user'
                }}
            }};
            
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            video.srcObject = stream;
            statusText.textContent = '🟢 Camera active - Automatic detection enabled';
            statusText.style.color = 'green';
            
            // Start automatic capture
            startAutoCapture();
            
        }} catch (err) {{
            console.error('Camera access error:', err);
            statusText.textContent = '🔴 Camera access denied or not available';
            statusText.style.color = 'red';
        }}
    }}

    function startAutoCapture() {{
        if (isCapturing) return;
        isCapturing = true;
        
        setInterval(() => {{
            if (video.videoWidth > 0 && video.videoHeight > 0) {{
                captureFrame();
            }}
        }}, {detection_interval * 1000});
    }}

    function captureFrame() {{
        // Set canvas dimensions to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Draw current frame
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Convert to base64
        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        
        // Send to Streamlit
        window.parent.postMessage({{
            type: 'camera-capture',
            image: imageData,
            timestamp: Date.now()
        }}, '*');
    }}

    // Initialize on load
    initCamera();
    </script>
    """

def show_automatic_recognition(tracker):
    """Automatic recognition mode with continuous detection"""
    st.subheader("🔄 Automatic Recognition Mode")
    st.info("Camera will continuously monitor for faces and automatically log attendance.")
    
    # Initialize session state for automatic mode
    if 'auto_mode_active' not in st.session_state:
        st.session_state.auto_mode_active = False
    if 'auto_detection_logs' not in st.session_state:
        st.session_state.auto_detection_logs = []
    if 'last_processed_timestamp' not in st.session_state:
        st.session_state.last_processed_timestamp = 0
    
    # Auto mode settings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        auto_confidence_threshold = st.slider(
            "Auto-log Confidence Threshold:",
            min_value=0.5,
            max_value=1.0,
            value=max(0.8, tracker.similarity_threshold),
            step=0.05,
            help="Higher threshold for automatic logging (more strict)"
        )
    
    with col2:
        detection_interval = st.number_input(
            "Detection Interval (seconds):",
            min_value=1,
            max_value=30,
            value=3,
            help="How often to process frames for detection"
        )
    
    with col3:
        max_auto_logs = st.number_input(
            "Max Auto Logs:",
            min_value=5,
            max_value=100,
            value=20,
            help="Maximum automatic detections to show"
        )
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ Start Auto Detection", type="primary"):
            st.session_state.auto_mode_active = True
            st.rerun()
    
    with col2:
        if st.button("⏸️ Stop Auto Detection", type="secondary"):
            st.session_state.auto_mode_active = False
            st.rerun()
    
    with col3:
        if st.button("🗑️ Clear Logs"):
            st.session_state.auto_detection_logs = []
            st.rerun()
    
    # Show camera interface
    if st.session_state.auto_mode_active:
        st.success("🟢 Auto Detection Active")
        
        # Display automatic camera capture
        camera_html = get_camera_html(detection_interval)
        components.html(camera_html, height=600)
        
        # Handle captured images via JavaScript message
        captured_image = st.session_state.get('captured_image_data', None)
        
        if captured_image:
            current_timestamp = st.session_state.get('capture_timestamp', 0)
            
            # Only process if it's a new capture
            if current_timestamp > st.session_state.last_processed_timestamp:
                st.session_state.last_processed_timestamp = current_timestamp
                process_captured_frame(tracker, captured_image, auto_confidence_threshold, max_auto_logs)
        
        # Auto-refresh to handle new captures
        time.sleep(0.5)
        st.rerun()
        
    else:
        st.info("🔴 Auto Detection Stopped")
        st.write("Click 'Start Auto Detection' to begin continuous monitoring.")
    
    # Display automatic detection logs
    display_detection_logs(max_auto_logs)



def diagnose_tracker_object(tracker):
    """Comprehensive diagnosis of the tracker object"""
    st.subheader("🔧 Tracker Object Diagnosis")
    
    # Check if tracker exists
    if tracker is None:
        st.error("❌ Tracker object is None!")
        return False
    
    st.success("✅ Tracker object exists")
    
    # Check tracker attributes
    st.write("**Tracker Attributes:**")
    
    # Check for known_embeddings
    if hasattr(tracker, 'known_embeddings'):
        if tracker.known_embeddings:
            st.success(f"✅ known_embeddings: {len(tracker.known_embeddings)} embeddings found")
        else:
            st.error("❌ known_embeddings exists but is empty!")
    else:
        st.error("❌ Tracker has no 'known_embeddings' attribute!")
    
    # Check for known_names
    if hasattr(tracker, 'known_names'):
        if tracker.known_names:
            st.success(f"✅ known_names: {len(tracker.known_names)} names found")
            st.write(f"Names: {list(set(tracker.known_names))}")
        else:
            st.error("❌ known_names exists but is empty!")
    else:
        st.error("❌ Tracker has no 'known_names' attribute!")
    
    # Check essential methods
    essential_methods = [
        'extract_face_embedding',
        'recognize_face_from_embedding', 
        'determine_action',
        'log_attendance',
        'load_embeddings'
    ]
    
    st.write("**Required Methods:**")
    missing_methods = []
    
    for method in essential_methods:
        if hasattr(tracker, method):
            st.success(f"✅ {method}")
        else:
            st.error(f"❌ {method}")
            missing_methods.append(method)
    
    # Check other important attributes
    other_attributes = [
        'similarity_threshold',
        'detection_backend',
        'model_name'
    ]
    
    st.write("**Other Attributes:**")
    for attr in other_attributes:
        if hasattr(tracker, attr):
            value = getattr(tracker, attr)
            st.success(f"✅ {attr}: {value}")
        else:
            st.warning(f"⚠️ {attr}: Not found")
    
    # Show all attributes of the tracker
    with st.expander("🔍 All Tracker Attributes"):
        all_attrs = [attr for attr in dir(tracker) if not attr.startswith('_')]
        for attr in all_attrs:
            try:
                value = getattr(tracker, attr)
                if callable(value):
                    st.write(f"**{attr}()** - Method")
                else:
                    st.write(f"**{attr}** = {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
            except Exception as e:
                st.write(f"**{attr}** - Error accessing: {str(e)}")
    
    return len(missing_methods) == 0

def test_face_detection_pipeline():
    """Test the complete face detection pipeline step by step"""
    st.subheader("🧪 Face Detection Pipeline Test")
    
    # Create a test image with a face
    uploaded_file = st.file_uploader("Upload a test image with a clear face", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Test Image", width=400)
        
        # Convert to OpenCV format
        image_array = np.array(image)
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        st.write("**Step 1: Image Conversion**")
        st.success(f"✅ Image converted to OpenCV format: {image_array.shape}")
        
        # Test different detection backends
        backends = ['opencv', 'ssd', 'mtcnn', 'retinaface', 'mediapipe']
        
        for backend in backends:
            st.write(f"**Step 2: Testing {backend} backend**")
            
            try:
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    temp_path = tmp_file.name
                
                success = cv2.imwrite(temp_path, image_array)
                
                if not success:
                    st.error(f"❌ Failed to save image for {backend}")
                    continue
                
                st.success(f"✅ Image saved to temporary file")
                
                # Extract faces
                faces = DeepFace.extract_faces(
                    img_path=temp_path,
                    detector_backend=backend,
                    enforce_detection=False
                )
                
                # Clean up
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                
                if faces and len(faces) > 0:
                    st.success(f"✅ {backend}: Detected {len(faces)} face(s)")
                    
                    # Show detected faces
                    for i, face in enumerate(faces):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if isinstance(face, dict):
                                face_array = (face['face'] * 255).astype(np.uint8)
                            else:
                                face_array = (face * 255).astype(np.uint8)
                            
                            st.image(face_array, caption=f"Face {i+1} ({backend})", width=150)
                        
                        with col2:
                            st.write(f"Face shape: {face_array.shape}")
                            st.write(f"Face type: {type(face_array)}")
                            st.write(f"Face dtype: {face_array.dtype}")
                            
                        # This is where we would test embedding extraction
                        st.write(f"**Step 3: Ready for embedding extraction with {backend}**")
                        
                        return backend, face_array  # Return successful backend and face for further testing
                        
                else:
                    st.error(f"❌ {backend}: No faces detected")
                    
            except Exception as e:
                st.error(f"❌ {backend}: Error - {str(e)}")
    
    return None, None

def test_embedding_extraction(tracker, face_array):
    """Test embedding extraction with the tracker"""
    st.subheader("🔍 Embedding Extraction Test")
    
    if face_array is None:
        st.warning("No face array provided. Run face detection test first.")
        return None
    
    try:
        st.write("**Testing tracker.extract_face_embedding()**")
        
        # Test embedding extraction
        embedding = tracker.extract_face_embedding(face_array)
        
        if embedding is not None:
            st.success(f"✅ Embedding extracted successfully!")
            st.write(f"Embedding shape: {embedding.shape if hasattr(embedding, 'shape') else 'No shape attr'}")
            st.write(f"Embedding type: {type(embedding)}")
            st.write(f"Embedding sample: {str(embedding)[:100]}...")
            return embedding
        else:
            st.error("❌ extract_face_embedding returned None")
            
            # Try to debug why
            st.write("**Debugging embedding extraction:**")
            
            # Check if it's a method
            if not hasattr(tracker, 'extract_face_embedding'):
                st.error("Tracker has no extract_face_embedding method")
            elif not callable(getattr(tracker, 'extract_face_embedding')):
                st.error("extract_face_embedding is not callable")
            else:
                st.write("extract_face_embedding method exists and is callable")
                
                # Try with different face array formats
                st.write("Trying different face array formats...")
                
                # Try as float
                try:
                    face_float = face_array.astype(np.float32) / 255.0
                    embedding = tracker.extract_face_embedding(face_float)
                    if embedding is not None:
                        st.success("✅ Works with float32 normalized array")
                        return embedding
                except Exception as e:
                    st.write(f"Float32 format failed: {str(e)}")
                
                # Try with PIL Image
                try:
                    face_pil = Image.fromarray(cv2.cvtColor(face_array, cv2.COLOR_BGR2RGB))
                    embedding = tracker.extract_face_embedding(face_pil)
                    if embedding is not None:
                        st.success("✅ Works with PIL Image")
                        return embedding
                except Exception as e:
                    st.write(f"PIL Image format failed: {str(e)}")
            
            return None
            
    except Exception as e:
        st.error(f"❌ Error in embedding extraction: {str(e)}")
        
        # Show full traceback
        import traceback
        st.code(traceback.format_exc())
        
        return None

def test_face_recognition(tracker, embedding):
    """Test face recognition with the embedding"""
    st.subheader("🎯 Face Recognition Test")
    
    if embedding is None:
        st.warning("No embedding provided. Run embedding extraction test first.")
        return None, None
    
    try:
        st.write("**Testing tracker.recognize_face_from_embedding()**")
        
        # Test recognition
        name, confidence = tracker.recognize_face_from_embedding(embedding)
        
        st.success(f"✅ Recognition completed!")
        st.write(f"**Name:** {name}")
        st.write(f"**Confidence:** {confidence}")
        st.write(f"**Name type:** {type(name)}")
        st.write(f"**Confidence type:** {type(confidence)}")
        
        return name, confidence
        
    except Exception as e:
        st.error(f"❌ Error in face recognition: {str(e)}")
        
        # Show full traceback
        import traceback
        st.code(traceback.format_exc())
        
        return None, None

def test_attendance_logging(tracker, name, confidence):
    """Test attendance logging"""
    st.subheader("📝 Attendance Logging Test")
    
    if name is None or confidence is None:
        st.warning("No name/confidence provided. Run face recognition test first.")
        return
    
    try:
        if name == "Unknown":
            st.info("Cannot test logging with Unknown person")
            return
        
        st.write("**Testing tracker.determine_action()**")
        action = tracker.determine_action(name)
        st.success(f"✅ Action determined: {action}")
        
        st.write("**Testing tracker.log_attendance()**")
        
        # Show what would be logged
        st.write(f"Would log: {action} for {name} with {confidence} confidence")
        
        if st.button("Actually Log This Attendance"):
            success = tracker.log_attendance(name, action, confidence)
            
            if success:
                st.success(f"✅ Successfully logged {action} for {name}!")
            else:
                st.warning("⚠️ Logging returned False (might be too recent)")
        
    except Exception as e:
        st.error(f"❌ Error in attendance logging: {str(e)}")
        
        # Show full traceback
        import traceback
        st.code(traceback.format_exc())

def show_complete_diagnostic(tracker):
    """Show complete diagnostic interface"""
    st.header("🔧 Complete Face Recognition Diagnostic")
    st.info("This will help identify exactly what's wrong with your face recognition system")
    
    # Step 1: Diagnose tracker
    st.markdown("---")
    tracker_ok = diagnose_tracker_object(tracker)
    
    if not tracker_ok:
        st.error("🛑 Tracker object has missing methods. Fix tracker first before proceeding.")
        return
    
    # Step 2: Test face detection
    st.markdown("---")
    backend, face_array = test_face_detection_pipeline()
    
    if face_array is None:
        st.warning("🛑 Face detection failed. Upload a clear image with a visible face.")
        return
    
    # Step 3: Test embedding extraction
    st.markdown("---")
    embedding = test_embedding_extraction(tracker, face_array)
    
    if embedding is None:
        st.error("🛑 Embedding extraction failed. Check tracker's extract_face_embedding method.")
        return
    
    # Step 4: Test face recognition
    st.markdown("---")
    name, confidence = test_face_recognition(tracker, embedding)
    
    if name is None:
        st.error("🛑 Face recognition failed. Check tracker's recognize_face_from_embedding method.")
        return
    
    # Step 5: Test attendance logging
    st.markdown("---")
    test_attendance_logging(tracker, name, confidence)
    
    # Final summary
    st.markdown("---")
    st.subheader("📋 Diagnostic Summary")
    
    if name != "Unknown":
        st.success("🎉 Complete pipeline working! Face recognition system is functional.")
        st.info("If automatic detection still doesn't work, the issue is likely in the camera loop or frame processing.")
    else:
        st.warning("⚠️ Pipeline works but person not recognized. This means:")
        st.write("1. Face detection: ✅ Working")
        st.write("2. Embedding extraction: ✅ Working") 
        st.write("3. Face recognition: ✅ Working (but person unknown)")
        st.write("4. Issue: Person not in training data or confidence too low")

def create_simple_test_camera():
    """Create a simple camera test that manually processes one image"""
    st.subheader("📸 Simple Camera Test")
    st.info("Take a photo and manually process it step by step")
    
    picture = st.camera_input("Take a test picture")
    
    if picture is not None:
        return picture
    
    return None

def process_test_image_step_by_step(tracker, picture):
    """Process a test image step by step with full debugging"""
    st.subheader("🔍 Step-by-Step Image Processing")
    
    try:
        # Step 1: Load and convert image
        st.write("**Step 1: Loading image...**")
        image = Image.open(picture)
        st.image(image, caption="Original Image", width=300)
        
        image_array = np.array(image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        st.success(f"✅ Image loaded and converted: {image_array.shape}")
        
        # Step 2: Save to temporary file
        st.write("**Step 2: Saving to temporary file...**")
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        success = cv2.imwrite(temp_path, image_array)
        if success:
            st.success(f"✅ Image saved to: {temp_path}")
        else:
            st.error("❌ Failed to save image")
            return
        
        # Step 3: Face detection
        st.write("**Step 3: Detecting faces...**")
        
        backend = st.selectbox("Choose detection backend:", ['opencv', 'ssd', 'mtcnn'], key="test_backend")
        
        faces = DeepFace.extract_faces(
            img_path=temp_path,
            detector_backend=backend,
            enforce_detection=False
        )
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        
        if not faces or len(faces) == 0:
            st.error(f"❌ No faces detected with {backend} backend")
            return
        
        st.success(f"✅ Found {len(faces)} face(s)")
        
        # Step 4: Process each face
        for i, face in enumerate(faces):
            st.write(f"**Step 4.{i+1}: Processing face {i+1}...**")
            
            # Convert face to proper format
            if isinstance(face, dict):
                face_array = (face['face'] * 255).astype(np.uint8)
            else:
                face_array = (face * 255).astype(np.uint8)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(face_array, caption=f"Detected Face {i+1}", width=200)
            
            with col2:
                st.write(f"Face shape: {face_array.shape}")
                st.write(f"Face dtype: {face_array.dtype}")
                
                # Step 5: Extract embedding
                st.write(f"**Step 5.{i+1}: Extracting embedding...**")
                
                embedding = tracker.extract_face_embedding(face_array)
                
                if embedding is not None:
                    st.success("✅ Embedding extracted")
                    st.write(f"Embedding type: {type(embedding)}")
                    
                    # Step 6: Recognize face
                    st.write(f"**Step 6.{i+1}: Recognizing face...**")
                    
                    name, confidence = tracker.recognize_face_from_embedding(embedding)
                    
                    st.write(f"**Name:** {name}")
                    st.write(f"**Confidence:** {confidence:.1f}%")
                    
                    # Step 7: Check threshold and determine action
                    threshold = getattr(tracker, 'similarity_threshold', 0.8) * 100
                    st.write(f"**Threshold:** {threshold}%")
                    
                    if name != "Unknown" and confidence > threshold:
                        st.success(f"✅ Recognition successful! Above threshold.")
                        
                        action = tracker.determine_action(name)
                        st.write(f"**Action:** {action}")
                        
                        # Option to log
                        if st.button(f"Log {action} for {name}", key=f"log_test_{i}"):
                            success = tracker.log_attendance(name, action, confidence)
                            if success:
                                st.success(f"✅ Successfully logged {action} for {name}!")
                            else:
                                st.warning("⚠️ Not logged (too recent or other issue)")
                    else:
                        st.warning(f"⚠️ Below threshold or unknown: {name} ({confidence:.1f}% < {threshold}%)")
                
                else:
                    st.error("❌ Failed to extract embedding")
    
    except Exception as e:
        st.error(f"❌ Error in processing: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def process_captured_frame(tracker, image_data, confidence_threshold, max_logs):
    """Process automatically captured frame"""
    try:
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image_array is None:
            return
        
        # Detect faces
        temp_path = f"temp_auto_{time.time()}.jpg"
        cv2.imwrite(temp_path, image_array)
        
        faces = DeepFace.extract_faces(
            img_path=temp_path,
            detector_backend=tracker.detection_backend,
            enforce_detection=False
        )
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if faces:
            for i, face in enumerate(faces):
                if isinstance(face, dict):
                    face_array = (face['face'] * 255).astype(np.uint8)
                else:
                    face_array = (face * 255).astype(np.uint8)
                
                # Get embedding
                embedding = tracker.extract_face_embedding(face_array)
                
                if embedding is not None:
                    # Recognize face
                    name, confidence = tracker.recognize_face_from_embedding(embedding)
                    
                    if name != "Unknown" and confidence > (confidence_threshold * 100):
                        action = tracker.determine_action(name)
                        
                        # Try to log attendance automatically
                        success = tracker.log_attendance(name, action, confidence)
                        
                        # Add to detection log
                        log_entry = {
                            'name': name,
                            'confidence': confidence,
                            'action': action,
                            'logged': success,
                            'timestamp': datetime.datetime.now().strftime("%H:%M:%S"),
                            'face_image': face_array
                        }
                        
                        # Add to session state log (keep only recent entries)
                        st.session_state.auto_detection_logs.append(log_entry)
                        if len(st.session_state.auto_detection_logs) > max_logs * 2:
                            st.session_state.auto_detection_logs = st.session_state.auto_detection_logs[-max_logs:]
                        
                        # Show immediate notification
                        if success:
                            st.toast(f"✅ {name} - {action} logged!", icon="✅")
                        else:
                            st.toast(f"⚠️ {name} detected but not logged (too recent)", icon="⚠️")
    
    except Exception as e:
        st.error(f"Auto detection error: {e}")

def display_detection_logs(max_logs):
    """Display automatic detection logs"""
    st.subheader("📝 Automatic Detection Log")
    
    if st.session_state.auto_detection_logs:
        # Show recent detections
        recent_logs = st.session_state.auto_detection_logs[-max_logs:]
        
        for i, log_entry in enumerate(reversed(recent_logs)):
            with st.expander(f"{log_entry['name']} - {log_entry['timestamp']}", expanded=False):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if 'face_image' in log_entry:
                        st.image(log_entry['face_image'], caption="Detected Face", width=150)
                
                with col2:
                    st.write(f"**Name:** {log_entry['name']}")
                    st.write(f"**Confidence:** {log_entry['confidence']:.1f}%")
                    st.write(f"**Action:** {log_entry['action']}")
                    st.write(f"**Time:** {log_entry['timestamp']}")
                    
                    if log_entry['logged']:
                        st.success("✅ Successfully logged to attendance")
                    else:
                        st.warning("⚠️ Not logged (recent entry or low confidence)")
    else:
        st.info("No automatic detections yet. Start auto detection to begin monitoring.")

# Add JavaScript message handler for Streamlit
def add_message_handler():
    """Add JavaScript message handler to receive camera captures"""
    js_code = """
    <script>
    window.addEventListener('message', function(event) {
        if (event.data.type === 'camera-capture') {
            // Store the captured image data in session state
            // This would need to be handled via Streamlit's component communication
            console.log('Received camera capture:', event.data.timestamp);
        }
    });
    </script>
    """
    components.html(js_code, height=0)





def show_reports(tracker):
    """Show attendance reports"""
    st.header("Attendance Reports")
    
    if not os.path.exists(tracker.csv_file):
        st.info("No attendance data available yet.")
        return
    
    # Load data
    df = pd.read_csv(tracker.csv_file)
    
    if df.empty:
        st.info("No attendance records found.")
        return
    
    # Convert date column
    df['Date'] = pd.to_datetime(df['Date'])
    df['Confidence'] = pd.to_numeric(df['Confidence'])
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        date_range = st.date_input(
            "Select Date Range:",
            value=(df['Date'].min().date(), df['Date'].max().date()),
            min_value=df['Date'].min().date(),
            max_value=df['Date'].max().date()
        )
    
    with col2:
        employees = st.multiselect(
            "Select Employees:",
            options=df['Employee'].unique(),
            default=df['Employee'].unique()
        )
    
    # Filter data
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = df[
            (df['Date'].dt.date >= start_date) & 
            (df['Date'].dt.date <= end_date) &
            (df['Employee'].isin(employees))
        ]
    else:
        filtered_df = df[df['Employee'].isin(employees)]
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_entries = len(filtered_df)
        st.metric("Total Records", total_entries)
    
    with col2:
        unique_employees = filtered_df['Employee'].nunique()
        st.metric("Active Employees", unique_employees)
    
    with col3:
        avg_confidence = filtered_df['Confidence'].mean() if not filtered_df.empty else 0
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
    
    # Display data table
    st.subheader("Attendance Records")
    st.dataframe(filtered_df.sort_values('Date', ascending=False), use_container_width=True)
    
    # Download option
    if not filtered_df.empty:
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"attendance_report_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Daily summary
    st.subheader("Daily Summary")
    if not filtered_df.empty:
        daily_summary = filtered_df.groupby(['Date', 'Employee']).size().unstack(fill_value=0)
        st.dataframe(daily_summary, use_container_width=True)

def show_settings(tracker):
    """Show system settings"""
    st.header("System Settings")

    
    # Model settings
    st.subheader("Recognition Settings")
    
    col1, col2 = st.columns(2)

    with col1:
        st.info(f"Face Detection Model: {tracker.detection_backend}")
    
    with col2:
        st.info(f"Face Recognition Model: {tracker.recognition_model}")
    
    similarity_threshold = st.slider(
        "Recognition Threshold:",
        min_value=0.1,
        max_value=1.0,
        value=tracker.similarity_threshold, 
        step=0.05,
        help="Higher values require more similarity for recognition"
    )
    
    entry_exit_buffer = st.number_input(
        "Entry/Exit Buffer (seconds):",
        min_value=5,
        max_value=300,
        value=tracker.entry_exit_buffer,  
        help="Minimum time between consecutive logs for same person"
    )
    
    if st.button("Update Settings"):
        tracker.detection_backend = detection_backend
        tracker.recognition_model = recognition_model
        tracker.similarity_threshold = similarity_threshold
        tracker.entry_exit_buffer = entry_exit_buffer
        st.success("Settings updated!")
    
    # Database management
    st.subheader("Database Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Rebuild Face Database"):
            if os.path.exists(tracker.embeddings_file):
                os.remove(tracker.embeddings_file)
            tracker.known_embeddings = []
            tracker.known_names = []
            st.success("Face database cleared. Please re-add employees.")
    
    with col2:
        if st.button("Clear Attendance Logs"):
            if os.path.exists(tracker.csv_file):
                os.remove(tracker.csv_file)
                tracker.init_csv()
            st.success("Attendance logs cleared.")
    
    # System info
    st.subheader("System Information")
    
    info_data = {
        "Employees Folder": tracker.employees_folder,
        "CSV File": tracker.csv_file,
        "Embeddings File": tracker.embeddings_file,
        "Detection Backend": tracker.detection_backend,
        "Recognition Model": tracker.recognition_model,
        "Threshold": tracker.similarity_threshold,
        "Buffer Time": f"{tracker.entry_exit_buffer}s"
    }
    
    for key, value in info_data.items():
        st.write(f"**{key}:** {value}")

if __name__ == "__main__":
    main()
