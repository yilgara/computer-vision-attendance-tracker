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
        options=["Manual Mode", "Automatic Mode"],
        help="Manual: Take picture manually and approve logging. Automatic: Continuous detection with auto-logging."
    )
    
    if mode == "Manual Mode":
        show_manual_recognition(tracker)
    else:
        show_automatic_recognition_simple(tracker)
    
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



def show_automatic_recognition_simple(tracker):
    """Simple automatic recognition using native Streamlit camera input"""
    st.subheader("🔄 Simple Automatic Recognition Mode")
    st.info("This mode uses Streamlit's native camera for reliable automatic detection.")
    
    # Initialize session state
    if 'auto_mode_active' not in st.session_state:
        st.session_state.auto_mode_active = False
    if 'auto_detection_logs' not in st.session_state:
        st.session_state.auto_detection_logs = []
    if 'last_capture_time' not in st.session_state:
        st.session_state.last_capture_time = 0
    if 'capture_counter' not in st.session_state:
        st.session_state.capture_counter = 0
    if 'debug_captures' not in st.session_state:
        st.session_state.debug_captures = []
    
    # IMMEDIATE DEBUG INFO AT THE TOP - ALWAYS VISIBLE
    st.write("🔍 **IMMEDIATE DEBUG STATUS** (Top of page)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Auto Active", "✅" if st.session_state.auto_mode_active else "❌")
    with col2:
        st.metric("Captures", st.session_state.capture_counter)
    with col3:
        st.metric("Logs", len(st.session_state.auto_detection_logs))
    with col4:
        st.metric("Time", datetime.datetime.now().strftime("%H:%M:%S"))
    
    # Settings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        auto_confidence_threshold = st.slider(
            "Auto-log Confidence Threshold:",
            min_value=0.5,
            max_value=1.0,
            value=max(0.8, tracker.similarity_threshold),
            step=0.05
        )
    
    with col2:
        detection_interval = st.number_input(
            "Detection Interval (seconds):",
            min_value=2,
            max_value=30,
            value=5,
            help="How often to auto-refresh and capture"
        )
    
    with col3:
        max_auto_logs = st.number_input(
            "Max Auto Logs:",
            min_value=5,
            max_value=50,
            value=20
        )
    
    # Debug options
    st.subheader("🐛 Debug Options")
    col1, col2 = st.columns(2)
    
    with col1:
        debug_mode = st.checkbox("Show Debug Info", value=True)
        show_captures = st.checkbox("Show Recent Captures", value=True)
    
    with col2:
        max_debug_captures = st.number_input("Max Debug Captures:", min_value=1, max_value=5, value=3)
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ Start Auto Detection", type="primary"):
            st.session_state.auto_mode_active = True
            st.session_state.capture_counter = 0
            st.session_state.debug_captures = []
            st.success("✅ Auto detection started!")
            time.sleep(1)  # Brief pause before first capture
            st.rerun()
    
    with col2:
        if st.button("⏸️ Stop Auto Detection", type="secondary"):
            st.session_state.auto_mode_active = False
            st.info("⏸️ Auto detection stopped")
            st.rerun()
    
    with col3:
        if st.button("🗑️ Clear All Data"):
            st.session_state.auto_detection_logs = []
            st.session_state.debug_captures = []
            st.session_state.capture_counter = 0
            st.rerun()
    
    # Main auto detection logic
    if st.session_state.auto_mode_active:
        # SHOW DEBUG BEFORE RUNNING AUTO DETECTION
        if debug_mode:
            st.write("🐛 **DEBUG: About to run auto detection**")
            st.write(f"- Mode active: {st.session_state.auto_mode_active}")
            st.write(f"- Current counter: {st.session_state.capture_counter}")
            st.write(f"- Time: {datetime.datetime.now().strftime('%H:%M:%S')}")
        
        run_automatic_detection(
            tracker, 
            auto_confidence_threshold, 
            detection_interval,
            debug_mode, 
            show_captures, 
            max_debug_captures, 
            max_auto_logs
        )
    else:
        st.info("👆 Click 'Start Auto Detection' to begin monitoring")
        
        # SHOW DEBUG INFO WHEN NOT RUNNING - ALWAYS VISIBLE
        if debug_mode:
            st.write("🐛 **DEBUG: Auto detection not running**")
            show_comprehensive_debug_status()
        
        if show_captures:
            show_recent_captures(max_debug_captures)
        
        show_detection_results(max_auto_logs)


def run_automatic_detection(tracker, confidence_threshold, detection_interval, 
                          debug_mode, show_captures, max_debug_captures, max_auto_logs):
    """Run the automatic detection loop"""
    
    # SHOW DEBUG AT THE VERY START OF AUTO DETECTION
    if debug_mode:
        st.write("🐛 **DEBUG: Inside run_automatic_detection function**")
        st.write(f"- Confidence threshold: {confidence_threshold}")
        st.write(f"- Detection interval: {detection_interval}s")
        st.write(f"- Current capture counter: {st.session_state.capture_counter}")
    
    st.success("🟢 **Auto Detection Active**")
    
    if debug_mode:
        st.write(f"🔍 **Status**: Capture #{st.session_state.capture_counter + 1} starting...")
        st.write(f"🔍 **Settings**: Confidence {confidence_threshold:.2f}, Interval {detection_interval}s")
    
    # Create unique key for camera input to force refresh
    camera_key = f"auto_camera_{st.session_state.capture_counter}_{int(time.time())}"
    
    if debug_mode:
        st.write(f"🔍 **Camera Key**: {camera_key}")
    
    # Native Streamlit camera input - this is guaranteed to work!
    st.write("📷 **Live Camera Feed**")
    picture = st.camera_input(
        "Auto-capturing for face detection...", 
        key=camera_key,
        help=f"Capture #{st.session_state.capture_counter + 1} - Auto-refreshing every {detection_interval}s"
    )
    
    if debug_mode:
        st.write(f"🔍 **Camera Input Result**: {'Image received' if picture is not None else 'No image yet'}")
    
    # Show the captured image immediately
    if picture is not None:
        st.session_state.capture_counter += 1
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        
        if debug_mode:
            st.write(f"🔍 **SUCCESS**: Captured image #{st.session_state.capture_counter} at {current_time}")
        
        # Display the captured image prominently
        st.write("📸 **Current Capture**")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(
                picture, 
                caption=f"Capture #{st.session_state.capture_counter} at {current_time}",
                width=400
            )
        
        with col2:
            st.write(f"**Capture #:** {st.session_state.capture_counter}")
            st.write(f"**Time:** {current_time}")
            st.write(f"**Next capture in:** {detection_interval}s")
            
            if debug_mode:
                st.success("✅ Image captured successfully!")
        
        # Process the image
        if debug_mode:
            st.write("🔍 **Starting image processing...**")
        
        success = process_camera_capture(
            tracker, 
            picture, 
            confidence_threshold, 
            debug_mode, 
            show_captures, 
            max_debug_captures, 
            max_auto_logs
        )
        
        if debug_mode:
            st.write(f"🔍 **Processing result**: {'Success' if success else 'Failed'}")
    
    else:
        if debug_mode:
            st.write("⏳ **Waiting for camera capture...**")
        
        # Show placeholder when no image
        st.info("📷 Waiting for camera to capture image...")
    
    # SHOW DEBUG INFO BEFORE COUNTDOWN TO ENSURE IT'S VISIBLE
    if debug_mode:
        st.write("🔍 **About to start countdown...**")
        show_comprehensive_debug_status()
    
    # Show countdown and auto-refresh
    if st.session_state.auto_mode_active:
        show_countdown_and_refresh(detection_interval, debug_mode)


def process_camera_capture(tracker, picture, confidence_threshold, debug_mode, 
                         show_captures, max_debug_captures, max_auto_logs):
    """Process the captured image from camera input"""
    
    try:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        if debug_mode:
            st.write(f"🔍 **Step 1**: Converting image at {timestamp}")
        
        # Convert to OpenCV format (same as manual mode)
        image = Image.open(picture)
        image_array = np.array(image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        if debug_mode:
            st.write(f"🔍 **Step 2**: Image shape: {image_array.shape}")
        
        # Store for debugging if requested
        if show_captures:
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            st.session_state.debug_captures.append({
                'image': rgb_image,
                'timestamp': timestamp,
                'capture_number': st.session_state.capture_counter,
                'shape': image_array.shape
            })
            # Keep only recent captures
            if len(st.session_state.debug_captures) > max_debug_captures:
                st.session_state.debug_captures = st.session_state.debug_captures[-max_debug_captures:]
        
        # Process with DeepFace (identical to manual mode)
        temp_path = f"temp_auto_{time.time()}_{st.session_state.capture_counter}.jpg"
        cv2.imwrite(temp_path, image_array)
        
        try:
            if debug_mode:
                st.write(f"🔍 **Step 3**: Running DeepFace detection...")
            
            faces = DeepFace.extract_faces(
                img_path=temp_path,
                detector_backend=tracker.detection_backend,
                enforce_detection=False
            )
            
            faces_count = len(faces) if faces else 0
            
            if debug_mode:
                st.write(f"🔍 **Step 4**: Found {faces_count} face(s)")
            
            if faces:
                detection_results = []
                
                for i, face in enumerate(faces):
                    if debug_mode:
                        st.write(f"🔍 **Step 5.{i+1}**: Processing face {i+1}")
                    
                    # Extract face array
                    if isinstance(face, dict):
                        face_array = (face['face'] * 255).astype(np.uint8)
                    else:
                        face_array = (face * 255).astype(np.uint8)
                    
                    # Get embedding
                    embedding = tracker.extract_face_embedding(face_array)
                    
                    if embedding is not None:
                        # Recognize face
                        name, confidence = tracker.recognize_face_from_embedding(embedding)
                        
                        if debug_mode:
                            st.write(f"🔍 **Step 6.{i+1}**: {name} - {confidence:.1f}%")
                        
                        detection_results.append({
                            'name': name,
                            'confidence': confidence,
                            'face_image': face_array
                        })
                        
                        # Auto-log if confidence is high enough
                        if name != "Unknown" and confidence > (confidence_threshold * 100):
                            action = tracker.determine_action(name)
                            
                            # Try to log attendance
                            logged = tracker.log_attendance(name, action, confidence)
                            
                            # Add to detection log
                            log_entry = {
                                'name': name,
                                'confidence': confidence,
                                'action': action,
                                'logged': logged,
                                'timestamp': timestamp,
                                'capture_number': st.session_state.capture_counter,
                                'face_image': face_array
                            }
                            
                            st.session_state.auto_detection_logs.append(log_entry)
                            if len(st.session_state.auto_detection_logs) > max_auto_logs * 2:
                                st.session_state.auto_detection_logs = st.session_state.auto_detection_logs[-max_auto_logs:]
                            
                            # Show immediate feedback
                            if logged:
                                st.success(f"✅ **Auto-logged**: {name} - {action} ({confidence:.1f}%)")
                                st.toast(f"✅ {name} - {action}", icon="✅")
                            else:
                                st.warning(f"⚠️ **Detected but not logged**: {name} (recent entry)")
                                st.toast(f"⚠️ {name} detected", icon="⚠️")
                            
                            if debug_mode:
                                st.write(f"🔍 **Step 7.{i+1}**: Logged {action} - Success: {logged}")
                        else:
                            if debug_mode:
                                reason = "Unknown person" if name == "Unknown" else f"Low confidence ({confidence:.1f}%)"
                                st.write(f"🔍 **Step 6.{i+1}**: Not logging - {reason}")
                    else:
                        if debug_mode:
                            st.error(f"🔍 **ERROR**: Could not extract embedding for face {i+1}")
                
                if debug_mode and detection_results:
                    st.write(f"🔍 **Summary**: Processed {len(detection_results)} faces successfully")
                    
            else:
                if debug_mode:
                    st.write("🔍 **Step 4**: No faces detected")
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return True
        
    except Exception as e:
        st.error(f"❌ **Processing Error**: {str(e)}")
        if debug_mode:
            st.write(f"🔍 **ERROR Details**: {e}")
        return False


def show_countdown_and_refresh(detection_interval, debug_mode):
    """Show countdown and handle auto-refresh with visual countdown"""
    
    # Visual countdown display
    countdown_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    # Show current capture gallery while counting down
    if st.session_state.debug_captures:
        st.write("📸 **Recent Captures Gallery**")
        show_capture_gallery()
    
    with countdown_placeholder.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"⏱️ **Next capture in {detection_interval} seconds...**")
        with col2:
            if st.button("🔄 Capture Now", key="manual_trigger"):
                st.rerun()
        with col3:
            if st.button("⏸️ Stop Auto", key="stop_auto", type="secondary"):
                st.session_state.auto_mode_active = False
                st.rerun()
    
    # Visual progress bar countdown
    progress_bar = progress_placeholder.progress(0)
    status_text = st.empty()
    
    # Countdown with progress bar
    for i in range(detection_interval):
        remaining = detection_interval - i
        progress = (i + 1) / detection_interval
        
        progress_bar.progress(progress)
        status_text.write(f"⏰ Next capture in **{remaining}** seconds...")
        
        time.sleep(1)
        
        # Check if user stopped detection during countdown
        if not st.session_state.auto_mode_active:
            status_text.write("⏸️ Auto detection stopped by user")
            return
    
    # Final update
    progress_bar.progress(1.0)
    status_text.write("🔄 **Capturing now...**")
    
    if st.session_state.auto_mode_active:  # Check if still active
        if debug_mode:
            st.write(f"🔄 **Auto-refreshing** after {detection_interval}s interval...")
        time.sleep(0.5)  # Brief pause before refresh
        st.rerun()


def show_capture_gallery():
    """Show a gallery of recent captures in a compact format"""
    if not st.session_state.debug_captures:
        return
        
    # Show last 3 captures in a row
    recent_captures = st.session_state.debug_captures[-3:]
    
    if len(recent_captures) >= 1:
        cols = st.columns(len(recent_captures))
        
        for i, capture in enumerate(recent_captures):
            with cols[i]:
                st.image(
                    capture['image'],
                    caption=f"#{capture['capture_number']} - {capture['timestamp']}",
                    width=150
                )


def show_debug_status():
    """Show debug status information"""
    st.subheader("🐛 Debug Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Captures", st.session_state.capture_counter)
    
    with col2:
        st.metric("Detection Logs", len(st.session_state.auto_detection_logs))
    
    with col3:
        st.metric("Debug Captures Stored", len(st.session_state.debug_captures))
    
    # Show recent activity
    if st.session_state.auto_detection_logs:
        latest_log = st.session_state.auto_detection_logs[-1]
        st.info(f"🕐 **Latest Detection**: {latest_log['name']} at {latest_log['timestamp']}")


def show_recent_captures(max_debug_captures):
    """Show recent captured images with enhanced display"""
    st.subheader("📸 Recent Captures Timeline")
    
    if st.session_state.debug_captures:
        # Show capture statistics
        total_captures = len(st.session_state.debug_captures)
        latest_capture = st.session_state.debug_captures[-1]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Captures", total_captures)
        with col2:
            st.metric("Latest Capture", f"#{latest_capture['capture_number']}")
        with col3:
            st.metric("Latest Time", latest_capture['timestamp'])
        
        # Display captures in a grid
        st.write("**Capture Gallery:**")
        
        # Create columns for grid display
        num_cols = min(3, len(st.session_state.debug_captures))
        cols = st.columns(num_cols)
        
        for i, capture in enumerate(reversed(st.session_state.debug_captures)):
            col_idx = i % num_cols
            with cols[col_idx]:
                # Enhanced image display with border
                st.image(
                    capture['image'],
                    caption=f"📷 Capture #{capture['capture_number']}\n🕐 {capture['timestamp']}\n📐 {capture['shape'][1]}x{capture['shape'][0]}",
                    width=200,
                    use_column_width=False
                )
                
                # Add capture details
                with st.expander(f"Details #{capture['capture_number']}", expanded=False):
                    st.write(f"**Timestamp:** {capture['timestamp']}")
                    st.write(f"**Dimensions:** {capture['shape']}")
                    st.write(f"**Capture Number:** {capture['capture_number']}")
        
        # Add real-time update indicator
        st.write("🔄 *Gallery updates automatically with each new capture*")
        
    else:
        st.info("📷 No captures yet - images will appear here as they are captured")
        st.write("**What you'll see here:**")
        st.write("- Each captured image with timestamp")
        st.write("- Image dimensions and quality info") 
        st.write("- Automatic updates every N seconds")
        st.write("- Last 3 captures displayed in real-time")


def show_detection_results(max_auto_logs):
    """Show detection results log"""
    st.subheader("📝 Auto Detection Log")
    
    if st.session_state.auto_detection_logs:
        # Show summary
        total_detections = len(st.session_state.auto_detection_logs)
        successful_logs = len([log for log in st.session_state.auto_detection_logs if log['logged']])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Detections", total_detections)
        with col2:
            st.metric("Successfully Logged", successful_logs)
        
        # Show recent detections
        recent_logs = st.session_state.auto_detection_logs[-max_auto_logs:]
        
        for log_entry in reversed(recent_logs):
            status_icon = "✅" if log_entry['logged'] else "⚠️"
            
            with st.expander(f"{status_icon} Capture #{log_entry['capture_number']} - {log_entry['name']} ({log_entry['timestamp']})", expanded=False):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(log_entry['face_image'], caption="Detected Face", width=150)
                
                with col2:
                    st.write(f"**Name:** {log_entry['name']}")
                    st.write(f"**Confidence:** {log_entry['confidence']:.1f}%")
                    st.write(f"**Action:** {log_entry['action']}")
                    st.write(f"**Time:** {log_entry['timestamp']}")
                    st.write(f"**Capture:** #{log_entry['capture_number']}")
                    
                    if log_entry['logged']:
                        st.success("✅ Successfully logged to attendance")
                    else:
                        st.warning("⚠️ Not logged (recent entry)")
    else:
        st.info("No detections yet. Start auto detection to begin monitoring.")


   

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
    """Automatic recognition mode with continuous detection and debugging"""
    st.subheader("🔄 Automatic Recognition Mode")
    st.info("Camera will continuously monitor for faces and automatically log attendance.")
    
    # Initialize session state for automatic mode
    if 'auto_mode_active' not in st.session_state:
        st.session_state.auto_mode_active = False
    if 'auto_detection_logs' not in st.session_state:
        st.session_state.auto_detection_logs = []
    if 'last_processed_timestamp' not in st.session_state:
        st.session_state.last_processed_timestamp = 0
    if 'debug_images' not in st.session_state:
        st.session_state.debug_images = []
    if 'debug_info' not in st.session_state:
        st.session_state.debug_info = []
    
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
    
    # DEBUG CONTROLS
    st.subheader("🐛 Debug Controls")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        debug_mode = st.checkbox("Enable Debug Mode", value=True)
    
    with col2:
        show_raw_images = st.checkbox("Show Raw Captured Images", value=True)
    
    with col3:
        max_debug_images = st.number_input("Max Debug Images:", min_value=1, max_value=10, value=3)
    
    # Control buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("▶️ Start Auto Detection", type="primary"):
            st.session_state.auto_mode_active = True
            st.session_state.debug_images = []  # Clear debug images
            st.session_state.debug_info = []    # Clear debug info
            st.rerun()
    
    with col2:
        if st.button("⏸️ Stop Auto Detection", type="secondary"):
            st.session_state.auto_mode_active = False
            st.rerun()
    
    with col3:
        if st.button("🗑️ Clear Logs"):
            st.session_state.auto_detection_logs = []
            st.rerun()
    
    with col4:
        if st.button("🧹 Clear Debug Data"):
            st.session_state.debug_images = []
            st.session_state.debug_info = []
            st.rerun()
    
    # Show camera interface
    if st.session_state.auto_mode_active:
        st.success("🟢 Auto Detection Active")
        
        # Display automatic camera capture
        camera_html = get_camera_html_with_debug(detection_interval, debug_mode)
        components.html(camera_html, height=600)
        
        # Handle captured images via JavaScript message
        captured_image = st.session_state.get('captured_image_data', None)
        
        if captured_image:
            current_timestamp = st.session_state.get('capture_timestamp', 0)
            
            # Only process if it's a new capture
            if current_timestamp > st.session_state.last_processed_timestamp:
                st.session_state.last_processed_timestamp = current_timestamp
                
                if debug_mode:
                    st.write(f"🔍 **DEBUG**: Processing new frame at timestamp: {current_timestamp}")
                
                process_captured_frame_with_debug(
                    tracker, captured_image, auto_confidence_threshold, 
                    max_auto_logs, debug_mode, show_raw_images, max_debug_images
                )
        
        # Auto-refresh to handle new captures
        time.sleep(0.5)
        st.rerun()
        
    else:
        st.info("🔴 Auto Detection Stopped")
        st.write("Click 'Start Auto Detection' to begin continuous monitoring.")
    
    # Show debug information
    if debug_mode:
        show_debug_information()
    
    # Show raw captured images if enabled
    if show_raw_images and st.session_state.debug_images:
        show_debug_images(max_debug_images)
    
    # Display automatic detection logs
    display_detection_logs(max_auto_logs)


def get_camera_html_with_debug(detection_interval, debug_mode):
    """Generate HTML/JS for automatic camera capture with debug features"""
    debug_console = "true" if debug_mode else "false"
    
    return f"""
    <div id="camera-container">
        <video id="camera-feed" autoplay playsinline style="width: 100%; max-width: 640px; height: auto; border-radius: 10px;"></video>
        <canvas id="capture-canvas" style="display: none;"></canvas>
        <div id="camera-status" style="margin-top: 10px; padding: 10px; background: #f0f2f6; border-radius: 5px;">
            <p id="status-text">Initializing camera...</p>
            <p id="debug-text" style="font-size: 12px; color: #666;"></p>
        </div>
    </div>

    <script>
    let video = document.getElementById('camera-feed');
    let canvas = document.getElementById('capture-canvas');
    let context = canvas.getContext('2d');
    let statusText = document.getElementById('status-text');
    let debugText = document.getElementById('debug-text');
    let isCapturing = false;
    let captureCount = 0;
    const debugMode = {debug_console};

    function debugLog(message) {{
        if (debugMode) {{
            console.log('[DEBUG]', message);
            debugText.textContent = `Debug: ${{message}} (Captures: ${{captureCount}})`;
        }}
    }}

    // Initialize camera
    async function initCamera() {{
        try {{
            debugLog('Requesting camera access...');
            
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
            
            debugLog('Camera initialized successfully');
            
            // Wait for video to be ready
            video.onloadedmetadata = function() {{
                debugLog(`Video dimensions: ${{video.videoWidth}}x${{video.videoHeight}}`);
                startAutoCapture();
            }};
            
        }} catch (err) {{
            console.error('Camera access error:', err);
            statusText.textContent = '🔴 Camera access denied or not available';
            statusText.style.color = 'red';
            debugLog(`Camera error: ${{err.message}}`);
        }}
    }}

    function startAutoCapture() {{
        if (isCapturing) return;
        isCapturing = true;
        
        debugLog(`Starting auto capture with ${{detection_interval}}s interval`);
        
        setInterval(() => {{
            if (video.videoWidth > 0 && video.videoHeight > 0) {{
                captureFrame();
            }} else {{
                debugLog('Video not ready, skipping frame');
            }}
        }}, {detection_interval * 1000});
    }}

    function captureFrame() {{
        try {{
            captureCount++;
            debugLog(`Capturing frame #${{captureCount}}`);
            
            // Set canvas dimensions to match video
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            debugLog(`Canvas set to ${{canvas.width}}x${{canvas.height}}`);
            
            // Draw current frame
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            // Convert to base64
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            const imageSizeKB = Math.round(imageData.length / 1024);
            
            debugLog(`Image captured: ${{imageSizeKB}}KB`);
            
            // Send to Streamlit
            window.parent.postMessage({{
                type: 'camera-capture',
                image: imageData,
                timestamp: Date.now(),
                frameNumber: captureCount,
                dimensions: {{
                    width: canvas.width,
                    height: canvas.height
                }},
                sizeKB: imageSizeKB
            }}, '*');
            
            debugLog('Frame sent to Streamlit');
            
        }} catch (error) {{
            console.error('Capture error:', error);
            debugLog(`Capture error: ${{error.message}}`);
        }}
    }}

    // Initialize on load
    initCamera();
    </script>
    """


def process_captured_frame_with_debug(tracker, image_data, confidence_threshold, max_logs, debug_mode, show_raw_images, max_debug_images):
    """Process automatically captured frame with comprehensive debugging"""
    debug_info = {
        'timestamp': datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3],
        'step': 'Starting',
        'error': None,
        'faces_detected': 0,
        'recognition_results': []
    }
    
    try:
        if debug_mode:
            st.write(f"🔍 **DEBUG Step 1**: Starting frame processing at {debug_info['timestamp']}")
        
        debug_info['step'] = 'Decoding image'
        
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        debug_info['image_size_bytes'] = len(image_bytes)
        
        if debug_mode:
            st.write(f"🔍 **DEBUG Step 2**: Decoded base64 image, size: {debug_info['image_size_bytes']} bytes")
        
        debug_info['step'] = 'Converting to OpenCV'
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image_array is None:
            debug_info['error'] = 'Failed to decode image to OpenCV format'
            if debug_mode:
                st.error(f"🔍 **DEBUG ERROR**: {debug_info['error']}")
            return
        
        debug_info['image_shape'] = image_array.shape
        
        if debug_mode:
            st.write(f"🔍 **DEBUG Step 3**: Converted to OpenCV format, shape: {debug_info['image_shape']}")
        
        # Store raw image for debugging
        if show_raw_images:
            # Convert BGR to RGB for display
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            st.session_state.debug_images.append({
                'image': rgb_image,
                'timestamp': debug_info['timestamp'],
                'shape': debug_info['image_shape']
            })
            # Keep only recent debug images
            if len(st.session_state.debug_images) > max_debug_images:
                st.session_state.debug_images = st.session_state.debug_images[-max_debug_images:]
        
        debug_info['step'] = 'Writing temporary file'
        
        # Detect faces
        temp_path = f"temp_auto_{time.time()}.jpg"
        cv2.imwrite(temp_path, image_array)
        
        if debug_mode:
            st.write(f"🔍 **DEBUG Step 4**: Wrote temporary file: {temp_path}")
        
        debug_info['step'] = 'DeepFace face detection'
        
        try:
            faces = DeepFace.extract_faces(
                img_path=temp_path,
                detector_backend=tracker.detection_backend,
                enforce_detection=False
            )
            debug_info['faces_detected'] = len(faces) if faces else 0
            
            if debug_mode:
                st.write(f"🔍 **DEBUG Step 5**: DeepFace detected {debug_info['faces_detected']} faces")
        
        except Exception as deepface_error:
            debug_info['error'] = f'DeepFace error: {str(deepface_error)}'
            if debug_mode:
                st.error(f"🔍 **DEBUG ERROR**: {debug_info['error']}")
            return
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                if debug_mode:
                    st.write(f"🔍 **DEBUG**: Cleaned up temporary file")
        
        debug_info['step'] = 'Processing faces'
        
        if faces:
            for i, face in enumerate(faces):
                face_debug = {'face_index': i}
                
                try:
                    if isinstance(face, dict):
                        face_array = (face['face'] * 255).astype(np.uint8)
                        face_debug['face_type'] = 'dict'
                    else:
                        face_array = (face * 255).astype(np.uint8)
                        face_debug['face_type'] = 'array'
                    
                    face_debug['face_shape'] = face_array.shape
                    
                    if debug_mode:
                        st.write(f"🔍 **DEBUG Step 6.{i+1}**: Processing face {i+1}, shape: {face_debug['face_shape']}")
                    
                    debug_info['step'] = f'Extracting embedding for face {i+1}'
                    
                    # Get embedding
                    embedding = tracker.extract_face_embedding(face_array)
                    
                    if embedding is not None:
                        face_debug['embedding_shape'] = embedding.shape if hasattr(embedding, 'shape') else 'N/A'
                        
                        if debug_mode:
                            st.write(f"🔍 **DEBUG Step 7.{i+1}**: Extracted embedding, shape: {face_debug['embedding_shape']}")
                        
                        debug_info['step'] = f'Recognizing face {i+1}'
                        
                        # Recognize face
                        name, confidence = tracker.recognize_face_from_embedding(embedding)
                        
                        face_debug.update({
                            'name': name,
                            'confidence': confidence,
                            'above_threshold': confidence > (confidence_threshold * 100)
                        })
                        
                        if debug_mode:
                            st.write(f"🔍 **DEBUG Step 8.{i+1}**: Recognition result - Name: {name}, Confidence: {confidence:.1f}%")
                        
                        if name != "Unknown" and confidence > (confidence_threshold * 100):
                            action = tracker.determine_action(name)
                            face_debug['action'] = action
                            
                            debug_info['step'] = f'Logging attendance for {name}'
                            
                            # Try to log attendance automatically
                            success = tracker.log_attendance(name, action, confidence)
                            face_debug['logged'] = success
                            
                            if debug_mode:
                                st.write(f"🔍 **DEBUG Step 9.{i+1}**: Attendance logging - Action: {action}, Success: {success}")
                            
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
                        else:
                            face_debug['reason_not_logged'] = 'Low confidence or unknown person'
                            if debug_mode:
                                st.write(f"🔍 **DEBUG Step 8.{i+1}**: Not logging - {face_debug['reason_not_logged']}")
                    else:
                        face_debug['embedding_error'] = 'Failed to extract embedding'
                        if debug_mode:
                            st.error(f"🔍 **DEBUG ERROR**: Could not extract embedding for face {i+1}")
                
                except Exception as face_error:
                    face_debug['error'] = str(face_error)
                    if debug_mode:
                        st.error(f"🔍 **DEBUG ERROR**: Error processing face {i+1}: {face_error}")
                
                debug_info['recognition_results'].append(face_debug)
        
        else:
            if debug_mode:
                st.write("🔍 **DEBUG**: No faces detected in this frame")
        
        debug_info['step'] = 'Completed successfully'
    
    except Exception as e:
        debug_info['error'] = str(e)
        debug_info['step'] = f'Failed at step: {debug_info["step"]}'
        if debug_mode:
            st.error(f"🔍 **DEBUG ERROR**: Auto detection error at {debug_info['step']}: {e}")
    
    finally:
        # Store debug info
        st.session_state.debug_info.append(debug_info)
        # Keep only recent debug info
        if len(st.session_state.debug_info) > 20:
            st.session_state.debug_info = st.session_state.debug_info[-20:]


def show_debug_information():
    """Display debug information"""
    st.subheader("🐛 Debug Information")
    
    if st.session_state.debug_info:
        # Show latest debug info
        latest_debug = st.session_state.debug_info[-1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Latest Processing Info:**")
            st.json({
                'timestamp': latest_debug.get('timestamp'),
                'final_step': latest_debug.get('step'),
                'faces_detected': latest_debug.get('faces_detected', 0),
                'image_size_bytes': latest_debug.get('image_size_bytes'),
                'image_shape': latest_debug.get('image_shape'),
                'error': latest_debug.get('error')
            })
        
        with col2:
            st.write("**Recognition Results:**")
            if latest_debug.get('recognition_results'):
                for i, result in enumerate(latest_debug['recognition_results']):
                    st.write(f"Face {i+1}: {result}")
            else:
                st.write("No recognition results")
        
        # Show processing history
        with st.expander("Processing History (Last 10)", expanded=False):
            recent_debug = st.session_state.debug_info[-10:]
            for i, debug in enumerate(reversed(recent_debug)):
                status = "✅" if debug.get('error') is None else "❌"
                st.write(f"{status} **{debug.get('timestamp')}** - {debug.get('step')} - Faces: {debug.get('faces_detected', 0)}")
                if debug.get('error'):
                    st.error(f"Error: {debug['error']}")
    else:
        st.info("No debug information available yet.")


def show_debug_images(max_debug_images):
    """Display raw captured images for debugging"""
    st.subheader("📸 Raw Captured Images")
    
    if st.session_state.debug_images:
        st.write(f"Showing last {len(st.session_state.debug_images)} captured frames:")
        
        # Create columns for side-by-side display
        cols = st.columns(min(3, len(st.session_state.debug_images)))
        
        for i, debug_img in enumerate(reversed(st.session_state.debug_images)):
            col_idx = i % len(cols)
            with cols[col_idx]:
                st.image(
                    debug_img['image'], 
                    caption=f"Captured at {debug_img['timestamp']}\nShape: {debug_img['shape']}", 
                    width=200
                )
    else:
        st.info("No debug images captured yet.")








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
