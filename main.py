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
    st.title("👥 Employee Attendance Tracking System")
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
    
    # Camera input
    st.subheader("Camera Input")
    
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
