import os
import streamlit as st


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
