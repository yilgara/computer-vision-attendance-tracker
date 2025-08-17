import os
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import streamlit as st
from deepface import DeepFace



def show_live_recognition(tracker):
    """Show live face recognition interface"""
    st.header("Live Face Recognition")
    
    # Load embeddings if not loaded
    if not tracker.known_embeddings:
        if not tracker.load_embeddings():
            st.warning("No trained face embeddings found. Please add employees first.")
            return
    
    st.info(f"System ready with {len(set(tracker.known_names))} registered employees")
    
    
 
    show_manual_recognition(tracker)
    
    
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
