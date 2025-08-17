import os
import io
import zipfile
import shutil
import tempfile
import time
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import streamlit as st


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
        
        # Load embeddings 
        if not tracker.load_embeddings():
            st.warning("No trained face embeddings found. Please add employees first.")
            return
        
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

                folders = [
                    d for d in os.listdir(temp_dir)
                    if os.path.isdir(os.path.join(temp_dir, d))
                    and not d.startswith('.')
                    and d != "__MACOSX"
                ]
              
                
                if len(folders) == 1:
                    inner_path = os.path.join(temp_dir, folders[0])
                    inner_folders = [
                        d for d in os.listdir(inner_path)
                        if os.path.isdir(os.path.join(inner_path, d))
                        and not d.startswith('.')
                        and d != "__MACOSX"
                    ]
               
                    for i, folder_name in enumerate(inner_folders):
                        status_text.text(f"Processing {folder_name}...")
                    
                        
                        source_folder = os.path.join(inner_path, folder_name)
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
                        
                        progress_bar.progress((i + 1) / len(inner_folders))
                
                    
                    tracker.save_embeddings()
                    st.success(f"Processed {len(inner_folders)} employees from ZIP file!")
                    status_text.empty()
                    progress_bar.empty()
