import streamlit as st
import pandas as pd
import os
import datetime



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
