import streamlit as st
from tracker import get_tracker
from views import dashboard, employees, recognition, reports, settings

st.set_page_config(page_title="Employee Attendance System", layout="wide", initial_sidebar_state="expanded")


def main():
    st.title("Employee Attendance Tracking System")
    st.markdown("---")
    
    tracker = get_tracker()

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Page", [
        "🏠 Dashboard",
        "👤 Manage Employees", 
        "📷 Live Recognition",
        "📊 Attendance Reports",
        "⚙️ Settings"
    ])
    
    if page == "🏠 Dashboard":
        dashboard.show_dashboard(tracker)
    elif page == "👤 Manage Employees":
        employees.show_employee_management(tracker)
    elif page == "📷 Live Recognition":
        recognition.show_live_recognition(tracker)
    elif page == "📊 Attendance Reports":
        reports.show_reports(tracker)
    elif page == "⚙️ Settings":
        settings.show_settings(tracker)

if __name__ == "__main__":
    main()
