import os
import datetime
import pandas as pd
import streamlit as st


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
