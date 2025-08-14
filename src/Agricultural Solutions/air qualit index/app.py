"""
Air Quality Index (AQI) Dashboard

This module provides a comprehensive dashboard for monitoring and analyzing air quality data
for agricultural planning and decision making.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

# Add parent directory to path to import UI components
sys.path.append(str(Path(__file__).parent.parent.parent))
from ui_components import UIComponents

def get_aqi_data():
    """Generate or load sample AQI data"""
    # This would typically come from an API or database in a real application
    cities = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad']
    
    # Generate sample data for demonstration
    np.random.seed(42)
    data = {
        'city': [],
        'aqi': [],
        'pm25': [],
        'pm10': [],
        'o3': [],
        'no2': [],
        'so2': [],
        'co': [],
        'timestamp': [],
        'category': []
    }
    
    for city in cities:
        for i in range(30):  # Last 30 days
            base_aqi = np.random.normal(loc=100, scale=40)
            aqi = max(0, min(500, int(base_aqi)))
            
            # Categorize AQI
            if aqi <= 50:
                category = 'Good'
            elif aqi <= 100:
                category = 'Moderate'
            elif aqi <= 150:
                category = 'Unhealthy for Sensitive Groups'
            elif aqi <= 200:
                category = 'Unhealthy'
            elif aqi <= 300:
                category = 'Very Unhealthy'
            else:
                category = 'Hazardous'
            
            data['city'].append(city)
            data['aqi'].append(aqi)
            data['pm25'].append(round(np.random.uniform(10, 300), 1))
            data['pm10'].append(round(np.random.uniform(20, 400), 1))
            data['o3'].append(round(np.random.uniform(10, 200), 1))
            data['no2'].append(round(np.random.uniform(5, 150), 1))
            data['so2'].append(round(np.random.uniform(2, 100), 1))
            data['co'].append(round(np.random.uniform(0.5, 15), 1))
            data['timestamp'].append(datetime.now() - timedelta(days=29-i))
            data['category'].append(category)
    
    return pd.DataFrame(data)

def get_aqi_color(aqi_value):
    """Get color based on AQI value"""
    if aqi_value <= 50:
        return '#4CAF50'  # Green
    elif aqi_value <= 100:
        return '#FFEB3B'  # Yellow
    elif aqi_value <= 150:
        return '#FF9800'  # Orange
    elif aqi_value <= 200:
        return '#F44336'  # Red
    elif aqi_value <= 300:
        return '#9C27B0'  # Purple
    else:
        return '#880E4F'  # Deep Purple

def get_health_recommendation(aqi_value):
    """Get health recommendations based on AQI value"""
    if aqi_value <= 50:
        return "Air quality is satisfactory. Enjoy your outdoor activities!"
    elif aqi_value <= 100:
        return "Air quality is acceptable. Consider limiting prolonged outdoor exertion."
    elif aqi_value <= 150:
        return "Sensitive groups should reduce prolonged outdoor exertion."
    elif aqi_value <= 200:
        return "Everyone may begin to experience health effects. Avoid prolonged outdoor exertion."
    elif aqi_value <= 300:
        return "Health alert: everyone may experience more serious health effects. Minimize outdoor activities."
    else:
        return "Health warning of emergency conditions. Everyone should avoid all outdoor activities."

def get_agricultural_recommendation(aqi_value, pollutant):
    """Get agricultural recommendations based on AQI and main pollutant"""
    if aqi_value <= 50:
        return "Ideal conditions for all agricultural activities. Proceed with planned operations."
    elif aqi_value <= 100:
        return "Good conditions for most agricultural activities. Monitor sensitive crops."
    elif aqi_value <= 150:
        if pollutant == 'PM2.5' or pollutant == 'PM10':
            return "Consider delaying pesticide applications as particles may reduce effectiveness. Irrigate to reduce dust."
        return "Proceed with caution. Monitor sensitive crops for stress symptoms."
    elif aqi_value <= 200:
        if pollutant == 'O3':
            return "Ozone can damage plant leaves. Consider delaying foliar applications. Irrigate in early morning."
        return "Limit outdoor work for farm workers. Consider rescheduling non-essential activities."
    elif aqi_value <= 300:
        return "Postpone non-essential field work. Protect sensitive crops with row covers if possible."
    else:
        return "Avoid all outdoor agricultural activities. Implement emergency measures to protect crops and workers."

def main():
    """Main function for the AQI Dashboard"""
    # Page configuration
    st.set_page_config(
        page_title="Air Quality Index - Agricultural Solutions",
        page_icon="🌫️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .stProgress > div > div > div {
        background-color: #E0E0E0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    UIComponents.header(
        "🌫️ Air Quality Index (AQI) Dashboard",
        "Monitor and analyze air quality data for agricultural planning"
    )
    
    # Load data
    df = get_aqi_data()
    latest_data = df[df['timestamp'] == df['timestamp'].max()]
    
    # Sidebar filters
    st.sidebar.title("🔍 Filters")
    selected_city = st.sidebar.selectbox(
        "Select City",
        options=sorted(df['city'].unique()),
        index=0
    )
    
    # Date range filter
    date_range = st.sidebar.slider(
        "Select Date Range",
        min_value=df['timestamp'].min().date(),
        max_value=df['timestamp'].max().date(),
        value=(df['timestamp'].max() - timedelta(days=7)).date(),
        format="MM/DD/YYYY"
    )
    
    # Filter data based on selections
    filtered_df = df[
        (df['city'] == selected_city) & 
        (df['timestamp'].dt.date >= date_range)
    ]
    
    # Main metrics
    st.markdown("### 🌡️ Current Air Quality")
    
    # Get latest AQI for selected city
    current_aqi = latest_data[latest_data['city'] == selected_city]['aqi'].values[0]
    current_category = latest_data[latest_data['city'] == selected_city]['category'].values[0]
    aqi_color = get_aqi_color(current_aqi)
    
    # Main AQI display
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div style='background-color: #f5f5f5; padding: 20px; border-radius: 10px;'>
            <h1 style='color: {aqi_color}; text-align: center; font-size: 4rem; margin: 0;'>{current_aqi}</h1>
            <p style='text-align: center; font-size: 1.2rem; margin: 0;'>{current_category}</p>
            <p style='text-align: center;'>Air Quality Index</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Secondary metrics
    with col2:
        UIComponents.metric_card(
            "PM2.5", 
            f"{latest_data[latest_data['city'] == selected_city]['pm25'].values[0]} µg/m³",
            "Fine Particulate Matter",
            aqi_color
        )
    
    with col3:
        UIComponents.metric_card(
            "PM10", 
            f"{latest_data[latest_data['city'] == selected_city]['pm10'].values[0]} µg/m³",
            "Coarse Particulate Matter",
            aqi_color
        )
    
    with col4:
        UIComponents.metric_card(
            "O₃", 
            f"{latest_data[latest_data['city'] == selected_city]['o3'].values[0]} ppb",
            "Ozone",
            aqi_color
        )
    
    # Health and Agricultural Recommendations
    st.markdown("### 📋 Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👥 Health Advisory")
        st.info(get_health_recommendation(current_aqi))
    
    with col2:
        st.markdown("#### 🌱 Agricultural Advisory")
        main_pollutant = "PM2.5" if current_aqi > 100 else "O3"
        st.warning(get_agricultural_recommendation(current_aqi, main_pollutant))
    
    # AQI Trend Chart
    st.markdown("### 📈 AQI Trend")
    
    fig_trend = go.Figure()
    
    # Add AQI line
    fig_trend.add_trace(go.Scatter(
        x=filtered_df['timestamp'],
        y=filtered_df['aqi'],
        mode='lines+markers',
        name='AQI',
        line=dict(color=aqi_color, width=3),
        marker=dict(size=8, color=aqi_color)
    ))
    
    # Add threshold lines
    fig_trend.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Good", annotation_position="bottom right")
    fig_trend.add_hline(y=100, line_dash="dash", line_color="yellow", annotation_text="Moderate", annotation_position="bottom right")
    fig_trend.add_hline(y=150, line_dash="dash", line_color="orange", annotation_text="Unhealthy for Sensitive", annotation_position="bottom right")
    fig_trend.add_hline(y=200, line_dash="dash", line_color="red", annotation_text="Unhealthy", annotation_position="bottom right")
    
    fig_trend.update_layout(
        title=f"AQI Trend for {selected_city}",
        xaxis_title="Date",
        yaxis_title="AQI Value",
        height=400,
        showlegend=True,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Pollutant Analysis
    st.markdown("### 🔍 Pollutant Analysis")
    
    pollutants = ['PM2.5', 'PM10', 'O₃', 'NO₂', 'SO₂', 'CO']
    pollutant_values = [
        latest_data[latest_data['city'] == selected_city]['pm25'].values[0],
        latest_data[latest_data['city'] == selected_city]['pm10'].values[0],
        latest_data[latest_data['city'] == selected_city]['o3'].values[0],
        latest_data[latest_data['city'] == selected_city]['no2'].values[0],
        latest_data[latest_data['city'] == selected_city]['so2'].values[0],
        latest_data[latest_data['city'] == selected_city]['co'].values[0]
    ]
    
    # Create a horizontal bar chart for pollutants
    fig_pollutants = go.Figure(go.Bar(
        x=pollutant_values,
        y=pollutants,
        orientation='h',
        marker_color=[aqi_color] * len(pollutants)
    ))
    
    fig_pollutants.update_layout(
        title="Current Pollutant Levels",
        xaxis_title="Concentration",
        yaxis_title="Pollutant",
        height=300
    )
    
    st.plotly_chart(fig_pollutants, use_container_width=True)
    
    # City Comparison
    st.markdown("### 🌆 City Comparison")
    
    fig_cities = px.bar(
        latest_data.sort_values('aqi', ascending=False),
        x='city',
        y='aqi',
        color='category',
        color_discrete_map={
            'Good': '#4CAF50',
            'Moderate': '#FFEB3B',
            'Unhealthy for Sensitive Groups': '#FF9800',
            'Unhealthy': '#F44336',
            'Very Unhealthy': '#9C27B0',
            'Hazardous': '#880E4F'
        },
        labels={'aqi': 'AQI', 'city': 'City', 'category': 'Category'}
    )
    
    fig_cities.update_layout(
        title="AQI Comparison Across Cities",
        height=400
    )
    
    st.plotly_chart(fig_cities, use_container_width=True)
    
    # Data Export
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📤 Export Data")
    
    export_format = st.sidebar.radio("Select Format", ["CSV", "Excel"])
    
    if st.sidebar.button("💾 Export Data"):
        if export_format == "CSV":
            csv = filtered_df.to_csv(index=False)
            st.sidebar.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"aqi_data_{selected_city.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
        else:
            excel = filtered_df.to_excel(index=False)
            st.sidebar.download_button(
                label="Download Excel",
                data=excel,
                file_name=f"aqi_data_{selected_city.lower().replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # Back button
    st.sidebar.markdown("---")
    if st.sidebar.button("⬅️ Back to Agricultural Solutions"):
        st.switch_page("../app.py")
    
    # Footer
    UIComponents.footer()

if __name__ == "__main__":
    main()
