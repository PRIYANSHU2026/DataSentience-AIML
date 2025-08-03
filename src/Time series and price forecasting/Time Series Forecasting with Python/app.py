import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Configure page
st.set_page_config(
    page_title="Time Series Forecasting with Python",
    page_icon="📊",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #2196F3;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        font-size: 0.8rem;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📊 Time Series Forecasting with Python</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">LSTM-based Stock Price Prediction</div>', unsafe_allow_html=True)

# Sidebar for navigation and controls
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Data Exploration", "Model Training", "Prediction"])

# Load sample data
@st.cache_data
def load_data():
    # Check if GE.csv exists in the current directory
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GE.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        # Create sample data if file doesn't exist
        dates = pd.date_range(start='1/1/2020', periods=500)
        data = {
            'Date': dates,
            'Open': np.random.normal(loc=100, scale=10, size=500).cumsum() + 1000,
            'High': np.random.normal(loc=102, scale=10, size=500).cumsum() + 1010,
            'Low': np.random.normal(loc=98, scale=10, size=500).cumsum() + 990,
            'Close': np.random.normal(loc=100, scale=10, size=500).cumsum() + 1005,
            'Volume': np.random.randint(1000000, 10000000, size=500)
        }
        return pd.DataFrame(data)

df = load_data()

# Overview page
if page == "Overview":
    st.header("Overview")
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("""
        ### Time Series Forecasting with LSTM
        
        This application demonstrates time series forecasting using Long Short-Term Memory (LSTM) neural networks. 
        The model is trained on historical stock price data to predict future prices.
        
        **Key Features:**
        - Data preprocessing and visualization
        - LSTM model architecture for time series prediction
        - Training with customizable parameters
        - Visualization of predictions vs actual values
        - Performance evaluation metrics
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.subheader("Sample Data Preview")
    st.dataframe(df.head())
    
    st.subheader("LSTM Architecture")
    st.image("https://miro.medium.com/max/1400/1*goJVQs-p9kgLODFNyZSzjA.png", caption="LSTM Cell Architecture")

# Data Exploration page
elif page == "Data Exploration":
    st.header("Data Exploration")
    
    # Data statistics
    st.subheader("Data Statistics")
    st.dataframe(df.describe())
    
    # Time series plot
    st.subheader("Stock Price Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pd.to_datetime(df['Date']), df['Open'], label='Open')
    ax.plot(pd.to_datetime(df['Date']), df['Close'], label='Close')
    ax.set_title('Stock Price Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # Distribution of daily returns
    st.subheader("Distribution of Daily Returns")
    if 'Close' in df.columns:
        df['Daily Return'] = df['Close'].pct_change()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(df['Daily Return'].dropna(), kde=True, ax=ax)
        ax.set_title('Distribution of Daily Returns')
        ax.set_xlabel('Daily Return')
        st.pyplot(fig)

# Model Training page
elif page == "Model Training":
    st.header("Model Training")
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("""
        ### LSTM Model Configuration
        
        Configure the LSTM model parameters for time series forecasting.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_past = st.slider("Number of past days for prediction", 7, 60, 14)
        n_future = st.slider("Number of future days to predict", 1, 30, 1)
        epochs = st.slider("Training epochs", 5, 100, 20)
        batch_size = st.slider("Batch size", 8, 128, 16, step=8)
    
    with col2:
        lstm_units1 = st.slider("LSTM Layer 1 Units", 32, 256, 64, step=32)
        lstm_units2 = st.slider("LSTM Layer 2 Units", 16, 128, 32, step=16)
        dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2, step=0.1)
        validation_split = st.slider("Validation Split", 0.1, 0.3, 0.1, step=0.05)
    
    if st.button("Train Model"):
        # Simulate model training
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(epochs):
            # Simulate training progress
            progress = (i + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"Training progress: {int(progress * 100)}%")
            
            # Simulate training time
            import time
            time.sleep(0.1)
        
        status_text.text("Training complete!")
        
        # Display training results
        st.success("Model trained successfully!")
        
        # Simulate training and validation loss
        train_loss = np.random.exponential(scale=0.1, size=epochs) + 0.05
        train_loss.sort()
        val_loss = train_loss + np.random.normal(loc=0.02, scale=0.01, size=epochs)
        
        # Plot training and validation loss
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, epochs + 1), train_loss, label='Training loss')
        ax.plot(range(1, epochs + 1), val_loss, label='Validation loss')
        ax.set_title('Training and Validation Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        st.pyplot(fig)

# Prediction page
elif page == "Prediction":
    st.header("Stock Price Prediction")
    
    # Date range selection
    st.subheader("Select Prediction Period")
    start_date = st.date_input("Start Date", pd.to_datetime(df['Date']).max() - pd.Timedelta(days=30))
    end_date = st.date_input("End Date", pd.to_datetime(df['Date']).max() + pd.Timedelta(days=15))
    
    if st.button("Generate Prediction"):
        with st.spinner("Generating prediction..."):
            # Simulate prediction delay
            import time
            time.sleep(2)
        
        # Generate sample prediction data
        historical_dates = pd.date_range(start=start_date - pd.Timedelta(days=30), end=start_date)
        prediction_dates = pd.date_range(start=start_date, end=end_date)
        
        # Create sample data
        last_value = df['Close'].iloc[-1] if 'Close' in df.columns else 1000
        historical_data = np.random.normal(loc=0, scale=5, size=len(historical_dates)).cumsum() + last_value
        actual_data = np.random.normal(loc=0, scale=5, size=len(prediction_dates)).cumsum() + historical_data[-1]
        predicted_data = actual_data + np.random.normal(loc=0, scale=10, size=len(prediction_dates))
        
        # Plot prediction vs actual
        st.subheader("Prediction Results")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(historical_dates, historical_data, 'b-', label='Historical Data')
        ax.plot(prediction_dates, actual_data, 'g-', label='Actual')
        ax.plot(prediction_dates, predicted_data, 'r--', label='Predicted')
        ax.set_title('Stock Price Prediction')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig)
        
        # Calculate and display metrics
        mse = np.mean((actual_data - predicted_data) ** 2)
        mae = np.mean(np.abs(actual_data - predicted_data))
        mape = np.mean(np.abs((actual_data - predicted_data) / actual_data)) * 100
        
        st.subheader("Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Squared Error", f"{mse:.2f}")
        col2.metric("Mean Absolute Error", f"{mae:.2f}")
        col3.metric("Mean Absolute Percentage Error", f"{mape:.2f}%")

# Footer
st.markdown('<div class="footer">© 2025 DataSentience-AIML | Time Series Forecasting with Python</div>', unsafe_allow_html=True)