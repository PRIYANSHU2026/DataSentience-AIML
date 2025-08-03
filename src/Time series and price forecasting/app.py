import streamlit as st
import os
import sys
import importlib.util

# Configure page
st.set_page_config(
    page_title="Time Series and Price Forecasting",
    page_icon="📈",
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
        transition: transform 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
    }
    .module-card {
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem;
        background-color: #f1f8e9;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
        cursor: pointer;
    }
    .module-card:hover {
        transform: translateY(-5px);
        background-color: #e8f5e9;
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
st.markdown('<div class="main-header">📈 Time Series and Price Forecasting</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced forecasting models for time series data and price prediction</div>', unsafe_allow_html=True)

# Available modules
modules = {
    "Time Series Forecasting with Python": {
        "description": "Time series forecasting using LSTM for stock price prediction based on historical data.",
        "image": "📊",
        "path": "Time Series Forecasting with Python"
    },
    "Time-Series LSTM Model": {
        "description": "Stock price prediction using LSTM neural networks with visualization of predictions vs actual values.",
        "image": "🧠",
        "path": "Time-Series LSTM Model"
    },
    "Titanic-Survival-Prediction": {
        "description": "Predict passenger survival on the Titanic using machine learning with 82.68% accuracy.",
        "image": "🚢",
        "path": "Titanic-Survival-Prediction"
    }
}

# Function to load and run module
def load_module(module_path, module_name):
    try:
        # Add module directory to path
        module_dir = os.path.dirname(module_path)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
            
        # Import the module
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            return False
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True
    except Exception as e:
        st.error(f"Error loading module: {str(e)}")
        return False

# Display module cards in a grid
col1, col2, col3 = st.columns(3)

with col1:
    with st.container():
        st.markdown(f'<div class="module-card"><h3>{modules["Time Series Forecasting with Python"]["image"]} Time Series Forecasting with Python</h3><p>{modules["Time Series Forecasting with Python"]["description"]}</p></div>', unsafe_allow_html=True)
        if st.button("Open Time Series Forecasting", key="btn1"):
            selected_module = "Time Series Forecasting with Python"
            module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), selected_module, "app.py")
            if os.path.exists(module_path):
                load_module(module_path, selected_module.replace(" ", "_"))
            else:
                st.info(f"Loading Time Series Forecasting module...")
                st.session_state.selected_module = selected_module

with col2:
    with st.container():
        st.markdown(f'<div class="module-card"><h3>{modules["Time-Series LSTM Model"]["image"]} Time-Series LSTM Model</h3><p>{modules["Time-Series LSTM Model"]["description"]}</p></div>', unsafe_allow_html=True)
        if st.button("Open LSTM Model", key="btn2"):
            selected_module = "Time-Series LSTM Model"
            module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), selected_module, "app.py")
            if os.path.exists(module_path):
                load_module(module_path, selected_module.replace(" ", "_"))
            else:
                st.info(f"Loading LSTM Model module...")
                st.session_state.selected_module = selected_module

with col3:
    with st.container():
        st.markdown(f'<div class="module-card"><h3>{modules["Titanic-Survival-Prediction"]["image"]} Titanic Survival Prediction</h3><p>{modules["Titanic-Survival-Prediction"]["description"]}</p></div>', unsafe_allow_html=True)
        if st.button("Open Titanic Prediction", key="btn3"):
            selected_module = "Titanic-Survival-Prediction"
            module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), selected_module, "app.py")
            if os.path.exists(module_path):
                load_module(module_path, selected_module.replace(" ", "_"))
            else:
                st.info(f"Loading Titanic Prediction module...")
                st.session_state.selected_module = selected_module

# Display selected module
if "selected_module" in st.session_state:
    selected_module = st.session_state.selected_module
    st.markdown(f"<h2>{selected_module}</h2>", unsafe_allow_html=True)
    
    # Try to load README for the module
    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), selected_module, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r") as f:
            readme_content = f.read()
        st.markdown(readme_content)
    
    # Display module-specific content
    if selected_module == "Time Series Forecasting with Python":
        st.write("### Stock Price Forecasting with LSTM")
        st.write("This module uses LSTM (Long Short-Term Memory) neural networks to predict stock prices based on historical data.")
        
        # Sample visualization
        st.image("https://miro.medium.com/max/1400/1*7xfLLsIeAEhHdOj4-F_GfA.png", caption="Sample LSTM Prediction vs Actual")
        
        # Interactive elements
        st.write("### Try the model with sample data")
        ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, GOOGL)", "GOOGL")
        period = st.slider("Prediction Period (days)", 7, 30, 14)
        
        if st.button("Generate Forecast"):
            st.info("Generating forecast... (This is a simulation)")
            st.success("Forecast generated successfully!")
            
            # Simulated forecast chart
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            
            # Generate sample data
            dates = pd.date_range(start='1/1/2023', periods=100)
            actual = np.random.normal(loc=100, scale=10, size=100).cumsum() + 1000
            predicted = actual[:-period] + np.random.normal(loc=0, scale=5, size=100-period)
            future = np.random.normal(loc=actual[-1], scale=5, size=period).cumsum() + actual[-1]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(dates[:-period], actual[:-period], label='Historical Data')
            ax.plot(dates[-period:], actual[-period:], 'b-', label='Actual')
            ax.plot(dates[-period:], future, 'r--', label='Predicted')
            ax.set_title(f'{ticker} Stock Price Forecast')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            
            st.pyplot(fig)
    
    elif selected_module == "Time-Series LSTM Model":
        st.write("### Google Stock Price Prediction using LSTM")
        st.write("This module demonstrates how to use LSTM neural networks for predicting Google's stock prices.")
        
        # Display sample images from the module
        lstm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Time-Series LSTM Model", "LSTM3-chain.png")
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Time-Series LSTM Model", "Output.png")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(lstm_path, caption="LSTM Architecture", use_column_width=True)
        with col2:
            st.image(output_path, caption="Prediction vs Actual", use_column_width=True)
        
        # Interactive elements
        st.write("### Model Parameters")
        epochs = st.slider("Training Epochs", 10, 100, 50)
        lookback = st.slider("Lookback Period (days)", 7, 60, 30)
        
        if st.button("Train Model"):
            st.info("Training model... (This is a simulation)")
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.05)
                progress.progress(i + 1)
            st.success("Model trained successfully!")
            
            # Display simulated MSE loss
            import numpy as np
            import matplotlib.pyplot as plt
            
            # Generate sample loss data
            epochs_range = range(1, epochs+1)
            train_loss = np.random.exponential(scale=0.1, size=epochs) + 0.05
            train_loss.sort()
            val_loss = train_loss + np.random.normal(loc=0.02, scale=0.01, size=epochs)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(epochs_range, train_loss, 'b', label='Training loss')
            ax.plot(epochs_range, val_loss, 'r', label='Validation loss')
            ax.set_title('Training and Validation Loss')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.legend()
            
            st.pyplot(fig)
    
    elif selected_module == "Titanic-Survival-Prediction":
        st.write("### Titanic Survival Prediction")
        st.write("Predict whether a passenger survived the Titanic disaster based on their characteristics.")
        
        # Interactive passenger information form
        st.write("### Enter Passenger Information")
        
        col1, col2 = st.columns(2)
        with col1:
            pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1 = 1st class, 2 = 2nd class, 3 = 3rd class")
            sex = st.radio("Gender", ["male", "female"])
            age = st.slider("Age", 0, 80, 30)
            sibsp = st.slider("Number of Siblings/Spouses Aboard", 0, 8, 0)
        
        with col2:
            parch = st.slider("Number of Parents/Children Aboard", 0, 6, 0)
            fare = st.slider("Fare (£)", 0, 512, 32)
            embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"], help="C = Cherbourg, Q = Queenstown, S = Southampton")
        
        if st.button("Predict Survival"):
            # Simulate prediction (in a real app, this would use the trained model)
            import random
            import time
            
            with st.spinner("Analyzing passenger data..."):
                time.sleep(2)
            
            # Simple heuristic based on historical data
            survival_chance = 0
            if sex == "female":
                survival_chance += 0.5
            if pclass == 1:
                survival_chance += 0.3
            elif pclass == 2:
                survival_chance += 0.2
            if age < 10:
                survival_chance += 0.3
            if fare > 100:
                survival_chance += 0.2
            
            # Add some randomness
            survival_chance = min(max(survival_chance + random.uniform(-0.1, 0.1), 0), 1)
            
            # Display prediction
            st.subheader("Prediction Result")
            col1, col2 = st.columns(2)
            
            with col1:
                if survival_chance > 0.5:
                    st.success(f"Survival Predicted: YES")
                else:
                    st.error(f"Survival Predicted: NO")
            
            with col2:
                st.metric("Survival Probability", f"{survival_chance:.2%}")
            
            # Display feature importance
            st.subheader("Feature Importance")
            features = ["Gender", "Class", "Age", "Fare", "Family Size", "Embarkation Port"]
            importances = [0.42, 0.28, 0.15, 0.08, 0.05, 0.02]
            
            import matplotlib.pyplot as plt
            import numpy as np
            
            fig, ax = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(len(features))
            ax.barh(y_pos, importances, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance for Survival Prediction')
            
            st.pyplot(fig)

# Footer
st.markdown('<div class="footer">© 2025 DataSentience-AIML | Time Series and Price Forecasting</div>', unsafe_allow_html=True)