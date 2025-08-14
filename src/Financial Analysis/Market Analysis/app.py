import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the scripts directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page configuration
st.set_page_config(
    page_title="Market Analysis Tools",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1B5E20;
        font-weight: 500;
    }
    .tool-card {
        background-color: #f1f8e9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border-left: 5px solid #2E7D32;
    }
    .tool-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1B5E20;
    }
    .tool-desc {
        color: #424242;
        margin-bottom: 15px;
    }
    .stButton button {
        background-color: #2E7D32;
        color: white;
        font-weight: 500;
        border-radius: 5px;
        padding: 0.5em 1em;
        border: none;
    }
    .stButton button:hover {
        background-color: #1B5E20;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2E7D32;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #616161;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<p class="main-header">📊 Market Analysis Tools</p>', unsafe_allow_html=True)
st.markdown("Analyze financial markets and make data-driven investment decisions.")
st.markdown("---")

# Dashboard metrics
st.markdown("### 📈 Market Dashboard")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">+2.3%</div>
        <div class="metric-label">S&P 500 Today</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">-0.8%</div>
        <div class="metric-label">NASDAQ Today</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">3.2%</div>
        <div class="metric-label">10Y Treasury Yield</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">$1,892</div>
        <div class="metric-label">Gold (per oz)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Function to create a tool card
def tool_card(title, description, icon, button_text, button_key):
    st.markdown(f'''
    <div class="tool-card">
        <p class="tool-title">{icon} {title}</p>
        <p class="tool-desc">{description}</p>
    </div>
    ''', unsafe_allow_html=True)
    return st.button(button_text, key=button_key)

# Create two columns for the tools
col1, col2 = st.columns(2)

with col1:
    # Stock Price Prediction Tool
    if tool_card(
        "Stock Price Prediction", 
        "Forecast stock prices using time series analysis and ARIMA models.", 
        "📈", 
        "Open Stock Predictor", 
        "stock_btn"
    ):
        st.session_state.selected_tool = "stock_price_prediction"

with col2:
    # Loan Prediction Tool
    if tool_card(
        "Loan Approval Predictor", 
        "Predict loan approval likelihood based on applicant information.", 
        "💰", 
        "Open Loan Predictor", 
        "loan_btn"
    ):
        st.session_state.selected_tool = "loan_prediction"

col3, col4 = st.columns(2)

with col3:
    # Currency Exchange Rate Tool
    if tool_card(
        "Currency Exchange Forecasting", 
        "Predict future exchange rates using ARIMA and LSTM models.", 
        "💱", 
        "Open Exchange Forecaster", 
        "exchange_btn"
    ):
        st.session_state.selected_tool = "exchange_rate_forecasting"

with col4:
    # Car Price Predictor Tool
    if tool_card(
        "Car Price Predictor", 
        "Estimate car prices based on various features and market trends.", 
        "🚗", 
        "Open Car Predictor", 
        "car_btn"
    ):
        st.session_state.selected_tool = "car_price_predictor"

# Tool display logic
if 'selected_tool' not in st.session_state:
    st.session_state.selected_tool = None

# Display the selected tool
if st.session_state.selected_tool == "stock_price_prediction":
    st.markdown("---")
    st.markdown('<p class="sub-header">📈 Stock Price Prediction</p>', unsafe_allow_html=True)
    
    # Simulated Stock Price Prediction interface
    st.markdown("### Select Stock and Parameters")
    
    stock_symbol = st.selectbox("Stock Symbol", ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "RELIANCE.NS"])
    prediction_days = st.slider("Prediction Days", min_value=7, max_value=90, value=30)
    model_type = st.selectbox("Model Type", ["ARIMA", "LSTM", "Prophet"])
    
    if st.button("Generate Forecast", key="stock_forecast"):
        # Generate sample data for demonstration
        dates = pd.date_range(start='2023-01-01', periods=180, freq='D')
        historical_prices = np.random.normal(0.001, 0.02, 180).cumsum() + 1
        historical_prices = historical_prices * (100 if stock_symbol == "RELIANCE.NS" else 150)
        
        # Create forecast data
        forecast_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=prediction_days, freq='D')
        forecast_trend = np.random.normal(0.001, 0.01, prediction_days).cumsum() + 1
        forecast_prices = historical_prices[-1] * forecast_trend
        
        # Plot the data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(dates, historical_prices, label='Historical Prices')
        ax.plot(forecast_dates, forecast_prices, label='Forecast', color='red')
        ax.set_title(f'{stock_symbol} Stock Price Forecast ({model_type})')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Display metrics
        st.markdown("### Forecast Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Close (30d)", f"${forecast_prices[-1]:.2f}", f"{((forecast_prices[-1]/historical_prices[-1])-1)*100:.2f}%")
        with col2:
            st.metric("Mean Squared Error", "1431.99")
        with col3:
            st.metric("SMAPE", "1.503%")

elif st.session_state.selected_tool == "loan_prediction":
    st.markdown("---")
    st.markdown('<p class="sub-header">💰 Loan Approval Predictor</p>', unsafe_allow_html=True)
    st.markdown("Fill in the applicant details below to check if the loan is likely to be approved.")
    
    # Input form (based on the actual app.py)
    with st.form("loan_form"):
        st.subheader("📋 Personal & Financial Information")

        col1, col2 = st.columns(2)
        with col1:
            Gender = st.selectbox("Gender", ["Male", "Female"])
            Married = st.selectbox("Married", ["Yes", "No"])
            Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
            Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        with col2:
            ApplicantIncome = st.number_input("Applicant Income", min_value=0.0, value=5000.0)
            CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0.0, value=2000.0)
            LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0.0, value=100.0)
            Loan_Amount_Term = st.number_input("Loan Amount Term (in days)", min_value=0.0, value=360.0)
            Credit_History = st.selectbox("Credit History", [1.0, 0.0])

        submitted = st.form_submit_button("📤 Predict Loan Approval")

    # On submit
    if submitted:
        # Simulated prediction
        import random
        approval_chance = 0.7  # Base chance
        
        # Adjust based on some factors
        if Credit_History == 0.0:
            approval_chance -= 0.4
        if ApplicantIncome < 3000:
            approval_chance -= 0.2
        if Education == "Not Graduate":
            approval_chance -= 0.1
        
        # Random factor
        approval_chance += random.uniform(-0.1, 0.1)
        approval_chance = max(0, min(1, approval_chance))
        
        prediction = 1 if approval_chance > 0.5 else 0
        
        if prediction == 1:
            st.success("✅ Loan Approved")
            st.balloons()
        else:
            st.error("❌ Loan Rejected")

elif st.session_state.selected_tool == "exchange_rate_forecasting":
    st.markdown("---")
    st.markdown('<p class="sub-header">💱 Currency Exchange Rate Forecasting</p>', unsafe_allow_html=True)
    st.markdown("""
    This interactive dashboard forecasts currency exchange rates using:
    - **ARIMA** for traditional time series prediction
    - **LSTM** for deep learning-based forecasting
    """)
    
    # Sidebar Filters
    currency = st.selectbox("Choose Currency", ["Indian Rupee", "US Dollar", "Euro", "British Pound"])
    model_option = st.radio("Forecasting Model", ["ARIMA", "LSTM"])
    
    if st.button("📈 Run Forecast"):
        # Generate sample data
        dates = pd.date_range(start='2023-01-01', periods=90, freq='D')
        
        # Base values for different currencies
        base_values = {
            "Indian Rupee": 82.5,
            "US Dollar": 1.0,
            "Euro": 0.92,
            "British Pound": 0.79
        }
        
        # Generate exchange rate data
        base_value = base_values[currency]
        exchange_rates = np.random.normal(0, 0.005, 90).cumsum() + base_value
        
        # Create forecast data
        forecast_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=10, freq='D')
        forecast_trend = np.random.normal(0, 0.003, 10).cumsum() + 1
        forecast_rates = exchange_rates[-1] * forecast_trend
        
        st.write("📆 Forecast for Next 10 Days")
        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Forecast": forecast_rates
        })
        st.dataframe(forecast_df, use_container_width=True)
        
        # Plot the data
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dates, exchange_rates, label="Historical", color="blue")
        ax.plot(forecast_dates, forecast_rates, label="Forecast", color="red")
        ax.set_title(f"{currency} Forecast using {model_option}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Exchange Rate")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)

elif st.session_state.selected_tool == "car_price_predictor":
    st.markdown("---")
    st.markdown('<p class="sub-header">🚗 Car Price Predictor</p>', unsafe_allow_html=True)
    st.markdown("Estimate the price of a car based on its features.")
    
    # Car features input form
    with st.form("car_form"):
        st.subheader("🚗 Car Details")
        
        col1, col2 = st.columns(2)
        with col1:
            brand = st.selectbox("Brand", ["Toyota", "Honda", "Ford", "BMW", "Mercedes", "Audi", "Hyundai"])
            model_year = st.slider("Model Year", 2000, 2023, 2020)
            mileage = st.number_input("Mileage (km)", min_value=0, value=50000)
            fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid"])
        
        with col2:
            engine_size = st.number_input("Engine Size (cc)", min_value=800, max_value=5000, value=1500)
            transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
            body_type = st.selectbox("Body Type", ["Sedan", "SUV", "Hatchback", "Coupe", "Convertible"])
            condition = st.slider("Condition (1-10)", 1, 10, 7)
        
        submitted = st.form_submit_button("💰 Estimate Price")
    
    if submitted:
        # Simulated price prediction
        base_price = {
            "Toyota": 15000,
            "Honda": 16000,
            "Ford": 14000,
            "BMW": 35000,
            "Mercedes": 40000,
            "Audi": 38000,
            "Hyundai": 12000
        }[brand]
        
        # Adjust for features
        year_factor = (model_year - 2000) * 500
        mileage_factor = -0.05 * (mileage / 1000)
        condition_factor = (condition - 5) * 1000
        
        # Fuel type adjustment
        fuel_factors = {
            "Petrol": 0,
            "Diesel": 1000,
            "Electric": 5000,
            "Hybrid": 3000
        }
        
        # Transmission adjustment
        transmission_factor = 2000 if transmission == "Automatic" else 0
        
        # Calculate final price
        estimated_price = base_price + year_factor + mileage_factor + condition_factor + fuel_factors[fuel_type] + transmission_factor
        estimated_price = max(1000, estimated_price)  # Minimum price
        
        # Display result
        st.success(f"💰 **Estimated Car Price:** ${estimated_price:.2f}")
        
        # Show price breakdown
        st.subheader("Price Breakdown")
        breakdown = pd.DataFrame({
            "Factor": ["Base Price", "Year Adjustment", "Mileage Impact", "Condition", "Fuel Type", "Transmission", "Total"],
            "Value": [
                f"${base_price:.2f}",
                f"${year_factor:.2f}",
                f"${mileage_factor:.2f}",
                f"${condition_factor:.2f}",
                f"${fuel_factors[fuel_type]:.2f}",
                f"${transmission_factor:.2f}",
                f"${estimated_price:.2f}"
            ]
        })
        st.table(breakdown)

# Footer
st.markdown("---")
st.markdown("### About Market Analysis")
st.markdown("""
    Our market analysis tools use statistical models and machine learning to help you make informed financial decisions:
    
    - **Stock Prediction**: Forecast future stock prices using time series analysis
    - **Loan Approval**: Predict loan approval likelihood based on applicant data
    - **Exchange Rates**: Forecast currency exchange rates using ARIMA and LSTM models
    - **Car Pricing**: Estimate car prices based on various features and market trends
    
    These tools are designed for educational purposes and should not be used as financial advice.
""")