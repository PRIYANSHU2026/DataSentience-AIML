import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64

# Configure page
st.set_page_config(
    page_title="Financial Analysis Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1.5rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #0277BD;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-left: 4px solid #1E88E5;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .card-business {
        border-left: 4px solid #43A047;
    }
    .card-credit {
        border-left: 4px solid #FB8C00;
    }
    .card-crypto {
        border-left: 4px solid #8E24AA;
    }
    .card-market {
        border-left: 4px solid #E53935;
    }
    .card-title {
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        color: #37474F;
    }
    .card-description {
        color: #546E7A;
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        font-size: 0.8rem;
        color: #78909C;
        border-top: 1px solid #eceff1;
    }
    .metric-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e7eb 100%);
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #546E7A;
        margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">💰 Financial Analysis Dashboard</div>', unsafe_allow_html=True)

# Function to create a card with hover effect
def create_card(title, description, key, icon, card_class=""):
    st.markdown(f'''
    <div class="card {card_class}" id="{key}-card">
        <div class="card-title">{icon} {title}</div>
        <div class="card-description">{description}</div>
    </div>
    ''', unsafe_allow_html=True)
    return st.button(f"Explore {title}", key=key)

# Main content - Dashboard Overview
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Key Financial Metrics")
    
    # Sample metrics - these would be replaced with real data in a production app
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">$2.4T</div>
            <div class="metric-label">Global Market Cap</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">+4.2%</div>
            <div class="metric-label">Market Growth</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col3:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">$48K</div>
            <div class="metric-label">BTC Price</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Sample chart
    st.markdown("### Market Trends")
    
    # Generate sample data for demonstration
    dates = pd.date_range(start='2023-01-01', periods=90, freq='D')
    market_data = pd.DataFrame({
        'Date': dates,
        'S&P 500': 4000 + np.cumsum(np.random.normal(0.5, 10, 90)),
        'NASDAQ': 12000 + np.cumsum(np.random.normal(1.5, 30, 90)),
        'DOW': 33000 + np.cumsum(np.random.normal(0.3, 20, 90))
    })
    
    fig = px.line(market_data, x='Date', y=['S&P 500', 'NASDAQ', 'DOW'],
                 title='Market Index Performance (Last 90 Days)')
    fig.update_layout(height=400, legend_title_text='Index')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Financial Analysis Modules")
    
    # Business Analytics
    if create_card(
        "Business Analytics", 
        "Advanced analytics for business performance, startup evaluation, and profit prediction.", 
        "business", 
        "📊",
        "card-business"
    ):
        st.session_state.selected_module = "Business Analytics"
    
    # Credit Card Eligibility
    if create_card(
        "Credit Card Eligibility", 
        "Smart eligibility assessment with personalized card recommendations and financial analysis.", 
        "credit", 
        "💳",
        "card-credit"
    ):
        st.session_state.selected_module = "Credit Card Eligibility Checker"
    
    # Cryptocurrency
    if create_card(
        "Cryptocurrency Analysis", 
        "Bitcoin and Ethereum price prediction, crypto sentiment analysis, and market trends.", 
        "crypto", 
        "🪙",
        "card-crypto"
    ):
        st.session_state.selected_module = "Cryptocurrency"
    
    # Market Analysis
    if create_card(
        "Market Analysis", 
        "Stock price prediction, GDP forecasting, loan eligibility, and insurance risk assessment.", 
        "market", 
        "📈",
        "card-market"
    ):
        st.session_state.selected_module = "Market Analysis"

# Initialize session state for module selection if not already set
if 'selected_module' not in st.session_state:
    st.session_state.selected_module = None

# Display selected module content
if st.session_state.selected_module:
    st.markdown(f"<div class='sub-header'>{st.session_state.selected_module}</div>", unsafe_allow_html=True)
    
    # Business Analytics Module
    if st.session_state.selected_module == "Business Analytics":
        st.markdown("### Business Analytics Tools")
        
        ba_col1, ba_col2 = st.columns(2)
        
        with ba_col1:
            if create_card("Startup Success Predictor", "ML-powered platform to evaluate startup potential and success factors.", "startup_success", "🚀"):
                # This would load the specific module in a production app
                st.info("Loading Startup Success Predictor module...")
                
            if create_card("Employee Attrition Prediction", "Predict and analyze factors affecting employee retention.", "employee_attrition", "👥"):
                st.info("Loading Employee Attrition Prediction module...")
        
        with ba_col2:
            if create_card("Startup Profit Prediction", "Analyze and predict startup profit based on key business metrics.", "profit_prediction", "💰"):
                st.info("Loading Startup Profit Prediction module...")
                
            if create_card("Statement Analysis", "Analyze financial statements and news sentiment for market impact.", "statement_analysis", "📑"):
                st.info("Loading Statement Analysis module...")
    
    # Credit Card Eligibility Module
    elif st.session_state.selected_module == "Credit Card Eligibility Checker":
        st.markdown("### Credit Card Eligibility Assessment")
        
        # Simplified version of the credit card eligibility checker
        with st.form("eligibility_form"):
            cc_col1, cc_col2 = st.columns(2)
            
            with cc_col1:
                age = st.number_input("Age", min_value=18, max_value=80, value=30)
                income = st.number_input("Monthly Income (₹)", min_value=10000, value=50000)
                employment = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Student", "Unemployed"])
            
            with cc_col2:
                experience = st.slider("Work Experience (Years)", 0, 30, 5)
                existing_cards = st.number_input("Existing Credit Cards", 0, 10, 1)
                city_tier = st.selectbox("City Tier", ["Tier 1 (Metro)", "Tier 2", "Tier 3"])
            
            submitted = st.form_submit_button("Check Eligibility")
        
        if submitted:
            # Simulate eligibility check
            if age >= 21 and income >= 20000 and employment in ["Salaried", "Self-employed"]:
                st.success("✅ Congratulations! You are eligible for a credit card.")
                
                # Credit score gauge
                score = min(850, 300 + age + (income/1000) + (experience*10))
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Estimated Credit Score"},
                    gauge = {
                        'axis': {'range': [300, 850]},
                        'bar': {'color': "royalblue"},
                        'steps': [
                            {'range': [300, 580], 'color': "#EF5350"},
                            {'range': [580, 670], 'color': "#FFCA28"},
                            {'range': [670, 740], 'color': "#66BB6A"},
                            {'range': [740, 850], 'color': "#43A047"}
                        ]
                    }
                ))
                st.plotly_chart(fig)
                
                # Card recommendations based on income
                st.subheader("Recommended Cards")
                if income >= 200000:
                    st.markdown("🌟 **Platinum Tier** - Premium benefits and high credit limits")
                elif income >= 100000:
                    st.markdown("✨ **Gold Tier** - Enhanced rewards and good benefits")
                elif income >= 50000:
                    st.markdown("👍 **Silver Tier** - Standard benefits and cashback options")
                else:
                    st.markdown("📝 **Basic Tier** - Entry-level features for building credit")
            else:
                st.error("❌ Sorry, you don't meet the minimum eligibility criteria.")
                st.info("Minimum requirements: Age 21+, Income ₹20,000+, Employed status")
    
    # Cryptocurrency Module
    elif st.session_state.selected_module == "Cryptocurrency":
        st.markdown("### Cryptocurrency Analysis Tools")
        
        crypto_col1, crypto_col2 = st.columns(2)
        
        with crypto_col1:
            if create_card("Bitcoin Price Prediction", "LSTM-based prediction of Bitcoin prices using historical data.", "btc_prediction", "₿"):
                st.info("Loading Bitcoin Price Prediction module...")
                
            if create_card("Ethereum Price Predictor", "Advanced forecasting for Ethereum cryptocurrency prices.", "eth_prediction", "Ξ"):
                st.info("Loading Ethereum Price Predictor module...")
        
        with crypto_col2:
            if create_card("Crypto Sentiment Analysis", "Analyze market sentiment from news and social media for crypto assets.", "crypto_sentiment", "🔍"):
                st.info("Loading Crypto Sentiment Analysis module...")
                
            # Demo crypto price chart
            st.subheader("Bitcoin Price Trend (Sample)")
            
            # Generate sample Bitcoin price data
            dates = pd.date_range(start='2023-01-01', periods=180, freq='D')
            btc_price = 30000 + np.cumsum(np.random.normal(50, 800, 180))
            crypto_data = pd.DataFrame({
                'Date': dates,
                'Price': btc_price
            })
            
            fig = px.line(crypto_data, x='Date', y='Price',
                         title='Bitcoin Price (USD)')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Market Analysis Module
    elif st.session_state.selected_module == "Market Analysis":
        st.markdown("### Market Analysis Tools")
        
        market_col1, market_col2 = st.columns(2)
        
        with market_col1:
            if create_card("Stock Price Analysis", "Analyze and predict stock market prices using time series forecasting.", "stock_analysis", "📈"):
                st.info("Loading Stock Price Analysis module...")
                
            if create_card("GDP Prediction", "Economic forecasting model for GDP prediction and analysis.", "gdp_prediction", "🏛️"):
                st.info("Loading GDP Prediction module...")
                
            if create_card("Car Price Predictor", "ML model to predict car prices based on various features.", "car_price", "🚗"):
                st.info("Loading Car Price Predictor module...")
        
        with market_col2:
            if create_card("Loan Prediction", "Predict loan approval probability based on applicant information.", "loan_prediction", "💵"):
                st.info("Loading Loan Prediction module...")
                
            if create_card("Insurance Predictor", "Predict insurance premiums based on customer profiles and history.", "insurance_predictor", "🏥"):
                st.info("Loading Insurance Predictor module...")
                
            if create_card("Mutual Fund Risk Predictor", "Assess and predict risk levels for mutual fund investments.", "mutual_fund", "📊"):
                st.info("Loading Mutual Fund Risk Predictor module...")

# Footer
st.markdown('<div class="footer">© 2025 DataSentience-AIML | Financial Analysis Module</div>', unsafe_allow_html=True)
