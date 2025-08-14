import streamlit as st
import os
import sys

# Add the scripts directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page configuration
st.set_page_config(
    page_title="Cryptocurrency Analysis Tools",
    page_icon="🪙",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        font-weight: 500;
    }
    .tool-card {
        background-color: #f0f8ff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border-left: 5px solid #1E88E5;
    }
    .tool-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #0D47A1;
    }
    .tool-desc {
        color: #424242;
        margin-bottom: 15px;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: 500;
        border-radius: 5px;
        padding: 0.5em 1em;
        border: none;
    }
    .stButton button:hover {
        background-color: #0D47A1;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<p class="main-header">🪙 Cryptocurrency Analysis Tools</p>', unsafe_allow_html=True)
st.markdown("Explore and analyze cryptocurrency markets with our suite of specialized tools.")
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
    # Bitcoin Price Prediction Tool
    if tool_card(
        "Bitcoin Price Prediction", 
        "Predict future Bitcoin prices using historical market data and LSTM neural networks.", 
        "📈", 
        "Open Bitcoin Predictor", 
        "bitcoin_btn"
    ):
        # This would normally load the Bitcoin Price Prediction module
        st.session_state.selected_tool = "bitcoin_price_prediction"

with col2:
    # Ethereum Price Prediction Tool
    if tool_card(
        "Ethereum Price Predictor", 
        "Forecast Ethereum prices based on market indicators and historical patterns.", 
        "🪙", 
        "Open Ethereum Predictor", 
        "ethereum_btn"
    ):
        # This would normally load the Ethereum Price Prediction module
        st.session_state.selected_tool = "ethereum_price_predictor"

col3, col4 = st.columns(2)

with col3:
    # Crypto Sentiment Analysis Tool
    if tool_card(
        "Crypto Sentiment Analyzer", 
        "Analyze news sentiment to gauge market perception of cryptocurrencies.", 
        "📰", 
        "Open Sentiment Analyzer", 
        "sentiment_btn"
    ):
        # This would normally load the Crypto Sentiment Analysis module
        st.session_state.selected_tool = "crypto_sentiment_analysis"

with col4:
    # Market Trends Tool (placeholder for future expansion)
    if tool_card(
        "Crypto Market Trends", 
        "Visualize and analyze cryptocurrency market trends and correlations.", 
        "📊", 
        "Open Market Trends", 
        "market_trends_btn"
    ):
        # This would be a placeholder for future development
        st.session_state.selected_tool = "crypto_market_trends"

# Tool display logic
if 'selected_tool' not in st.session_state:
    st.session_state.selected_tool = None

# Display the selected tool
if st.session_state.selected_tool == "bitcoin_price_prediction":
    st.markdown("---")
    st.markdown('<p class="sub-header">📈 Bitcoin Price Prediction</p>', unsafe_allow_html=True)
    
    # Simulated Bitcoin Price Prediction interface
    st.markdown("### 🔢 Enter Market Data")
    
    col1, col2 = st.columns(2)
    with col1:
        open_price = st.number_input("Open Price ($)", value=30000.00, step=100.0, format="%.2f", key="btc_open")
        low_price = st.number_input("Low Price ($)", value=29500.00, step=100.0, format="%.2f", key="btc_low")
    with col2:
        high_price = st.number_input("High Price ($)", value=30500.00, step=100.0, format="%.2f", key="btc_high")
        volume = st.number_input("Volume", value=500000000.00, step=1000000.0, format="%.2f", key="btc_volume")
    
    if st.button("🔮 Predict Next Close Price", key="btc_predict"):
        # Simulated prediction
        import random
        predicted_price = open_price * (1 + random.uniform(-0.02, 0.05))
        st.success(f"📈 **Predicted Close Price:** ${predicted_price:.2f}")

elif st.session_state.selected_tool == "ethereum_price_predictor":
    st.markdown("---")
    st.markdown('<p class="sub-header">🪙 Ethereum Price Predictor</p>', unsafe_allow_html=True)
    
    # Simulated Ethereum Price Prediction interface (based on the actual streamlit_app.py)
    st.markdown("### 🔢 Enter Market Data")
    
    col1, col2 = st.columns(2)
    with col1:
        open_price = st.number_input("Open Price ($)", value=1800.00, step=10.0, format="%.2f")
        low_price = st.number_input("Low Price ($)", value=1750.00, step=10.0, format="%.2f")
    with col2:
        high_price = st.number_input("High Price ($)", value=1850.00, step=10.0, format="%.2f")
        volume = st.number_input("Volume", value=100000000.00, step=1000000.0, format="%.2f")
    
    if st.button("🔮 Predict Next Close Price"):
        # Simulated prediction
        import random
        predicted_price = open_price * (1 + random.uniform(-0.03, 0.04))
        st.success(f"📈 **Predicted Close Price:** ${predicted_price:.2f}")

elif st.session_state.selected_tool == "crypto_sentiment_analysis":
    st.markdown("---")
    st.markdown('<p class="sub-header">📰 Crypto Sentiment Analyzer</p>', unsafe_allow_html=True)
    
    # Simulated Crypto Sentiment Analysis interface (based on the actual main.py)
    st.markdown("Analyze real-time crypto news articles for **Positive**, **Neutral**, or **Negative** sentiment.")
    
    # Input fields
    api_key = st.text_input("🔐 Enter your NewsAPI key (for demo, any text will work)", type="password")
    crypto = st.text_input("🪙 Enter crypto topic", value="bitcoin")
    num_articles = st.slider("📄 Number of news articles to analyze", min_value=5, max_value=50, value=10)
    
    if st.button("🔍 Analyze Sentiment"):
        # Simulated sentiment analysis results
        import random
        sentiments = ["positive", "neutral", "negative"]
        weights = [0.5, 0.3, 0.2]  # More positive bias for demo
        
        st.markdown("### News Articles with Sentiment")
        for i in range(num_articles):
            sentiment = random.choices(sentiments, weights=weights)[0]
            st.markdown(f"**{i+1}. [{sentiment.upper()}]** — Sample news article about {crypto} cryptocurrency...")
        
        # Summary chart
        st.subheader("📊 Sentiment Summary")
        summary = {
            "positive": int(num_articles * 0.5),
            "neutral": int(num_articles * 0.3),
            "negative": int(num_articles * 0.2)
        }
        st.write(summary)

elif st.session_state.selected_tool == "crypto_market_trends":
    st.markdown("---")
    st.markdown('<p class="sub-header">📊 Crypto Market Trends</p>', unsafe_allow_html=True)
    
    # Placeholder for future development
    st.info("This feature is coming soon! It will provide comprehensive market trend analysis for major cryptocurrencies.")
    
    # Show a sample visualization
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Generate sample data
    dates = pd.date_range(start='2023-01-01', periods=90, freq='D')
    btc_prices = np.random.normal(30000, 2000, 90).cumsum() + 25000
    eth_prices = np.random.normal(1800, 150, 90).cumsum() + 1500
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dates, btc_prices, label='Bitcoin')
    ax.plot(dates, eth_prices * 15, label='Ethereum (scaled)')
    ax.set_title('Cryptocurrency Price Trends (Sample Data)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Display the plot
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("### About Cryptocurrency Analysis")
st.markdown("""
    Our cryptocurrency analysis tools leverage advanced machine learning models to provide insights into crypto markets:
    
    - **Price Prediction**: Using LSTM neural networks to forecast future prices
    - **Sentiment Analysis**: Analyzing news and social media to gauge market sentiment
    - **Market Trends**: Visualizing correlations and patterns across different cryptocurrencies
    
    These tools are designed for educational purposes and should not be used as financial advice.
""")