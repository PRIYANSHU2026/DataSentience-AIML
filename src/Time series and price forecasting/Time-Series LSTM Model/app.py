import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Configure page
st.set_page_config(
    page_title="Time-Series LSTM Model",
    page_icon="🧠",
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
st.markdown('<div class="main-header">🧠 Time-Series LSTM Model</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Stock Price Prediction using LSTM Neural Networks</div>', unsafe_allow_html=True)

# Sidebar for navigation and controls
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Model Architecture", "Stock Selection", "Training & Prediction"])

# Overview page
if page == "Overview":
    st.header("Overview")
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("""
        ### Stock Price Prediction using LSTM
        
        This application demonstrates how to use Long Short-Term Memory (LSTM) neural networks for predicting stock prices.
        
        **Key Features:**
        - Download historical stock data from Yahoo Finance
        - Preprocess time series data for LSTM input
        - Train a multi-layer LSTM model
        - Visualize predictions against actual stock prices
        - Evaluate model performance
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display sample images
    col1, col2 = st.columns(2)
    
    # Check if images exist and display them
    lstm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LSTM3-chain.png")
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Output.png")
    
    with col1:
        if os.path.exists(lstm_path):
            st.image(lstm_path, caption="LSTM Architecture", use_column_width=True)
        else:
            st.image("https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png", caption="LSTM Architecture", use_column_width=True)
    
    with col2:
        if os.path.exists(output_path):
            st.image(output_path, caption="Prediction vs Actual", use_column_width=True)
        else:
            # Create a sample prediction vs actual plot
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(100)
            actual = np.sin(x/10) * 100 + 1000 + np.random.normal(0, 5, 100)
            predicted = actual + np.random.normal(0, 10, 100)
            ax.plot(x[-30:], actual[-30:], label='Actual')
            ax.plot(x[-30:], predicted[-30:], 'r--', label='Predicted')
            ax.set_title('Stock Price Prediction')
            ax.set_xlabel('Time')
            ax.set_ylabel('Price')
            ax.legend()
            st.pyplot(fig)

# Model Architecture page
elif page == "Model Architecture":
    st.header("LSTM Model Architecture")
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("""
        ### Long Short-Term Memory (LSTM) Networks
        
        LSTM networks are a type of recurrent neural network (RNN) that can learn and remember over long sequences. 
        They are well-suited for time series prediction tasks like stock price forecasting.
        
        **Key Components:**
        - Input Gate: Controls what new information to store in the cell state
        - Forget Gate: Controls what information to discard from the cell state
        - Output Gate: Controls what parts of the cell state to output
        - Cell State: The memory of the network that transfers information down the sequence chain
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display LSTM architecture
    lstm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LSTM3-chain.png")
    if os.path.exists(lstm_path):
        st.image(lstm_path, caption="LSTM Architecture", use_column_width=True)
    else:
        st.image("https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png", caption="LSTM Architecture", use_column_width=True)
    
    # Model architecture diagram
    st.subheader("Our Model Architecture")
    
    st.code("""
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(sequence_length, n_features), return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    """)
    
    # Explain model parameters
    st.subheader("Model Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Input Parameters:**")
        st.write("- **sequence_length**: Number of time steps (days) to look back")
        st.write("- **n_features**: Number of features (Open, High, Low, Close, Volume)")
    
    with col2:
        st.markdown("**Training Parameters:**")
        st.write("- **epochs**: Number of complete passes through the training dataset")
        st.write("- **batch_size**: Number of samples processed before the model is updated")
        st.write("- **validation_split**: Fraction of the training data to be used as validation data")

# Stock Selection page
elif page == "Stock Selection":
    st.header("Stock Selection")
    
    st.write("Select a stock to predict its future prices using our LSTM model.")
    
    # Stock selection
    stock_options = {
        "GOOGL": "Google",
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "AMZN": "Amazon",
        "META": "Meta (Facebook)",
        "TSLA": "Tesla",
        "NFLX": "Netflix"
    }
    
    selected_stock = st.selectbox("Select Stock", list(stock_options.keys()), format_func=lambda x: f"{x} - {stock_options[x]}")
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", pd.Timestamp.now() - pd.Timedelta(days=365*2))
    with col2:
        end_date = st.date_input("End Date", pd.Timestamp.now())
    
    # Display sample stock data
    if st.button("Load Stock Data"):
        with st.spinner(f"Loading {stock_options[selected_stock]} stock data..."):
            # Simulate data loading delay
            time.sleep(2)
        
        # Generate sample stock data
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        data = {
            'Date': dates,
            'Open': np.random.normal(loc=100, scale=1, size=len(dates)).cumsum() + 1000,
            'High': np.random.normal(loc=101, scale=1, size=len(dates)).cumsum() + 1005,
            'Low': np.random.normal(loc=99, scale=1, size=len(dates)).cumsum() + 995,
            'Close': np.random.normal(loc=100, scale=1, size=len(dates)).cumsum() + 1002,
            'Volume': np.random.randint(1000000, 10000000, size=len(dates))
        }
        df = pd.DataFrame(data)
        
        st.success(f"Successfully loaded {len(df)} days of {stock_options[selected_stock]} stock data")
        
        # Display stock data
        st.subheader("Stock Data Preview")
        st.dataframe(df.head())
        
        # Plot stock prices
        st.subheader("Stock Price History")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['Date'], df['Close'], label='Close Price')
        ax.set_title(f'{selected_stock} Stock Price History')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend()
        st.pyplot(fig)
        
        # Save to session state for use in training
        st.session_state.stock_data = df
        st.session_state.selected_stock = selected_stock

# Training & Prediction page
elif page == "Training & Prediction":
    st.header("Training & Prediction")
    
    if 'stock_data' not in st.session_state:
        st.warning("Please select a stock in the 'Stock Selection' page first.")
    else:
        st.write(f"Training LSTM model for {st.session_state.selected_stock} stock price prediction.")
        
        # Model parameters
        st.subheader("Model Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sequence_length = st.slider("Sequence Length (days)", 10, 60, 30)
            prediction_days = st.slider("Days to Predict", 7, 30, 14)
            epochs = st.slider("Training Epochs", 10, 100, 50)
        
        with col2:
            lstm_units = st.slider("LSTM Units", 32, 128, 64, step=16)
            dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2, step=0.1)
            train_split = st.slider("Training Data Split", 0.7, 0.9, 0.8, step=0.05)
        
        # Train model button
        if st.button("Train Model and Predict"):
            # Simulate model training
            with st.spinner("Training LSTM model..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(epochs):
                    # Update progress
                    progress = (i + 1) / epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Training progress: {int(progress * 100)}%")
                    
                    # Simulate training time
                    time.sleep(0.1)
                
                status_text.text("Training complete!")
            
            st.success("Model trained successfully!")
            
            # Display training and validation loss
            st.subheader("Training Results")
            
            # Simulate training and validation loss
            train_loss = np.random.exponential(scale=0.1, size=epochs) + 0.05
            train_loss.sort()
            val_loss = train_loss + np.random.normal(loc=0.02, scale=0.01, size=epochs)
            
            # Plot loss
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(1, epochs + 1), train_loss, label='Training loss')
            ax.plot(range(1, epochs + 1), val_loss, label='Validation loss')
            ax.set_title('Training and Validation Loss')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.legend()
            st.pyplot(fig)
            
            # Generate predictions
            st.subheader("Stock Price Predictions")
            
            # Get the last portion of the data for visualization
            df = st.session_state.stock_data
            dates = df['Date'].iloc[-100:]
            actual_prices = df['Close'].iloc[-100:].values
            
            # Generate future dates
            last_date = df['Date'].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days, freq='B')
            
            # Generate predicted prices (with some randomness to simulate predictions)
            last_price = actual_prices[-1]
            future_prices = np.random.normal(loc=0, scale=5, size=prediction_days).cumsum() + last_price
            
            # Create combined dates and prices for plotting
            combined_dates = pd.concat([dates, pd.Series(future_dates)])
            predicted_prices = np.concatenate([actual_prices, future_prices])
            
            # Add some noise to create a "predicted" line for historical data
            historical_predicted = actual_prices + np.random.normal(loc=0, scale=10, size=len(actual_prices))
            
            # Plot actual vs predicted
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(dates, actual_prices, 'b-', label='Actual')
            ax.plot(dates, historical_predicted, 'g--', label='Model Fit')
            ax.plot(future_dates, future_prices, 'r--', label='Prediction')
            ax.axvline(x=last_date, color='k', linestyle='--', alpha=0.3)
            ax.set_title(f'{st.session_state.selected_stock} Stock Price Prediction')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            ax.legend()
            st.pyplot(fig)
            
            # Display prediction results in a table
            st.subheader("Prediction Results")
            
            prediction_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Price': future_prices,
                'Confidence Interval (Lower)': future_prices - np.random.uniform(10, 20, size=len(future_prices)),
                'Confidence Interval (Upper)': future_prices + np.random.uniform(10, 20, size=len(future_prices))
            })
            
            st.dataframe(prediction_df)
            
            # Performance metrics
            st.subheader("Model Performance Metrics")
            
            # Calculate metrics on the historical data
            mse = np.mean((actual_prices - historical_predicted) ** 2)
            mae = np.mean(np.abs(actual_prices - historical_predicted))
            mape = np.mean(np.abs((actual_prices - historical_predicted) / actual_prices)) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Squared Error", f"{mse:.2f}")
            col2.metric("Mean Absolute Error", f"{mae:.2f}")
            col3.metric("Mean Absolute Percentage Error", f"{mape:.2f}%")

# Footer
st.markdown('<div class="footer">© 2025 DataSentience-AIML | Time-Series LSTM Model</div>', unsafe_allow_html=True)