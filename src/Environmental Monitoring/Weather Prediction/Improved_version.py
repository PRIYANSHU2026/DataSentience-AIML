# Importing libraries with standard conventions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from pmdarima.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose

# Setting styles
plt.style.use('fivethirtyeight')
sns.set(style='whitegrid', palette='muted')

# Constants
TEST_SIZE = 0.2  # For train-test split
RANDOM_STATE = 42  # For reproducibility

def load_and_preprocess_data(filepath):
    """Load and preprocess the time series data."""
    # Load data with proper datetime parsing
    df = pd.read_csv(filepath, 
                    parse_dates=['datetime_utc'], 
                    index_col='datetime_utc')
    
    # Forward fill missing values
    df.ffill(inplace=True)
    
    # Resample to daily frequency to handle irregular timestamps
    df = df.resample('D').mean()
    
    # Remove outliers using IQR method
    Q1 = df['dewpoint'].quantile(0.25)
    Q3 = df['dewpoint'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['dewpoint'] >= (Q1 - 1.5 * IQR)) & 
           (df['dewpoint'] <= (Q3 + 1.5 * IQR))]
    
    return df

def plot_time_series(df, title='Time Series Data'):
    """Plot the time series with rolling statistics."""
    rolling_mean = df.rolling(window=30).mean()
    rolling_std = df.rolling(window=30).std()
    
    plt.figure(figsize=(14, 7))
    plt.plot(df, label='Original', alpha=0.5)
    plt.plot(rolling_mean, label='Rolling Mean (30 days)')
    plt.plot(rolling_std, label='Rolling Std (30 days)')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Dew Point')
    plt.legend()
    plt.show()

def check_stationarity(df):
    """Perform Augmented Dickey-Fuller test for stationarity."""
    result = adfuller(df.dropna())
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    
    if result[1] <= 0.05:
        print("Data is stationary")
    else:
        print("Data is non-stationary")

def seasonal_decompose_plot(df, period=365):
    """Perform and plot seasonal decomposition."""
    decomposition = seasonal_decompose(df.dropna(), period=period)
    
    plt.figure(figsize=(14, 8))
    plt.subplot(411)
    plt.plot(df, label='Original')
    plt.legend()
    plt.subplot(412)
    plt.plot(decomposition.trend, label='Trend')
    plt.legend()
    plt.subplot(413)
    plt.plot(decomposition.seasonal, label='Seasonality')
    plt.legend()
    plt.subplot(414)
    plt.plot(decomposition.resid, label='Residuals')
    plt.legend()
    plt.tight_layout()
    plt.show()

def train_arima_model(train_data):
    """Train ARIMA model with auto parameter selection."""
    model = pm.auto_arima(
        train_data,
        seasonal=True,
        m=12,
        start_p=0,
        start_q=0,
        max_p=5,
        max_q=5,
        d=None,  # Let model determine differencing
        test='adf',
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        random_state=RANDOM_STATE
    )
    
    print(model.summary())
    return model

def evaluate_model(model, test_data):
    """Evaluate the model and plot results."""
    # Generate forecasts
    forecast, conf_int = model.predict(
        n_periods=len(test_data),
        return_conf_int=True
    )
    
    # Create forecast index
    forecast_index = pd.date_range(
        start=test_data.index[0],
        periods=len(test_data),
        freq=test_data.index.freq
    )
    
    forecast_series = pd.Series(forecast, index=forecast_index)
    lower_series = pd.Series(conf_int[:, 0], index=forecast_index)
    upper_series = pd.Series(conf_int[:, 1], index=forecast_index)
    
    # Calculate metrics
    mse = mean_squared_error(test_data, forecast)
    mae = mean_absolute_error(test_data, forecast)
    rmse = np.sqrt(mse)
    
    print(f"Model Evaluation Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Plot results
    plt.figure(figsize=(14, 7))
    plt.plot(train_data.index, train_data, label='Training Data')
    plt.plot(test_data.index, test_data, label='Actual Values')
    plt.plot(forecast_series.index, forecast_series, label='Forecast')
    plt.fill_between(lower_series.index, 
                    lower_series, 
                    upper_series, 
                    color='k', alpha=0.1)
    plt.title('ARIMA Forecast vs Actuals')
    plt.xlabel('Date')
    plt.ylabel('Dew Point')
    plt.legend()
    plt.show()
    
    return forecast_series, (mse, mae, rmse)

if __name__ == "__main__":
    # Load and preprocess data
    weather_df = load_and_preprocess_data('delhi_dewpoint.csv')
    
    # Exploratory Data Analysis
    plot_time_series(weather_df, 'Dew Point Time Series')
    seasonal_decompose_plot(weather_df)
    check_stationarity(weather_df)
    
    # Split data into train and test sets
    train_data, test_data = train_test_split(
        weather_df,
        test_size=TEST_SIZE,
        shuffle=False
    )
    
    # Train ARIMA model
    model = train_arima_model(train_data)
    
    # Evaluate model
    forecast, metrics = evaluate_model(model, test_data)
