Volatility & Trend Classification Model
This project aims to automatically classify each day in a cryptocurrency's historical price data into a market state based on price trends and volatility levels.
[!ui](assets/image.png)
The classification combines trend direction (e.g., Bullish, Bearish, Sideways) with volatility conditions (e.g., High Volatility, Low Volatility), resulting in categories such as:

Bullish_HighVol — Strong upward trend with large price swings.

Bearish_LowVol — Downward trend with relatively small daily price fluctuations.

Sideways_HighVol — No clear price trend, but with significant volatility.

(and other combinations depending on computed features).

By using rolling-window statistical features and technical indicators, the model can provide insights into the current market regime for each cryptocurrency, which is valuable for traders, analysts, and automated trading systems.

Workflow
The project follows a simple SSOC-style (Single Source of Code) workflow with modular scripts for preprocessing, training, and prediction:

Preprocess the data
Cleans and transforms the historical OHLC (Open, High, Low, Close) price data, computes technical indicators like returns, rolling volatility, and moving averages, and assigns market state labels for supervised learning.

Commands for using it and testing ->
python preprocess.py --symbol BTCUSDT
python train.py --symbol BTCUSDT
python predict.py --symbol BTCUSDT --rows 7
