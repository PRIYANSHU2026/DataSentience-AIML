Influencer Campaign ROI Predictor
📌 Overview
The Influencer Campaign ROI Predictor is a machine learning application designed to estimate product sales from influencer marketing campaigns based on campaign details, influencer category, engagement metrics, and timing.
It helps marketing teams forecast ROI and make data-driven decisions before launching campaigns.
[!ui](assets/image.png)
🚀 Features
Data Preprocessing: Automatically cleans, encodes, and prepares input data.

Model Training: Trains a machine learning regression model to predict sales.

Prediction Pipeline: Accepts raw campaign details and returns an estimated sales figure.

Date Feature Handling: Can parse start_date into start_month and start_dayofweek automatically.

Customizable: Easily retrain on updated datasets.

📊 How It Works
Train the model
Load the dataset, preprocess it, and train the regression model.

python train.py
python predict.py
