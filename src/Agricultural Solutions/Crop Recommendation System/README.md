🌾 Crop Recommendation System using Machine Learning
This project is a machine learning-based Crop Recommendation System that suggests the most suitable crop to grow based on environmental conditions such as soil nutrients and weather data.

🖼️ Live UI Preview

[!ui screenshot](assets/cofee.jpeg)
[!ui screenshot](assets/rice.jpeg)

📊 Dataset
The dataset used is Crop_recommendation.csv and contains the following features:

N: Nitrogen content in soil

P: Phosphorus content in soil

K: Potassium content in soil

temperature: Temperature in °C

humidity: Relative humidity in %

ph: pH value of the soil

rainfall: Rainfall in mm

label: Target crop label (e.g., rice, maize, etc.)

🧠 Model
A classification model (e.g., RandomForest or DecisionTree) is trained to predict the best crop based on the input features.

The label encoder is used to map crop names to numeric labels and vice versa.

The model is saved using joblib as crop_model.pkl and label_encoder.pkl.

**📈 Feature Importance Visualization**

After training, the feature importance of each input variable is visualized using `matplotlib` and `seaborn`.

