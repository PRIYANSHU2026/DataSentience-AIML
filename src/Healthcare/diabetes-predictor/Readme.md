Diabetes Prediction using Logistic Regression

Project Goal

The goal of this project is to build a machine learning model that predicts
whether a person has diabetes based on their medical attributes. 
The model is trained on a labeled dataset and evaluated using metrics such as accuracy, 
 F1-score, and confusion matrix visualization.

Installation and Requirements

Install dependencies using:


`pip install -r requirements.txt`

Required libraries include:

pandas

numpy

scikit-learn

matplotlib

How It Works
Data Loading: The diabetes dataset is loaded from a CSV file.

Data Preprocessing:

Features (X) and labels (y) are separated.

StandardScaler is used to standardize the feature values.

Data Splitting: The dataset is split into training (80%) and testing (20%) 
sets with stratification for balanced target distribution.

Model Training: Logistic Regression is trained with max_iter=1000 to ensure convergence.

Evaluation:

Accuracy score

F1-score

Confusion Matrix Visualization

Predictive System:

Allows the user to input medical parameters.

Returns a prediction whether the person is diabetic or not (with a friendly Bengali touch).

Conclusion
The classification model achieved strong performance with:

Accuracy: ~75%

F1-Score: ~60%

These results indicate that the Logistic Regression model is a good starting point for medical 
risk prediction and can be improved with further feature engineering and hyperparameter tuning.

Author
Shirsha Nag | GSSOC'25 | Contributor

