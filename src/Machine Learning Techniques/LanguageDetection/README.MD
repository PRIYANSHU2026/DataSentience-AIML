# Language Detection Model

Data set : https://www.kaggle.com/datasets/basilb2s/language-detection

## Steps Followed
##### 1. Load the data set
##### 2. Encode the labels into categoical form
##### 3. Pre-process the Text content
##### 4. Tokenizing
##### 5. Create a Dictionary for Vocabulary
##### 6. Count the Word Frequencies (Unigrams were considered)
##### 7. Split the dataset into train and test sets
##### 8. Perform Supervised Classification 

### Could achieve 97.3% accuracy using Naive Bayes Classifier 


## Additional Improvements Made Under GSSoC 2025

**1. Dataset Exploration & Cleaning**

**Need:** 
- I noticed that the dataset was uncleaned, which gave an accuracy of around 98% with Naive Bayes. Therefore, I first decided to clean the dataset to carefully understand it, identify and remove any missing, duplicate, or invalid records, and perform text cleaning so that the data would be properly structured and ready for model training.

<br>

**Steps taken:**
- Checked dataset shape, column names, missing values, empty strings, and duplicate rows.
- Removed 66 duplicate entries.
- Label encoding - Converted language names to numeric labels for ML models
- Normalized text to lowercase and removed numbers/special characters.
<br>

**Result:** 
- After performing proper data cleaning (removing noise, duplicates, and inconsistencies), the Naive Bayes accuracy is now ~94%.
<br>

**2. Language-wise Data Distribution Visualization**

**Need:**  
- To visually analyze the dataset and understand how many samples are available for each language.
<br> 

**Step taken:**
- Created a bar plot showing number of samples per language.
<br>

**Result:** 
- This helps in detecting class imbalance issues (e.g., Hindi had the fewest samples) and ensures we have a clear picture of data distribution before training the model.
<br>

**3. Replaced Manual Count Vectorization with TfidfVectorizer**

**Need:**
- Manual count vectorization only considers word frequency, which can lead to bias toward frequently occurring but less informative words. 
- It doesn’t account for the importance of rare yet meaningful terms in classification tasks.
<br>

**Step taken:**
- Replaced custom count vectorization logic with TfidfVectorizer from scikit-learn.
<br>

**Result:** 
- TF-IDF improves feature weighting by reducing the influence of very frequent but less informative words.
<br>

**4. Added Multiple Classifiers for Performance Comparison**

**Need:**
- To evaluate multiple machine learning classifiers on the same dataset to identify which model performs best for language detection, ensuring that the chosen model provides the highest accuracy and reliability.
<br>

**Steps taken:**

(1) Selected multiple classifiers for evaluation:
- Logistic Regression
- Random Forest
- Decision Tree
- SVM
(Along with previously implemented Naive Bayes)
<br>

(2) Trained each classifier using the same training dataset (x_train, y_train).
<br>

(3) Evaluated performance of each model using:
- Accuracy (overall correctness)
- Precision (correctness for each predicted class)
- Recall (ability to find all correct examples)
<br>

**Results:** 

**Naive Bayes Results after data cleaning :**
- Accuracy: 94.84%
- Precision: 95.86%
- Recall: 94.84%
<br>

**Logistic Regression Results:**
- Accuracy: 94.68%
- Precision: 96.32%
- Recall: 94.68%
<br>

**Random Forest Results:**
- Accuracy: 92.44%
- Precision: 95.62%
- Recall: 92.44%
<br>

**Decision Tree Results:**
- Accuracy: 86.44 %
- Precision: 91.29 %
- Recall: 86.44 %
<br>

**Issue Faced**

An attempt was made to implement the SVM classifier for this dataset. However, due to frequent Colab runtime disconnections and the high computation time required, the model could not be executed successfully. Therefore, the SVM results have not been included in the comparison.

### Important observation 

**Naive Bayes accuracy before data cleaning :**
- Accuracy: 97.3%
<br>

**Naive Bayes accuracy after data cleaning :**
- Accuracy: 94.84%

While the accuracy slightly dropped, the model is now trained on cleaner data, which should improve its generalization ability and reduce overfitting. 
I have kept both versions for comparison in the README.

## Best model based on accuracy is still Naive Bayes

**5. Added Confusion Matrix Heatmaps**

**Need:**
- To visualize the classification performance of each model by showing the distribution of correct and incorrect predictions for each language. 
- This helps in identifying where the model is making mistakes and which classes are being confused with each other.
<br>

**Steps taken:**
- Implemented a confusion matrix for each classifier using sklearn.metrics.confusion_matrix.
- Used seaborn.heatmap to create a visually clear heatmap representation of the matrix.
- Added proper labels for languages on both axes for easy interpretation.
<br>

## Visualizations

**Language-wise Data Distribution**  
![Language-wise Data Distribution](https://github.com/Simran-ch/DataSentience-AIML/blob/main/src/Machine%20Learning%20Techniques/LanguageDetection/Language-wise%20Data%20Distribution.png?raw=true)

**Model Accuracy Comparison**  
![Model Accuracy Comparison](https://github.com/Simran-ch/DataSentience-AIML/blob/main/src/Machine%20Learning%20Techniques/LanguageDetection/Model%20Accuracy%20Comparision.png?raw=true)

**Confusion Matrix - Naive Bayes**  
![Confusion Matrix - Naive Bayes](https://github.com/Simran-ch/DataSentience-AIML/blob/main/src/Machine%20Learning%20Techniques/LanguageDetection/Confusion%20Matrix%20Heatmap%20-%20Naive%20Bayes.png?raw=true)

**Confusion Matrix - Logistic Regression**  
![Confusion Matrix - Logistic Regression](https://github.com/Simran-ch/DataSentience-AIML/blob/main/src/Machine%20Learning%20Techniques/LanguageDetection/Confusion%20Matrix%20Heatmap%20-%20Logistic%20Regression.png?raw=true)

**Confusion Matrix - Decision Tree**  
![Confusion Matrix - Decision Tree](https://github.com/Simran-ch/DataSentience-AIML/blob/main/src/Machine%20Learning%20Techniques/LanguageDetection/Confusion%20Matrix%20Heatmap%20-%20Decision%20Tree.png?raw=true)

**Confusion Matrix - Random Forest**  
![Confusion Matrix - Random Forest](https://github.com/Simran-ch/DataSentience-AIML/blob/main/src/Machine%20Learning%20Techniques/LanguageDetection/Confusion%20Matrix%20Heatmap%20-%20Random%20Forest.png?raw=true)
