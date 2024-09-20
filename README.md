
# Microsoft : Classifying Cybersecurity Incidents with Machine Learning

Cybersecurity and Machine Learning


## Problem Statement:

**Objective:**
 As a data scientist at Microsoft(Assumption), I was tasked with enhancing the efficiency of Security Operation Centers (SOCs) by developing a machine learning model that can accurately predict the triage grade of cybersecurity incidents. Utilizing the comprehensive GUIDE dataset, your goal is to create a classification model that categorizes incidents as true positive (TP), benign positive (BP), or false positive (FP) based on historical evidence and customer responses. The model should be robust enough to support guided response systems in providing SOC analysts with precise, context-rich recommendations, ultimately improving the overall security posture of enterprise environments.

**Project Scope:**
I need to train the model using the train.csv dataset and provide evaluation metrics—macro-F1 score, precision, and recall—based on the model's performance on the test.csv dataset. This ensures that the model is not only well-trained but also generalizes effectively to unseen data, making it reliable for real-world applications.


## Business Use Cases

The solution developed in this project can be implemented in various business scenarios, particularly in the field of cybersecurity. Some potential applications include:

- **Security Operation Centers (SOCs):** Automating the triage process by accurately classifying cybersecurity incidents, thereby allowing SOC analysts to prioritize their efforts and respond to critical threats more efficiently.

- **Incident Response Automation:** Enabling guided response systems to automatically suggest appropriate actions for different types of incidents, leading to quicker mitigation of potential threats.

- **Threat Intelligence:** Enhancing threat detection capabilities by incorporating historical evidence and customer responses into the triage process, which can lead to more accurate identification of true and false positives.

- **Enterprise Security Management:** Improving the overall security posture of enterprise environments by reducing the number of false positives and ensuring that true threats are addressed promptly.



## 🛠 Skills
- Data Cleaning and Preprocessing
- Model Evaluation Metrics (Macro-F1 Score, Precision, Recall)
- Machine Learning Classification Techniques
- Cybersecurity Concepts and Frameworks (MITRE ATT&CK)
- Handling Imbalanced Datasets
- Model Benchmarking and Optimization




## Approach

- **Data Exploration and Understanding:** Start by loading the train.csv dataset and perform an initial inspection to understand the structure of the data, including the number of features, types of variables (categorical, numerical), and the distribution of the target variable (TP, BP, FP). Use visualizations and statistical summaries to identify patterns, correlations, and potential anomalies in the data

- **Data Preprocessing:** Import and concatenate, Clean and preprocess the data, handling missing values, Standardising Data Formats, encoding categorical variables. Create new features or modify existing ones to improve model performance. For example, combining related features, deriving new features from timestamps (like hour of the day or day of the week), or normalizing numerical variables.

-  **Data Splitting:** Before diving into model training, split the train.csv data into training and validation sets. This allows for tuning and evaluating the model before final testing on test.csv. Typically, a 70-30 or 80-20 split is used, but this can vary depending on the dataset's size.

- **Model Selection and Training:** Start with a simple baseline model, such as a logistic regression or decision tree, to establish a performance benchmark. Experiment with more sophisticated models such as Random Forests, Gradient Boosting Machines (e.g., XGBoost, LightGBM), and Neural Networks.


- **Model Evaluation and Tuning:** Evaluate the model using the validation set, focusing on macro-F1 score, precision, and recall. Analyze these metrics across different classes (TP, BP, FP) to ensure balanced performance.Based on the initial evaluation, fine-tune hyperparameters to optimize model performance. This may involve adjusting learning rates, regularization parameters, tree depths, or the number of estimators, depending on the model type. If class imbalance is a significant issue, consider techniques such as SMOTE (Synthetic Minority Over-sampling Technique), adjusting class weights, or using ensemble methods to boost the model's ability to handle minority classes effectively.

- **Model Interpretation:** After selecting the best model, analyze feature importance to understand which features contribute most to the predictions. This can be done using methods like SHAP values, permutation importance, or model-specific feature importance measures. Perform an error analysis to identify common misclassifications. This can provide insights into potential improvements, such as additional feature engineering or refining the model's complexity.

- **Final Evaluation on Test Set:** Once the model is finalized and optimized, evaluate it on the test.csv dataset. Report the final macro-F1 score, precision, and recall to assess how well the model generalizes to unseen data. Compare the performance on the test set to the baseline model and initial validation results to ensure consistency and improvement.


-  **Documentation and Reporting:** Thoroughly document the entire process, including the rationale behind chosen methods, challenges faced, and how they were addressed. Include a summary of key findings and model performance. Provide recommendations on how the model can be integrated into SOC workflows, potential areas for future improvement, and considerations for deployment in a real-world setting.


## Dataset

The primary objective of the dataset is to accurately predict incident triage grades—true positive (TP), benign positive (BP), and false positive (FP)—based on historical customer responses. To support this, we provide a training dataset containing 45 features, labels, and unique identifiers across 1M triage-annotated incidents. We divide the dataset into a train set containing 70% of the data and a test set with 30%, stratified based on triage grade ground-truth, OrgId, and DetectorId. We ensure that incidents are stratified together within the train and test sets to ensure the relevance of evidence and alert rows.


## Run Locally

Clone the project

```bash
  git clone https://github.com/Vijay6383/Microsoft-Classifying-Cybersecurity-Incidents-with-Machine-Learning.git
```

Install dependencies

```bash
  pip install dask[dataframe], dask-ml, scikit-learn, scipy, seaborn 
```


## Tags

- Machine Learning
- Classification
- Data Science
- Model Evaluation
- Feature Engineering
- SOC
- Threat Detection



## Project Evaluation metrics

The success and effectiveness of the project will be evaluated based on the following metrics:

- **Macro-F1 Score:** A balanced metric that accounts for the performance across all classes (TP, BP, FP), ensuring that each class is treated equally.

- **Precision:** Measures the accuracy of the positive predictions made by the model, which is crucial for minimizing false positives.

- **Recall:** Measures the model's ability to correctly identify all relevant instances (true positives), which is important for ensuring that real threats are not missed.
## 🔗 Links

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/vijay-moses-avm/)


