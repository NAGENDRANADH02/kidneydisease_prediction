 Chronic Kidney Disease Prediction using Machine Learning & Django
🧠 Project Overview
Chronic Kidney Disease (CKD) is a progressive condition in which the kidneys gradually lose their ability to function effectively over time. Early detection is crucial for timely medical intervention and improving patient outcomes.

This project leverages machine learning models to predict whether a patient is likely to have CKD based on several medical parameters. The prediction system is then deployed using a Django-based web interface, allowing users to input health details and receive real-time predictions and risk analysis.

🎯 Objectives
Predict whether a patient has CKD or not using clinical parameters.

Identify the most influential features using correlation.

Compare different machine learning algorithms (KNN, Naive Bayes, Logistic Regression).

Deploy the best-performing model using a user-friendly web interface built with Django.

Provide a risk score (%) and downloadable PDF report for the user.

🧬 Dataset Description
The dataset used in this project contains 400 rows and 25 columns. Each row corresponds to a patient and includes features like:

Blood pressure (bp)

Blood urea

Serum creatinine

Hemoglobin

Red blood cells (rbc)

Albumin

Blood glucose random

White blood cell count

Specific gravity

Age, and many others

The target variable is:

classification: ckd or notckd

🧪 Workflow
1. Data Preprocessing
Missing values are handled using mode imputation.

Categorical values are label-encoded.

Data is split into training and testing sets (80:20 ratio).

2. Feature Selection
Pearson correlation analysis is performed.

Highly correlated features with the target variable are selected to reduce dimensionality and improve model performance.

3. Model Building
Three models are built and compared:

K-Nearest Neighbors (KNN)

Naive Bayes

Logistic Regression

Each model is trained, evaluated, and the best one (based on accuracy) is saved using joblib.

4. Visualization
A bar chart is plotted showing the accuracy of all models to visualize performance comparison.
<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/136dc5da-7fbc-4bca-973d-6b6025b6d719" />


🛠️ Technologies Used
Technology	Purpose
Python	Core scripting and ML model
Pandas, NumPy	Data manipulation
Scikit-learn	Model building & evaluation
Matplotlib	Data visualization
Joblib	Model serialization
Django	Web framework for deployment
Bootstrap	UI styling
xhtml2pdf	Generating PDF reports

🖥️ Web Application Features
📤 Upload patient details using a structured form.

🔍 Predict CKD presence using the best-trained model.

📈 Show Risk Score (%) of CKD using model’s probability estimate.

📄 Generate and download a detailed PDF report of the prediction.

🔁 Option to reset and re-predict with new values.

🧠 Final Model Selection
All three models were trained and evaluated:

Model	Accuracy (%)
KNN	88.75%
Naive Bayes	95.00%
Logistic Regression	95.00%

The Naive Bayes model was selected for deployment as the best-performing model based on its balance of speed, simplicity, and accuracy.

✅ Key Highlights
✅ Real-world medical dataset

✅ Multiple ML models comparison

✅ Feature engineering with correlation

✅ Interactive and modern UI

✅ Risk score + downloadable PDF report

✅ Clean, modular, and well-documented codebase
 

 
