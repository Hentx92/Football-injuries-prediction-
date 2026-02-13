⚽ Football Injury Risk Prediction
Random Forest + PCA + Streamlit Application

📌 Project Overview

This project is a machine learning-based injury risk prediction system designed for football players.

It uses a Random Forest Classifier combined with PCA (Principal Component Analysis) to predict the probability of injury based on 18 key physical, performance, and wellness metrics.

The model is deployed through an interactive Streamlit web application that supports:

🔮 Single player prediction

📁 Batch team analysis via CSV upload

📊 Risk visualization dashboard

💡 Automated personalized recommendations

🧠 Machine Learning Model
Algorithm

Random Forest Classifier

Dimensionality Reduction

Principal Component Analysis (PCA)

Input Features (18 Total)

Physical Metrics

Height

Weight

BMI

Training Load

Training hours per week

Matches played last season

Warmup adherence

Strength & Biomechanics

Knee strength score

Hamstring flexibility

Balance test score

Performance

Sprint speed (10m)

Agility score

Reaction time

Wellness

Sleep hours per night

Stress level

Nutrition quality

Player Profile

Age

Position

Previous injury count

📊 Risk Classification System

Risk %	Category	Action Required
0–25%	🟢 Low	Maintain current training
25–50%	🟡 Moderate	Monitor and adjust load
50–75%	🟠 High	Preventive intervention
75–100%	🔴 Critical	Immediate medical evaluation

🚀 Application Features

1️⃣ Single Prediction Mode

Interactive form input

Live risk gauge visualization

Contributing risk factors analysis

Personalized recommendations

2️⃣ Batch Analysis Mode

CSV upload for full team evaluation

Automatic risk distribution chart

Risk categorization

Downloadable injury risk report

3️⃣ Professional UI

Wide layout dashboard

Risk color coding

Interactive Plotly visualizations

🛠 Tech Stack

Python

Scikit-learn

PCA

Streamlit

Pandas

NumPy

Plotly


📂 Project Structure
football-injury-prediction/
│
├── app.py
├── random_forest_model.pkl
├── pca_object.pkl
├── model_columns.pkl
├── requirements.txt
├── README.md
├── notebooks/
│   └── training_notebook.ipynb
└── data/
    └── dataset.csv

▶️ Run Locally

1️⃣ Clone Repository
git clone https://github.com/Hentx92/football-injury-prediction.git
cd football-injury-prediction

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Launch Application
streamlit run app.py


The app will open automatically in your browser.

📈 Future Improvements

Feature importance visualization

Model performance metrics dashboard

Cross-validation reporting

SHAP explainability integration

Cloud deployment (Streamlit Cloud / Docker)

🎯 Purpose

This project demonstrates:

Applied Machine Learning in Sports Analytics

End-to-end ML deployment

Model-to-UI integration

Data-driven risk assessment

👨‍💻 Author

Abdallah Nagiub
Data Science & AI Enthusiast
