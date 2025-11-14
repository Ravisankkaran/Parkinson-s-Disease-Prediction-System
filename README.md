🧠 Parkinson’s Disease Prediction System

An end-to-end Machine Learning project that predicts Parkinson’s Disease using biomedical voice measurements and patient metrics. This project includes full data preprocessing, model training, evaluation, and a Gradio UI for real-time predictions.

📌 Overview

Parkinson’s Disease is a neurodegenerative disorder affecting millions worldwide.
This project builds an ML pipeline capable of predicting Parkinson’s disease using biomedical voice parameters.

🔍 Key Features

Complete data preprocessing

Class imbalance handling (SMOTE, SMOTETomek)

Multiple ML models: Random Forest, XGBoost, Neural Network

Hyperparameter tuning

Feature scaling and selection

Model evaluation using industry-standard metrics

Saved models using .pkl & .h5

Interactive Gradio-based prediction UI

📊 Dataset

The dataset contains biomedical voice measurements such as:

MDVP frequency measures

Jitter

Shimmer

Harmonic-to-Noise Ratio (HNR)

Voice intensity metrics

UPDRS scores

🧹 Preprocessing Pipeline

✔ Handling missing values
✔ Dropping duplicates
✔ Feature scaling (StandardScaler)
✔ Encoding classes
✔ Train–test split with stratification
✔ Balancing using SMOTE and SMOTETomek

🤖 Models Used
1️⃣ Random Forest Classifier

Strong baseline model

Good performance with tabular features

2️⃣ XGBoost Classifier

Powerful boosting model

Handles noise & nonlinearity

Hyperparameter tuning applied

3️⃣ Artificial Neural Network (Keras)

Dense network architecture

Dropout + EarlyStopping for regularization

📈 Evaluation Metrics

Each model was evaluated using:

Accuracy

Precision

Recall

F1-score

ROC-AUC

Confusion matrix visualization

XGBoost and the Neural Network showed the highest overall performance.

🚀 Deployment (Gradio App)

This project includes a Gradio UI for real-time Parkinson’s prediction.

Users can:

Enter voice metrics manually

Get instant prediction results

View model confidence

Run using:

python gradio_app.py



🛠 Tech Stack

Python

NumPy, Pandas

Scikit-learn

XGBoost

TensorFlow/Keras

Imbalanced-learn (SMOTE, SMOTETomek)

Matplotlib, Seaborn

Gradio

▶️ How to Run
Step 1 — Install dependencies
pip install -r requirements.txt

Step 2 — Run the training notebook

Execute:

Parkinson's_Disease_prediction.ipynb

Step 3 — Launch the UI
python gradio_app.py

🧭 Future Improvements

Model explainability (SHAP, LIME)

Web deployment using Flask/FastAPI + Docker

More robust deep voice feature extraction

Dataset expansion for real-world variability

👨‍💻 Author

Ravi Sankkaran
Machine Learning & AI Developer
