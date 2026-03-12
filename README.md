# 🩺 Pima Diabetes Prediction System

A full-stack **Machine Learning Web Application** that predicts the likelihood of diabetes using the **Pima Indians Diabetes Dataset**.
The application integrates **Flask, Machine Learning, Explainable AI (SHAP), and MongoDB** to provide predictions along with feature explanations.

---

## 🚀 Project Overview

This project predicts whether a patient is likely to have diabetes based on medical attributes such as:

* Pregnancies
* Glucose Level
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age

The system not only predicts diabetes risk but also explains the **most influential features** affecting the prediction.

---

## 🧠 Key Features

✔ Machine Learning Diabetes Prediction
✔ Explainable AI using SHAP
✔ User Authentication System (Register & Login)
✔ Admin Dashboard
✔ Prediction History Storage
✔ MongoDB Database Integration
✔ REST API Prediction Endpoint

---

## 🛠 Tech Stack

**Backend**

* Python
* Flask

**Machine Learning**

* Scikit-Learn
* XGBoost
* SHAP (Explainable AI)

**Database**

* MongoDB

**Frontend**

* HTML
* CSS
* JavaScript

---

## 📂 Project Structure

```
pima-diabetes-ml-model
│
├── app.py
├── diabetes_model.pkl
├── requirements.txt
│
├── templates
│   ├── home.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── predict.html
│   └── history.html
│
├── static
│
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```
git clone https://github.com/sanskarrajawat/pima-diabetes-ml-model.git
```

### 2️⃣ Navigate to Project

```
cd pima-diabetes-ml-model
```

### 3️⃣ Create Virtual Environment

```
python -m venv venv
```

Activate:

Mac/Linux

```
source venv/bin/activate
```

Windows

```
venv\Scripts\activate
```

### 4️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 5️⃣ Run Application

```
python app.py
```

App will run at:

```
http://127.0.0.1:5000
```

---

## 📊 Model Information

The model was trained using the **Pima Indians Diabetes Dataset** and predicts diabetes risk using medical indicators.

Model Pipeline Includes:

* Data Preprocessing
* Feature Scaling
* XGBoost Classifier
* Probability Prediction

Explainability is implemented using **SHAP** to highlight the most influential factors in each prediction.

---

## 📈 Prediction Output

The system returns:

* Diabetes Prediction (0 / 1)
* Probability Score
* Top Influential Features

Example:

```
Prediction: High Diabetes Risk
Probability: 78.45%
Top Factors:
- Glucose
- BMI
- Age
```

---

## 🔐 Authentication System

The application includes:

**User Features**

* Register Account
* Login
* Make Predictions
* View Prediction History

**Admin Features**

* Dashboard Analytics
* Recent Predictions
* Risk Statistics

---

## 📌 Future Improvements

* Docker Deployment
* Cloud Deployment (AWS / Render)
* Data Visualization Dashboard
* Model Performance Monitoring
* API Documentation

---

## 👥 Team

This project was developed as part of a **team-based academic project**.

* Vaibhav Pratap Singh Rajawat
  BTech Data Science
  SRM Institute of Science and Technology

GitHub:
https://github.com/sanskarrajawat

---

## ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.
