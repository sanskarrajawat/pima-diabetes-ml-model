# PIMA Diabetes Prediction System

PIMA Diabetes Prediction System is a Flask-based machine learning web application for diabetes risk screening using the PIMA dataset feature set. The app supports user authentication, role-based access, prediction history, and admin analytics.

## Overview

The application predicts diabetes risk from the following medical inputs:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

Each prediction returns:

- Risk class (Low Risk / High Risk)
- Probability score
- Top model factors from SHAP explainability

## Current Features

- User registration and login
- Admin login with dedicated dashboard
- Prediction form with validation logic
- SHAP-based top factor explanation
- User-specific prediction history
- MongoDB persistence for users and predictions
- Professional dark UI across all pages

## Tech Stack

- Backend: Python, Flask, PyMongo
- ML: scikit-learn, XGBoost, SHAP, joblib
- Database: MongoDB
- Frontend: HTML, CSS, JavaScript, Chart.js

## Project Structure

```text
pima_project/
|- app.py
|- diabetes.csv
|- diabetes_model.pkl
|- train_model.py
|- requirements.txt
|- static/
|  |- styles.css
|  |- pro-ui.css
|- templates/
|  |- home.html
|  |- login.html
|  |- register.html
|  |- predict.html
|  |- dashboard.html
|  |- index.html
|  |- templates/
|     |- history.html
```

## Local Setup

1. Clone repository

```bash
git clone https://github.com/sanskarrajawat/pima-diabetes-ml-model.git
cd pima-diabetes-ml-model
```

2. Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Configure environment variables

Create `.env` in project root:

```env
MONGO_URI=mongodb://localhost:27017/
```

5. Run application

```bash
python app.py
```

App URL:

```text
http://127.0.0.1:5000
```

## Data Persistence

Predictions are saved in MongoDB database `diabetes_db` under collection `predictions`.

Each saved document includes:

- User identity fields (`user`, `email`)
- Full normalized input payload
- Top-level medical fields for easy inspection in MongoDB Compass
- Prediction class and probability
- Timestamp

## Routes

- `/` Public home page
- `/register` User registration
- `/login` User/Admin login
- `/dashboard` Admin dashboard (role restricted)
- `/predict_page` User prediction form
- `/predict` Prediction API endpoint
- `/history` Logged-in user history
- `/logout` Session logout

## Notes

- This project currently uses MongoDB-only persistence for prediction records.
- Keep `diabetes_model.pkl` in the root directory for successful model loading.

## Author

Sanskar Rajawat

GitHub: https://github.com/sanskarrajawat
