# ================================
# IMPORTS
# ================================
from flask import Flask, render_template, request, redirect, session, jsonify
from pymongo import MongoClient
import joblib
from datetime import datetime
import os

import shap
import numpy as np

from dotenv import load_dotenv


# ================================
# CREATE APP
# ================================
load_dotenv()

app = Flask(__name__)
app.secret_key = "super_secret_key_123"


def is_logged_in():
    return bool(session.get("email"))


# ================================
# LOAD MODEL
# ================================
model = joblib.load("diabetes_model.pkl")

xgb_model = model.named_steps["model"]

explainer = shap.TreeExplainer(xgb_model)


# ================================
# MONGODB CONNECTION
# ================================
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

client = MongoClient(MONGO_URI)

db = client["diabetes_db"]

print("MongoDB Connected")


# ==========================================================
# HOME PAGE
# ==========================================================
@app.route("/")
def home():
    return render_template("home.html")


# ==========================================================
# REGISTER PAGE
# ==========================================================
@app.route("/register", methods=["GET","POST"])
def register():

    if request.method == "POST":

        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not username or not email or not password:
            return "All fields are required"

        if "@" not in email or "." not in email:
            return "Please enter a valid email"

        if db.users.find_one({"$or": [{"username": username}, {"email": email}]}):

            return "User already exists with same username/email"

        db.users.insert_one({
            "username": username,
            "email": email,
            "password": password
        })

        return redirect("/login")

    return render_template("register.html")


# ==========================================================
# LOGIN
# ==========================================================
@app.route("/login", methods=["GET","POST"])
def login():

    if request.method == "POST":

        role = request.form.get("role")
        login_email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        # ADMIN LOGIN
        if role == "admin" and login_email == "rajawatsanskar769@gmail.com" and password == "123456789":

            session.clear()

            session["role"] = "admin"
            session["username"] = "Admin"
            session["email"] = login_email

            return redirect("/dashboard")

        # USER LOGIN
        if role == "user":

            user = db.users.find_one({
                "email": login_email,
                "password": password
            })

            if user:

                session.clear()

                session["role"] = "user"

                session["username"] = user.get("username")
                session["email"] = user.get("email", "")

                return redirect("/dashboard")

        return "Wrong credentials"

    return render_template("login.html")


# ==========================================================
# DASHBOARD
# ==========================================================
@app.route("/dashboard")
def dashboard():

    if not is_logged_in():

        return redirect("/login")

    if session.get("role") != "admin":
        return redirect("/predict_page")

    data = list(db.predictions.find().sort("time",-1).limit(10))

    total = db.predictions.count_documents({})

    high = db.predictions.count_documents({"prediction":1})

    low = db.predictions.count_documents({"prediction":0})

    return render_template(
        "dashboard.html",
        data=data,
        total=total,
        high=high,
        low=low
    )


# ==========================================================
# USER HISTORY PAGE
# ==========================================================
@app.route("/history")
def history():

    if not is_logged_in():
        return redirect("/login")

    email = session.get("email", "")

    if not email:
        return redirect("/login")

    data = list(
        db.predictions
        .find({"email": email})
        .sort("time",-1)
    )

    return render_template("templates/history.html",data=data)


# ==========================================================
# PREDICTION PAGE
# ==========================================================
@app.route("/predict_page")
def predict_page():

    if not is_logged_in():
        return redirect("/login")

    return render_template("predict.html")


# ==========================================================
# PREDICT API
# ==========================================================
@app.route("/predict", methods=["POST"])
def predict():

    if not is_logged_in():
        return jsonify({"error":"Unauthorized"}),401

    try:

        data = request.json or {}

        gender = str(data.get("Gender", "")).strip().title()
        pregnancies_raw = data.get("Pregnancies", 0)
        if pregnancies_raw in [None, "", "null"]:
            pregnancies_raw = 0
        pregnancies = float(pregnancies_raw)
        glucose = float(data.get("Glucose",0))
        bp = float(data.get("BloodPressure",0))
        skin = float(data.get("SkinThickness",0))
        insulin = float(data.get("Insulin",0))
        bmi = float(data.get("BMI",0))
        dpf = float(data.get("DiabetesPedigreeFunction",0))
        age = float(data.get("Age",0))

        if gender not in ["Male", "Female"]:
            return jsonify({"error":"Gender must be Male or Female"})

        # VALIDATION
        min_age = 1 if (gender == "Male" or pregnancies == 0) else 18
        if age < min_age or age > 100:
            return jsonify({"error": f"Age must be between {min_age} and 100"})

        if glucose < 50 or glucose > 300:
            return jsonify({"error":"Invalid Glucose value"})

        if bmi < 10 or bmi > 70:
            return jsonify({"error":"Invalid BMI value"})

        if gender == "Male":
            pregnancies = 0

        normalized_input = {
            "Gender": gender,
            "Pregnancies": int(pregnancies),
            "Glucose": round(glucose, 2),
            "BloodPressure": round(bp, 2),
            "SkinThickness": round(skin, 2),
            "Insulin": round(insulin, 2),
            "BMI": round(bmi, 2),
            "DiabetesPedigreeFunction": round(dpf, 4),
            "Age": int(age),
        }

        input_data = [[
            normalized_input["Pregnancies"],
            normalized_input["Glucose"],
            normalized_input["BloodPressure"],
            normalized_input["SkinThickness"],
            normalized_input["Insulin"],
            normalized_input["BMI"],
            normalized_input["DiabetesPedigreeFunction"],
            normalized_input["Age"],
        ]]

        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0][1])

        # SHAP
        input_array = np.array(input_data)
        shap_values = explainer.shap_values(input_array)

        features = [
            "Pregnancies","Glucose","BloodPressure",
            "SkinThickness","Insulin","BMI",
            "DiabetesPedigreeFunction","Age"
        ]

        impact = dict(zip(features, shap_values[0]))

        sorted_impact = sorted(
            impact.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]

        top_factors = [f[0] for f in sorted_impact]

        # SAVE DB
        current_time = datetime.now()
        username = session.get("username") or session.get("email", "").split("@")[0]
        user_email = session.get("email", "")

        prediction_doc = {

            "user": username,
            "email": user_email,
            "input":normalized_input,
            # Duplicate key fields at top level so MongoDB Compass can show them without expanding input.
            "gender": normalized_input["Gender"],
            "pregnancies": normalized_input["Pregnancies"],
            "glucose": normalized_input["Glucose"],
            "blood_pressure": normalized_input["BloodPressure"],
            "skin_thickness": normalized_input["SkinThickness"],
            "insulin": normalized_input["Insulin"],
            "bmi": normalized_input["BMI"],
            "diabetes_pedigree_function": normalized_input["DiabetesPedigreeFunction"],
            "age": normalized_input["Age"],
            "prediction":prediction,
            "probability":probability,
            "time":current_time,
            "time_display":current_time.strftime("%d-%m-%Y %H:%M:%S")

        }

        db.predictions.insert_one(prediction_doc)

        return jsonify({

            "prediction":prediction,
            "probability":round(probability*100,2),
            "top_factors":top_factors

        })

    except Exception as e:

        print("Prediction Error:",e)

        return jsonify({"error":"Prediction failed"}),500


# ==========================================================
# LOGOUT
# ==========================================================
@app.route("/logout")
def logout():

    session.clear()

    return redirect("/login")


# ==========================================================
# RUN APP
# ==========================================================
if __name__ == "__main__":

    app.run(debug=True)