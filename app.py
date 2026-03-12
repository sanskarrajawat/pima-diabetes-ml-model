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


# ================================
# CREATE APP
# ================================
app = Flask(__name__)
app.secret_key = "super_secret_key_123"


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

        username = request.form.get("username")
        password = request.form.get("password")

        if db.users.find_one({"username":username}):

            return "User already exists"

        db.users.insert_one({
            "username":username,
            "password":password
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
        username = request.form.get("username")
        password = request.form.get("password")

        # ADMIN LOGIN
        if role == "admin" and username == "admin" and password == "vaibhav":

            session["role"] = "admin"

            return redirect("/dashboard")

        # USER LOGIN
        if role == "user":

            user = db.users.find_one({
                "username":username,
                "password":password
            })

            if user:

                session["role"] = "user"

                session["username"] = username

                return redirect("/predict_page")

        return "Wrong credentials"

    return render_template("login.html")


# ==========================================================
# DASHBOARD
# ==========================================================
@app.route("/dashboard")
def dashboard():

    if "role" not in session or session["role"] != "admin":

        return redirect("/login")

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

    if "role" not in session:
        return redirect("/login")

    username = session.get("username")

    data = list(
        db.predictions
        .find({"user":username})
        .sort("time",-1)
    )

    return render_template("history.html",data=data)


# ==========================================================
# PREDICTION PAGE
# ==========================================================
@app.route("/predict_page")
def predict_page():

    if "role" not in session:
        return redirect("/login")

    return render_template("predict.html")


# ==========================================================
# PREDICT API
# ==========================================================
@app.route("/predict", methods=["POST"])
def predict():

    if "role" not in session:
        return jsonify({"error":"Unauthorized"}),401

    try:

        data = request.json

        gender = data.get("Gender")
        pregnancies = float(data.get("Pregnancies",0))
        glucose = float(data.get("Glucose",0))
        bp = float(data.get("BloodPressure",0))
        skin = float(data.get("SkinThickness",0))
        insulin = float(data.get("Insulin",0))
        bmi = float(data.get("BMI",0))
        dpf = float(data.get("DiabetesPedigreeFunction",0))
        age = float(data.get("Age",0))

        # VALIDATION
        if age < 18 or age > 100:
            return jsonify({"error":"Age must be between 18 and 100"})

        if glucose < 50 or glucose > 300:
            return jsonify({"error":"Invalid Glucose value"})

        if bmi < 10 or bmi > 70:
            return jsonify({"error":"Invalid BMI value"})

        if gender == "Male":
            pregnancies = 0

        input_data = [[
            pregnancies,
            glucose,
            bp,
            skin,
            insulin,
            bmi,
            dpf,
            age
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
        db.predictions.insert_one({

            "user":session.get("username"),
            "input":data,
            "prediction":prediction,
            "probability":probability,
            "time":datetime.now()

        })

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