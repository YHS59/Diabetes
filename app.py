from flask import Flask, request, jsonify, render_template, flash, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime, timezone, timedelta
from flask_migrate import Migrate
import json
import requests

import pickle
import numpy as np
import pytz

# Initialize app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'this-should-be-secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
migrate = Migrate(app, db)

# User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    latest_recommendation = db.Column(db.Text)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    pregnancies = db.Column(db.Float)
    glucose = db.Column(db.Float)
    blood_pressure = db.Column(db.Float)
    skin_thickness = db.Column(db.Float)
    insulin = db.Column(db.Float)
    bmi = db.Column(db.Float)
    diabetes_pedigree_function = db.Column(db.Float)
    age = db.Column(db.Float)
    risk_level = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    user = db.relationship("User", backref="predictions")

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load ML model
with open("model/diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

# Generate personalized recommendations
def generate_recommendations(data, risk_level):
    recs = []

    if risk_level == "High Risk":
        recs.append("⚠️ You are at high risk for diabetes. It is important to talk to a healthcare provider soon.")
        if data["Glucose"] > 140:
            recs.append("🍭 Your blood sugar is very high. Reduce intake of sweet drinks, white rice, and desserts. Choose foods like oats, leafy greens, and lentils.")
        if data["BMI"] > 30:
            recs.append("🥗 Try eating smaller portions, avoiding fried foods, and drinking more water throughout the day.")

        recs.append("💪 Aim for at least 30 minutes of light to moderate exercise each day. Walking, dancing, or cycling can improve your sugar control.")

        if data["Insulin"] < 16:
            recs.append("💉 Your insulin level is low. This may require medical advice. Be sure to sleep well and manage stress.")
        if data["DiabetesPedigreeFunction"] > 1.0:
            recs.append("🧬 Strong family history found. Stay consistent with checkups, eat a balanced diet, and stay physically active.")
        if data["Age"] > 45 and data["Glucose"] > 125:
            recs.append("📅 You're over 45 with high blood sugar — schedule a regular screening every 3–6 months.")
    elif risk_level == "Low Risk":
        recs.append("✅ You are at low risk of diabetes. Keep up your healthy habits!")
        if data["BMI"] > 25:
            recs.append("🏃 Try adding light daily activities like walking or stretching. This can help maintain a healthy weight.")
        if data["Glucose"] > 120:
            recs.append("🩸 Your blood sugar is slightly high. Try limiting sugary snacks and drinks, and choose whole grains and vegetables more often.")

        recs.append("🛌 Aim for 7–8 hours of sleep per night. Good sleep supports healthy sugar levels.")
        recs.append("🚶 Consider short walks after meals — they help lower sugar spikes.")

    recs.append("😴 Try to sleep early and avoid screen time before bed.")
    recs.append("📵 Reduce stress through relaxation activities like reading, deep breathing, or talking with loved ones.")                  

    return recs

# Trend analysis helper function
def get_trend_message(values, label, timestamps=None):
    if timestamps is None or len(values) != len(timestamps):
        return f"No trend available for {label}."

    now = datetime.now(pytz.timezone('Asia/Kuala_Lumpur'))
    recent_values = [val for val, ts in zip(values, timestamps) if ts >= now - timedelta(days=90)]

    if len(recent_values) < 6:
        return f"Not enough recent data (last 90 days) to analyze {label} trend."

    last_6 = recent_values[-6:]
    previous_3 = last_6[:3]
    recent_3 = last_6[3:]

    avg_prev = sum(previous_3) / 3
    avg_recent = sum(recent_3) / 3

    if avg_recent > avg_prev + 0.5:
        trend = "increasing"
    elif avg_recent < avg_prev - 0.5:
        trend = "decreasing"
    else:
        trend = "stable"

    label = label.lower()
    if label == "bmi":
        if trend == "increasing":
            return "Your weight seems to be <strong>going up</strong> over time. This could raise your chances of developing diabetes or make it harder to manage. Try to include more fruits and vegetables in your meals, drink more water, and reduce sugary snacks. A daily walk or light exercise can also help a lot. If you are not sure where to start, talk to a health provider for advice."
        elif trend == "decreasing":
            return "Great progress! Your weight is <strong>going down</strong>, which is a good step toward better health. Losing even a small amount of weight can help your body control sugar levels better. Keep up the healthy eating and active lifestyle — you're on the right track!"
        else:
            return "Your weight is staying about the <strong>same</strong>. That is okay, but it is still important to eat balanced meals and move regularly. Staying active and avoiding too much sugar or fried food can help prevent future health problems."

    elif label == "glucose":
        if trend == "increasing":
            return "Your blood sugar levels have been <strong>rising</strong> recently. High sugar levels can lead to tiredness, blurry vision, and other serious issues if not managed well. Try cutting down on sweet drinks, desserts, and white bread. Choose whole grains, stay active, and drink water regularly. If this keeps happening, you may want to consult a doctor or pharmacist."
        elif trend == "decreasing":
            return "Your sugar levels are <strong>going down</strong> — that is a good sign! It means your current lifestyle might be helping. Keep making healthy choices, like eating on time, avoiding too much sugar, and staying active."
        else:
            return "Your sugar levels look <strong>stable</strong> for now. That is a positive sign, but it is still important to keep checking regularly, avoid too much sugar, and maintain a healthy lifestyle."

    elif label == "blood pressure":
        if trend == "increasing":
            return "Your blood pressure has been <strong>going up</strong> recently. This can put extra pressure on your heart and increase the risk of complications. Try to eat less salty food (like chips, instant noodles, or processed meats), reduce stress if possible, and get enough sleep. Even simple exercises like walking can help lower blood pressure."
        elif trend == "decreasing":
            return "Good job! Your blood pressure is <strong>going down</strong>. That means your heart might be getting some relief. Keep limiting salty foods, drink enough water, and stay active — your efforts are paying off."
        else:
            return "Your blood pressure is <strong>staying steady</strong>. That is great. Just continue to eat healthily, avoid too much salt, and manage stress to keep it that way."

    elif label == "insulin":
        if trend == "increasing":
            return "Your insulin level is <strong>rising</strong>, which might mean your body is working harder to manage sugar. This could be a sign of insulin resistance, especially if sugar levels are also high. Try to lower sugar and refined carbs in your diet. Whole foods, enough sleep, and regular activity can help your body respond better to insulin."
        elif trend == "decreasing":
            return "Your insulin level is <strong>going down</strong> — that could mean your body is starting to respond better. Keep up your good habits like eating balanced meals, exercising, and managing stress."
        else:
            return "Your insulin level is <strong>staying steady</strong>. That is a good sign, but it is still important to check regularly and continue your healthy habits."

    return f"{label.capitalize()} trend: {trend}."

# Routes
@app.route("/")
@login_required
def home():
    recommendations = []
    if current_user.latest_recommendation:
        try:
            recommendations = json.loads(current_user.latest_recommendation)
        except:
            recommendations = []
    chat_history = session.get('chat_history', [])
    return render_template("index.html", recommendations=recommendations, chat_history=chat_history)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        features = [float(data[key]) for key in [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]]
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Missing or invalid input fields. Please ensure all fields are numbers."}), 400

    input_array = np.array([features])
    prediction = model.predict(input_array)[0]
    risk = "High Risk" if prediction == 1 else "Low Risk"
    recommendations = generate_recommendations(data, risk)

    if current_user.is_authenticated:
        current_user.latest_recommendation = json.dumps(recommendations)

        new_prediction = Prediction(
            user_id=current_user.id,
            pregnancies=data["Pregnancies"],
            glucose=data["Glucose"],
            blood_pressure=data["BloodPressure"],
            skin_thickness=data["SkinThickness"],
            insulin=data["Insulin"],
            bmi=data["BMI"],
            diabetes_pedigree_function=data["DiabetesPedigreeFunction"],
            age=data["Age"],
            risk_level=risk
    )
        db.session.add(new_prediction)
        db.session.commit()

    return jsonify({
        "prediction": int(prediction),
        "risk_level": risk,
        "recommendations": recommendations
    })

@app.route("/history")
@login_required
def history():
    limit = request.args.get("limit", default=10, type=int)
    limit = min(limit, 50)  # Maximum 50 records

    user_predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).limit(limit).all()

    # Timezone setup
    malaysia_tz = pytz.timezone('Asia/Kuala_Lumpur')

    # Convert timestamps to Malaysia time
    for p in user_predictions:
        if p.timestamp.tzinfo is None:
            # Assume it's stored as naive UTC
            p.timestamp = pytz.utc.localize(p.timestamp)
        p.timestamp = p.timestamp.astimezone(malaysia_tz)

    # Extract glucose + Malaysia-time timestamps for the chart
    dates = [p.timestamp.strftime("%Y-%m-%d %H:%M") for p in reversed(user_predictions)]
    glucose_values = [p.glucose for p in reversed(user_predictions)]
    bmi_values = [p.bmi for p in reversed(user_predictions)]
    bp_values = [p.blood_pressure for p in reversed(user_predictions)]
    insulin_values = [p.insulin for p in reversed(user_predictions)]
    timestamps = [p.timestamp for p in reversed(user_predictions)]

    glucose_trend_msg = get_trend_message(glucose_values, "Glucose", timestamps)
    bmi_trend_msg = get_trend_message(bmi_values, "BMI", timestamps)
    bp_trend_msg = get_trend_message(bp_values, "Blood Pressure", timestamps)
    insulin_trend_msg = get_trend_message(insulin_values, "Insulin", timestamps)

    return render_template(
        "history.html",
        predictions=user_predictions,
        dates=dates,
        glucose_values=glucose_values,
        bmi_values=bmi_values,
        bp_values=bp_values,
        insulin_values=insulin_values,
        glucose_trend_msg=glucose_trend_msg,
        bmi_trend_msg=bmi_trend_msg,
        bp_trend_msg=bp_trend_msg,
        insulin_trend_msg=insulin_trend_msg
    )

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        # Validation: Check for empty inputs (extra safety)
        if not username or not password:
            flash("❗ Please fill in both username and password.", "warning")
            return redirect(url_for("register"))

        # Check if the username already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("⚠️ Username already exists. Please choose another one.", "danger")
            return redirect(url_for("register"))

        # Hash the password using bcrypt
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        # Create and save new user
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash("✅ Registration successful! You can now log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()
        user = User.query.filter_by(username=username).first()

        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash("✅ Login successful!", "success")
            return redirect(url_for("home"))
        else:
            flash("❌ Invalid username or password", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    session.pop('_flashes', None)
    logout_user()
    flash("🚪 You have been logged out.", "info")
    return redirect(url_for("login"))

OPENROUTER_API_KEY = "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"  # Replace with your OpenRouter key

@app.route('/chatgpt', methods=['POST'])
def chatgpt():
    data = request.get_json()
    user_message = data.get('message', '')

    # Load chat history from session or start new
    chat_history = session.get('chat_history', [])
    chat_history.append({"role": "user", "content": user_message})

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": chat_history
    }
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload
    )
    reply = response.json()["choices"][0]["message"]["content"]
    chat_history.append({"role": "assistant", "content": reply})

    # Limit history to last 20 messages
    chat_history = chat_history[-20:]
    session['chat_history'] = chat_history
    session.modified = True

    return jsonify({'reply': reply, 'history': chat_history})

if __name__ == "__main__":
    app.run(debug=True)




