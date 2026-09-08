import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "svm.pkl")
model = joblib.load(MODEL_PATH)

# Feature list matching the model's feature_names_in_
FEATURE_NAMES = [
    "age",
    "gender",
    "course",
    "study_hours",
    "class_attendance",
    "internet_access",
    "sleep_hours",
    "sleep_quality",
    "study_method",
    "facility_rating",
    "exam_difficulty",
]

# Categorical mappings if your model was trained on label-encoded values
CATEGORICAL_MAPPINGS = {
    "gender": {"Female": 0, "Male": 1, "Other": 2},
    "course": {
        "Computer Science": 0,
        "Engineering": 1,
        "Business": 2,
        "Medicine": 3,
        "Arts": 4,
        "Other": 5,
    },
    "internet_access": {"No": 0, "Yes": 1},
    "sleep_quality": {"Poor": 0, "Average": 1, "Good": 2},
    "study_method": {
        "Self Study": 0,
        "Group Study": 1,
        "Online Lectures": 2,
        "Coaching": 3,
    },
    "facility_rating": {"Low": 0, "Medium": 1, "High": 2},
    "exam_difficulty": {"Easy": 0, "Moderate": 1, "Hard": 2},
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance Predictor (SVR)</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
            --card-bg: rgba(255, 255, 255, 0.05);
            --card-border: rgba(255, 255, 255, 0.1);
            --card-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5), 0 0 30px rgba(99, 102, 241, 0.15);
            --input-bg: rgba(15, 23, 42, 0.6);
            --input-border: rgba(255, 255, 255, 0.15);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2.5rem 1rem;
            color: var(--text-main);
        }

        .container {
            width: 100%;
            max-width: 820px;
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 2.5rem;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            box-shadow: var(--card-shadow);
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            background: linear-gradient(to right, #ffffff, #c7d2fe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 1.25rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        label {
            font-size: 0.85rem;
            font-weight: 600;
            color: #cbd5e1;
            text-transform: capitalize;
        }

        input, select {
            background: var(--input-bg);
            border: 1px solid var(--input-border);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.2s ease-in-out;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
        }

        input:focus, select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.3);
        }

        select option {
            background-color: #1e293b;
            color: #ffffff;
        }

        .submit-btn {
            grid-column: 1 / -1;
            margin-top: 1.5rem;
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-hover) 100%);
            color: #ffffff;
            font-weight: 600;
            font-size: 1.05rem;
            padding: 1rem;
            border: none;
            border-radius: 14px;
            cursor: pointer;
            box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.4);
            transition: all 0.25s ease;
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 25px -5px rgba(99, 102, 241, 0.5);
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.5rem;
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 8px 30px rgba(99, 102, 241, 0.2);
            animation: fadeIn 0.4s ease-out;
        }

        .result-box h3 {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        .result-box .score {
            font-size: 2.25rem;
            font-weight: 700;
            color: #a5b4fc;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Academic Performance Predictor</h1>
            <p>Enter student academic parameters to predict score via SVR</p>
        </div>

        <form method="POST">
            <div class="form-grid">
                <!-- Numeric Inputs -->
                <div class="input-group">
                    <label>Age</label>
                    <input type="number" name="age" step="1" min="10" max="100" required value="{{ request.form.get('age', 20) }}">
                </div>

                <div class="input-group">
                    <label>Study Hours / Day</label>
                    <input type="number" name="study_hours" step="0.1" min="0" max="24" required value="{{ request.form.get('study_hours', 5.0) }}">
                </div>

                <div class="input-group">
                    <label>Class Attendance (%)</label>
                    <input type="number" name="class_attendance" step="0.1" min="0" max="100" required value="{{ request.form.get('class_attendance', 85.0) }}">
                </div>

                <div class="input-group">
                    <label>Sleep Hours / Day</label>
                    <input type="number" name="sleep_hours" step="0.1" min="0" max="24" required value="{{ request.form.get('sleep_hours', 7.0) }}">
                </div>

                <!-- Categorical Dropdowns -->
                <div class="input-group">
                    <label>Gender</label>
                    <select name="gender" required>
                        <option value="Female">Female</option>
                        <option value="Male">Male</option>
                        <option value="Other">Other</option>
                    </select>
                </div>

                <div class="input-group">
                    <label>Course</label>
                    <select name="course" required>
                        <option value="Computer Science">Computer Science</option>
                        <option value="Engineering">Engineering</option>
                        <option value="Business">Business</option>
                        <option value="Medicine">Medicine</option>
                        <option value="Arts">Arts</option>
                        <option value="Other">Other</option>
                    </select>
                </div>

                <div class="input-group">
                    <label>Internet Access</label>
                    <select name="internet_access" required>
                        <option value="Yes">Yes</option>
                        <option value="No">No</option>
                    </select>
                </div>

                <div class="input-group">
                    <label>Sleep Quality</label>
                    <select name="sleep_quality" required>
                        <option value="Good">Good</option>
                        <option value="Average">Average</option>
                        <option value="Poor">Poor</option>
                    </select>
                </div>

                <div class="input-group">
                    <label>Study Method</label>
                    <select name="study_method" required>
                        <option value="Self Study">Self Study</option>
                        <option value="Group Study">Group Study</option>
                        <option value="Online Lectures">Online Lectures</option>
                        <option value="Coaching">Coaching</option>
                    </select>
                </div>

                <div class="input-group">
                    <label>Facility Rating</label>
                    <select name="facility_rating" required>
                        <option value="High">High</option>
                        <option value="Medium">Medium</option>
                        <option value="Low">Low</option>
                    </select>
                </div>

                <div class="input-group">
                    <label>Exam Difficulty</label>
                    <select name="exam_difficulty" required>
                        <option value="Easy">Easy</option>
                        <option value="Moderate">Moderate</option>
                        <option value="Hard">Hard</option>
                    </select>
                </div>

                <button type="submit" class="submit-btn">Predict Score</button>
            </div>
        </form>

        {% if prediction is not none %}
        <div class="result-box">
            <h3>Predicted Performance Score</h3>
            <div class="score">{{ prediction }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            form_data = {}
            for col in FEATURE_NAMES:
                val = request.form.get(col)
                if col in CATEGORICAL_MAPPINGS:
                    form_data[col] = CATEGORICAL_MAPPINGS[col].get(val, 0)
                else:
                    form_data[col] = float(val)

            # Build DataFrame with exact feature order
            input_df = pd.DataFrame([form_data], columns=FEATURE_NAMES)
            raw_prediction = model.predict(input_df)[0]
            prediction = f"{raw_prediction:.2f}"
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
