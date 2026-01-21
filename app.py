from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# =======================
# Load ML model & encoders
# =======================
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
model = pickle.load(open(os.path.join(MODEL_DIR, "career_model.pkl"), "rb"))
edu_encoder = pickle.load(open(os.path.join(MODEL_DIR, "edu_encoder.pkl"), "rb"))
interest_encoder = pickle.load(open(os.path.join(MODEL_DIR, "interest_encoder.pkl"), "rb"))
role_encoder = pickle.load(open(os.path.join(MODEL_DIR, "role_encoder.pkl"), "rb"))
edu_encoder = pickle.load(open("model/edu_encoder.pkl", "rb"))
interest_encoder = pickle.load(open("model/interest_encoder.pkl", "rb"))
role_encoder = pickle.load(open("model/role_encoder.pkl", "rb"))

# =======================
# Interest mapping (safe for unseen labels)
# =======================
interest_map = {
    "ai": "AI",
    "cybersecurity": "Cybersecurity",
    "web development": "Data",
    "data": "Data"
}

# =======================
# Skill → Role mapping (rule-based override)
# =======================
skill_role_map = {
    "python": ["Data Scientist", "ML Engineer", "Data Analyst"],
    "ml": ["ML Engineer", "AI Researcher"],
    "sql": ["Data Analyst", "Data Scientist"],
    "java": ["Software Engineer", "Backend Developer"],
    "spring": ["Backend Developer"],
    "html": ["Web Developer"],
    "css": ["Web Developer"],
    "javascript": ["Frontend Developer", "Web Developer"]
}

# =======================
# Career explanations
# =======================
career_explanations = {
    "Data Analyst": "Your skills indicate strong analytical thinking and data handling ability.",
    "Data Scientist": "Your profile aligns with data analysis, statistics, and problem-solving.",
    "ML Engineer": "Your interest and skills suggest suitability for building intelligent systems.",
    "Software Engineer": "Your technical background supports software design and development.",
    "Backend Developer": "Your skills align with server-side logic and application architecture.",
    "Web Developer": "Your skills match frontend technologies and web development."
}

# =======================
# Career roadmaps
# =======================
career_roadmaps = {
    "Data Analyst": [
        "Learn SQL & Excel fundamentals",
        "Practice data visualization (Power BI / Tableau)",
        "Work on real-world datasets",
        "Apply for internships or entry-level roles"
    ],
    "ML Engineer": [
        "Strengthen Python & ML basics",
        "Learn deep learning frameworks",
        "Build ML projects",
        "Deploy models using Flask/FastAPI"
    ],
    "Software Engineer": [
        "Master DSA",
        "Learn backend frameworks",
        "Build full-stack projects",
        "Prepare for interviews"
    ],
    "Web Developer": [
        "Learn HTML, CSS, JavaScript",
        "Explore React",
        "Build responsive websites",
        "Deploy projects online"
    ]
}

# =======================
# Routes
# =======================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Form inputs
    education = request.form["education"]
    skills = request.form["skills"]
    experience = int(request.form["experience"])
    career_gap = int(request.form["career_gap"])

    # Safe interest handling
    raw_interest = request.form["interest"].strip().lower()
    interest = interest_map.get(raw_interest, "Data")

    # Feature engineering
    skill_list = skills.lower().split()
    skill_count = len(skill_list)

    # Encode categorical features
    edu_encoded = edu_encoder.transform([education])[0]
    interest_encoded = interest_encoder.transform([interest])[0]

    # ML prediction
    prediction = model.predict([
        [edu_encoded, skill_count, experience, career_gap, interest_encoded]
    ])
    ml_career = role_encoder.inverse_transform(prediction)[0]

    # Skill-based override logic
    matched_roles = []
    for skill in skill_list:
        if skill in skill_role_map:
            matched_roles.extend(skill_role_map[skill])

    career = (
        max(set(matched_roles), key=matched_roles.count)
        if matched_roles else ml_career
    )

    # Confidence score (heuristic)
    confidence = min(95, 50 + (skill_count * 10) + (experience * 5))

    # Explanation & roadmap
    explanation = career_explanations.get(
        career,
        "This career matches your skills, education, and interests."
    )
    roadmap = career_roadmaps.get(career, [])

    return render_template(
        "result.html",
        career=career,
        confidence=confidence,
        explanation=explanation,
        roadmap=roadmap
    )


# =======================
# App entry point
# =======================
if __name__ == "__main__":
    app.run(debug=True)
