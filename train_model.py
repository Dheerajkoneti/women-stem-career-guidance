import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("data/women_stem_career_dataset.csv")

# Feature engineering
df['skill_count'] = df['skills'].apply(lambda x: len(x.split()))

# Encode categorical columns
edu_encoder = LabelEncoder()
interest_encoder = LabelEncoder()
role_encoder = LabelEncoder()

df['education'] = edu_encoder.fit_transform(df['education'])
df['interest'] = interest_encoder.fit_transform(df['interest'])
df['preferred_role'] = role_encoder.fit_transform(df['preferred_role'])

X = df[['education', 'skill_count', 'experience', 'career_gap', 'interest']]
y = df['preferred_role']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model and encoders
pickle.dump(model, open("model/career_model.pkl", "wb"))
pickle.dump(edu_encoder, open("model/edu_encoder.pkl", "wb"))
pickle.dump(interest_encoder, open("model/interest_encoder.pkl", "wb"))
pickle.dump(role_encoder, open("model/role_encoder.pkl", "wb"))

print("✅ Model trained and saved successfully")
